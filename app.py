"""
Streamlit Dashboard — VLM Behavioral Boundary Explorer

Three-panel layout:
  A) Input Stimulus — Generated image + active variable listing
  B) Attention Map — Heatmap overlay from the VLM's attention layers
  C) Boundary Histogram — Pass rate vs. variable value (step-plot)

Sidebar: Session controls (Start / Pause / Export), model status, hypothesis log.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from PIL import Image
import asyncio
import sys
import os

# Fix for Playwright on Windows: Force ProactorEventLoop
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    
    # Fix for LibVIPS (Moondream dependency)
    # Try to locate vips-dev-8.18/bin or similar in Documents
    # Hardcoded path based on user logs, but could be made dynamic
    vips_bin = r"C:\Users\tardi\OneDrive\Documents\vips-dev-8.18\bin"
    if os.path.exists(vips_bin):
        if hasattr(os, "add_dll_directory"):
            # Must keep reference to prevent GC!
            global _vips_dir_handle
            _vips_dir_handle = os.add_dll_directory(vips_bin)
        os.environ["PATH"] = vips_bin + os.pathsep + os.environ["PATH"]


# ── Page Config ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="Black Box Labs — VLM Boundary Explorer",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    .stApp {
        font-family: 'Inter', sans-serif;
    }

    .main-header {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        color: white;
    }

    .main-header h1 {
        margin: 0;
        font-size: 1.8rem;
        font-weight: 700;
        background: linear-gradient(90deg, #a78bfa, #60a5fa, #34d399);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .main-header p {
        margin: 0.3rem 0 0 0;
        color: #94a3b8;
        font-size: 0.9rem;
    }

    .panel-card {
        background: #1e1e2e;
        border: 1px solid #313244;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
    }

    .panel-title {
        color: #cdd6f4;
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.75rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    .status-badge {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
    }

    .status-running { background: #166534; color: #4ade80; }
    .status-paused { background: #854d0e; color: #fbbf24; }
    .status-idle { background: #1e3a5f; color: #60a5fa; }

    .finding-card {
        background: #181825;
        border-left: 3px solid;
        padding: 0.6rem 0.8rem;
        margin: 0.4rem 0;
        border-radius: 0 6px 6px 0;
        font-size: 0.85rem;
    }

    .finding-breakpoint { border-color: #f38ba8; }
    .finding-spurious { border-color: #f9e2af; }
    .finding-stable { border-color: #a6e3a1; }
    .finding-brittle { border-color: #fab387; }

    .param-table {
        width: 100%;
        font-size: 0.8rem;
    }

    .param-table td {
        padding: 0.25rem 0.4rem;
        border-bottom: 1px solid #313244;
    }

    .param-name { color: #89b4fa; font-family: monospace; }
    .param-value { color: #cdd6f4; text-align: right; }
</style>
""", unsafe_allow_html=True)


# ── Session State Initialization ─────────────────────────────────────

def init_session_state():
    """Initialize all Streamlit session state variables."""
    defaults = {
        "orchestrator": None,
        "is_running": False,
        "is_paused": False,
        "step_results": [],
        "hypothesis_log": [],
        "current_image": None,
        "attention_overlay": None,
        "boundary_data": {},  # {variable: [(value, rate), ...]}
        "model_loaded": False,
        "error_message": None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


init_session_state()


# ── Header ───────────────────────────────────────────────────────────

st.markdown("""
<div class="main-header">
    <h1>🔬 Black Box Labs</h1>
    <p>Autonomous Behavioral Boundary Mapping for Vision-Language Models</p>
</div>
""", unsafe_allow_html=True)


# ── Sidebar ──────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### ⚙️ Session Controls")

    # Status indicator
    if st.session_state.is_running:
        if st.session_state.is_paused:
            st.markdown('<span class="status-badge status-paused">⏸ Paused</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="status-badge status-running">● Running</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-badge status-idle">○ Idle</span>', unsafe_allow_html=True)

    st.divider()

    # Control buttons
    col1, col2 = st.columns(2)
    with col1:
        start_btn = st.button(
            "▶ Start" if not st.session_state.is_running else "⏹ Stop",
            use_container_width=True,
            type="primary",
        )
    with col2:
        pause_btn = st.button(
            "⏸ Pause" if not st.session_state.is_paused else "▶ Resume",
            use_container_width=True,
            disabled=not st.session_state.is_running,
        )

    step_btn = st.button("⏭ Single Step", use_container_width=True)

    export_btn = st.button("📥 Export Session", use_container_width=True)

    st.divider()

    # Model & Config
    st.markdown("### 🤖 Model Config")
    st.caption(f"**Subject:** Moondream2")
    st.caption(f"**Agent:** Nemotron Ultra")
    st.caption(f"**Canvas:** 500×500px")
    st.caption(f"**Isomorphisms:** 5")
    st.caption(f"**Temp:** 0.0 (deterministic)")

    if st.session_state.model_loaded:
        st.success("Model loaded ✓", icon="🟢")
    else:
        st.info("Model not loaded", icon="⚪")

    st.divider()

    # Hypothesis Log
    st.markdown("### 📋 Hypothesis Log")
    if st.session_state.hypothesis_log:
        for i, entry in enumerate(reversed(st.session_state.hypothesis_log[-10:])):
            with st.expander(f"Step {len(st.session_state.hypothesis_log) - i}", expanded=i == 0):
                st.markdown(entry)
    else:
        st.caption("No hypotheses yet. Click Start to begin.")


# ── Main Content — Three Panels ──────────────────────────────────────

panel_a, panel_b, panel_c = st.columns([1, 1, 1.2])

# ── Panel A: Input Stimulus ──────────────────────────────────────────
with panel_a:
    st.markdown('<div class="panel-title">🖼️ Input Stimulus</div>', unsafe_allow_html=True)

    if st.session_state.current_image is not None:
        st.image(
            st.session_state.current_image,
            caption="Latest generated stimulus",
            use_container_width=True,
        )
    else:
        # Placeholder
        placeholder_html = """
        <div style="
            width: 100%; aspect-ratio: 1;
            background: linear-gradient(135deg, #1e1e2e, #313244);
            border: 2px dashed #45475a;
            border-radius: 8px;
            display: flex; align-items: center; justify-content: center;
            color: #6c7086; font-size: 0.9rem;
        ">
            Click "Start" to generate stimuli
        </div>
        """
        st.markdown(placeholder_html, unsafe_allow_html=True)

    # Active variables table
    st.markdown('<div class="panel-title" style="margin-top:1rem;">📊 Active Variables</div>', unsafe_allow_html=True)

    if st.session_state.orchestrator:
        params = st.session_state.orchestrator.state.params.to_dict()
    else:
        from generator.state import ModifiableParams
        params = ModifiableParams().to_dict()

    # Display as a clean table
    param_rows = ""
    for k, v in params.items():
        param_rows += f'<tr><td class="param-name">{k}</td><td class="param-value">{v}</td></tr>'

    st.markdown(f"""
    <table class="param-table">
        {param_rows}
    </table>
    """, unsafe_allow_html=True)

# ── Panel B: Attention Map ───────────────────────────────────────────
with panel_b:
    st.markdown('<div class="panel-title">🧠 Attention Map</div>', unsafe_allow_html=True)

    if st.session_state.attention_overlay is not None:
        st.image(
            st.session_state.attention_overlay,
            caption="Model attention heatmap overlay",
            use_container_width=True,
        )
    else:
        placeholder_html = """
        <div style="
            width: 100%; aspect-ratio: 1;
            background: linear-gradient(135deg, #1a1a2e, #16213e);
            border: 2px dashed #45475a;
            border-radius: 8px;
            display: flex; align-items: center; justify-content: center;
            color: #6c7086; font-size: 0.9rem;
        ">
            Attention map will appear here
        </div>
        """
        st.markdown(placeholder_html, unsafe_allow_html=True)

    # Attention stats
    st.markdown('<div class="panel-title" style="margin-top:1rem;">📈 Attention Stats</div>', unsafe_allow_html=True)

    if st.session_state.orchestrator and st.session_state.orchestrator.state.latest_eval:
        eval_r = st.session_state.orchestrator.state.latest_eval
        st.markdown(f"**Answer:** `{eval_r.raw_answer}`")
        st.markdown(f"**Parsed:** `{eval_r.parsed_answer}`")
        st.markdown(f"**Ground Truth:** `{eval_r.ground_truth}`")
        st.markdown(f"**Result:** {'✅ Pass' if eval_r.passed else '❌ Fail'}")
    else:
        st.caption("No evaluation results yet.")

# ── Panel C: Boundary Histogram ──────────────────────────────────────
with panel_c:
    st.markdown('<div class="panel-title">📉 Behavioral Boundary Map</div>', unsafe_allow_html=True)

    if st.session_state.boundary_data:
        # Create tabs for each variable
        variables = list(st.session_state.boundary_data.keys())
        if variables:
            selected_var = st.selectbox(
                "Variable", variables,
                label_visibility="collapsed",
            )
            data = st.session_state.boundary_data[selected_var]

            if data:
                values = [d[0] for d in data]
                rates = [d[1] for d in data]

                # Color-code by rate
                colors = []
                for r in rates:
                    if r >= 0.8:
                        colors.append("#a6e3a1")  # green
                    elif r >= 0.4:
                        colors.append("#f9e2af")  # yellow
                    else:
                        colors.append("#f38ba8")  # red

                fig = go.Figure()

                # Step plot
                fig.add_trace(go.Scatter(
                    x=values,
                    y=rates,
                    mode="lines+markers",
                    line=dict(
                        shape="hv",
                        color="#89b4fa",
                        width=2,
                    ),
                    marker=dict(
                        size=10,
                        color=colors,
                        line=dict(color="#1e1e2e", width=2),
                    ),
                    hovertemplate=(
                        "<b>%{x}</b><br>"
                        "Pass Rate: %{y:.0%}<br>"
                        "<extra></extra>"
                    ),
                ))

                # Threshold lines
                fig.add_hline(y=0.8, line_dash="dot", line_color="#a6e3a1", opacity=0.4,
                              annotation_text="Stable", annotation_position="left")
                fig.add_hline(y=0.4, line_dash="dot", line_color="#f9e2af", opacity=0.4,
                              annotation_text="Brittle", annotation_position="left")
                fig.add_hline(y=0.2, line_dash="dot", line_color="#f38ba8", opacity=0.4,
                              annotation_text="Breakpoint", annotation_position="left")

                fig.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(30,30,46,0.5)",
                    xaxis_title=selected_var,
                    yaxis_title="Pass Rate",
                    yaxis=dict(range=[-0.05, 1.05]),
                    margin=dict(l=50, r=20, t=10, b=50),
                    height=350,
                    font=dict(family="Inter", size=12),
                )

                st.plotly_chart(fig, use_container_width=True)

                # Summary stats
                avg_rate = sum(rates) / len(rates)
                min_rate = min(rates)
                max_rate = max(rates)
                st.markdown(
                    f"**Avg Rate:** {avg_rate:.0%} · "
                    f"**Min:** {min_rate:.0%} · "
                    f"**Max:** {max_rate:.0%} · "
                    f"**Points:** {len(rates)}"
                )
    else:
        # Empty state
        st.markdown("""
        <div style="
            height: 350px;
            background: linear-gradient(135deg, #1e1e2e, #181825);
            border: 2px dashed #45475a;
            border-radius: 8px;
            display: flex; align-items: center; justify-content: center;
            flex-direction: column; gap: 0.5rem;
            color: #6c7086;
        ">
            <span style="font-size: 2rem;">📊</span>
            <span>Boundary data will populate as the search runs</span>
        </div>
        """, unsafe_allow_html=True)

    # Findings summary
    st.markdown('<div class="panel-title" style="margin-top:1rem;">🔍 Findings</div>', unsafe_allow_html=True)

    if st.session_state.orchestrator and st.session_state.orchestrator.agent.findings:
        for f in st.session_state.orchestrator.agent.findings:
            css_class = f"finding-{f.transition_type}"
            emoji = {
                "breakpoint": "🔴",
                "spurious_resolution": "🟡",
                "stable": "🟢",
                "brittle": "🟠",
            }.get(f.transition_type, "⚪")

            st.markdown(f"""
            <div class="finding-card {css_class}">
                {emoji} <b>{f.variable}={f.value}</b> — {f.transition_type}
                <br><small>Rate: {f.success_rate:.0%} · Confidence: {f.confidence:.0%}</small>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.caption("No findings yet.")


# ── Button Logic (after layout) ──────────────────────────────────────

def initialize_system():
    """Initialize the full system (load models, create orchestrator)."""
    with st.spinner("Loading Moondream2 on CUDA..."):
        try:
            from evaluator.vlm import MoondreamEvaluator
            from agent.client import NemotronClient
            from agent.scientist import ScientistAgent
            from orchestrator import BoundarySearchOrchestrator

            evaluator = MoondreamEvaluator()
            client = NemotronClient()
            agent = ScientistAgent(client=client)
            orch = BoundarySearchOrchestrator(evaluator=evaluator, agent=agent)

            st.session_state.orchestrator = orch
            
            # Generate initial state for UI visibility
            from generator.state import calculate_scene
            from generator.renderer import render_scene
            
            initial_scene = calculate_scene(orch.state.params, seed=0)
            initial_image = render_scene(initial_scene)
            
            orch.state.latest_scene = initial_scene
            orch.state.latest_image = initial_image
            st.session_state.current_image = initial_image

            st.session_state.model_loaded = True
            st.session_state.is_running = True

        except Exception as e:
            import traceback
            traceback.print_exc()
            msg = str(e)
            if not msg:
                msg = f"{type(e).__name__}: {e}"
            st.session_state.error_message = msg


def execute_step():
    """Execute a single orchestration step and update UI state."""
    orch = st.session_state.orchestrator
    if orch is None:
        return

    try:
        result = orch.step()

        # Update UI state
        if orch.state.latest_image:
            st.session_state.current_image = orch.state.latest_image

        if orch.state.latest_attention_map is not None and orch.state.latest_image:
            from evaluator.attention import overlay_heatmap
            st.session_state.attention_overlay = overlay_heatmap(
                orch.state.latest_image,
                orch.state.latest_attention_map,
            )

        # Update boundary data
        for step in orch.session.steps:
            var = step.variable_name
            if var not in st.session_state.boundary_data:
                st.session_state.boundary_data[var] = []
            point = (step.variable_value, step.success_rate)
            if point not in st.session_state.boundary_data[var]:
                st.session_state.boundary_data[var].append(point)

        # Update hypothesis log
        if result["type"] == "message":
            st.session_state.hypothesis_log.append(result["content"])
        elif result["type"] == "tool_execution":
            log_entry = f"**Reasoning:** {result.get('reasoning', 'N/A')}\n\n"
            for tr in result.get("tool_results", []):
                log_entry += f"🔧 `{tr['tool']}({tr['args']})` → "
                r = tr["result"]
                if "success_rate" in r:
                    log_entry += f"Rate: {r['success_rate']:.0%}"
                elif "boundary_found" in r:
                    if r["boundary_found"]:
                        log_entry += f"Boundary at {r['boundary_value']} ({r['transition_type']})"
                    else:
                        log_entry += "No transition found"
                else:
                    log_entry += json.dumps(r)[:100]
                log_entry += "\n\n"
            st.session_state.hypothesis_log.append(log_entry)

        if result.get("done"):
            st.session_state.is_running = False

    except Exception as e:
        import traceback
        traceback.print_exc()
        msg = str(e)
        if not msg:
            msg = f"{type(e).__name__}: {e}"
        st.session_state.error_message = msg
        st.session_state.is_running = False


# Handle button clicks
if start_btn:
    if not st.session_state.is_running:
        initialize_system()
    else:
        st.session_state.is_running = False
        if st.session_state.orchestrator:
            st.session_state.orchestrator.state.is_running = False
    st.rerun()

if pause_btn:
    st.session_state.is_paused = not st.session_state.is_paused
    if st.session_state.orchestrator:
        st.session_state.orchestrator.state.is_paused = st.session_state.is_paused
    st.rerun()

if step_btn:
    if st.session_state.orchestrator is None:
        initialize_system()
    if st.session_state.orchestrator:
        execute_step()
        st.rerun()

if export_btn and st.session_state.orchestrator:
    st.session_state.orchestrator.session.save_summary()
    st.sidebar.success(
        f"Exported to `sessions/{st.session_state.orchestrator.session.session_id}_summary.json`"
    )

# Show errors
if st.session_state.error_message:
    st.error(f"⚠️ {st.session_state.error_message}")
    st.session_state.error_message = None

# Auto-step if running
if st.session_state.is_running and not st.session_state.is_paused:
    if st.session_state.orchestrator:
        execute_step()
        time.sleep(0.5)  # Brief delay between auto-steps
        st.rerun()
