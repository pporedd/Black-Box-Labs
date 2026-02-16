# 🔬 Black Box Labs
**Autonomous Behavioral Boundary Mapping for Vision-Language Models**

An experimental framework that uses an LLM-powered Scientist Agent (Nemotron-4 340B via OpenRouter) to autonomously probe and map the capability boundaries of a Vision-Language Model (Moondream2).

![Streamlit Dashboard](https://github.com/user-attachments/assets/placeholder)

## 🌟 Overview

This system autonomously discovers "islands of competence" and "cliffs of failure" in VLMs by:
1.  **Generating Parametric Stimuli**: Creating synthetic images (HTML/CSS + Playwright) controlled by variables like `item_count`, `occlusion`, `spacing`, etc.
2.  **Hypothesis Testing**: The Scientist Agent proposes experiments (e.g., "Does increasing occlusion cause the VLM to undercount?").
3.  **Binary Search**: Efficiently finding the exact tipping point where the model fails.
4.  **Isomorphic Validation**: verifying findings across random variations to distinguish true limits from brittle failures.

## 🛠️ Architecture

-   **Agent**: `nvidia/nemotron-4-340b-instruct` (via OpenRouter) — Proposes hypotheses and tools.
-   **Subject**: `vikhyatk/moondream2` (local execution) — The VLM being tested.
-   **Generator**: Jinja2 + Playwright — Renders pixel-perfect scenes.
-   **Dashboard**: Streamlit — Visualizes the active experiment, attention maps, and boundary curves.

## 📦 Installation

### Prerequisites
-   **Python 3.11** (Required for `moondream` compatibility).
-   **uv** (Recommended package manager).
-   **LibVIPS** (Required for image processing).

### 1. Install LibVIPS (Windows)
1.  Download `vips-dev-w64-all-8.18.x.zip` from [libvips releases](https://github.com/libvips/libvips/releases).
2.  Extract to a known location, e.g., `C:\Users\YourUser\Documents\vips-dev-8.18`.
3.  Add the `bin` folder to your System PATH.

### 2. Setup Python Environment
```powershell
# Clone repository
git clone https://github.com/yourusername/black-box-labs.git
cd black-box-labs

# Install dependencies with uv
uv sync
```

### 3. Install Playwright Browsers
Required for the image generator:
```powershell
uv run playwright install
```

### 4. Configuration
Create a `.env` file in the root directory:
```env
OPENROUTER_API_KEY=sk-or-your-key-here
```

## 🚀 Usage

Run the dashboard:
```powershell
uv run streamlit run app.py
```

The app will launch at `http://localhost:8501`.

1.  Click **▶ Start** to begin the autonomous search.
2.  Monitor the **Hypothesis Log** to see the agent's reasoning.
3.  View the **Behavioral Boundary Map** to see pass/fail rates evolve.

## 🧩 Project Structure

-   `agent/`: The Scientist Agent logic and prompts.
-   `evaluator/`: Moondream2 model wrapper and inference code.
-   `generator/`: Parametric scene generation (templates + renderer).
-   `orchestrator.py`: Main loop connecting Agent, Generator, and Evaluator.
-   `app.py`: Streamlit dashboard entry point.

## ⚠️ Troubleshooting

**`OSError: cannot load library 'libvips-42.dll'`**
-   Ensure `vips-dev-8.18\bin` is in your PATH.
-   On Windows, `app.py` attempts to automatically register this path if found in `Documents`. Update `app.py` if your path differs.
