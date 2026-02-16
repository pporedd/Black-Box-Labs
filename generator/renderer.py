"""
Playwright headless renderer — HTML → 500×500 PNG.

Renders a Jinja2 template with the calculated scene state,
then captures a screenshot using Playwright's headless Chromium.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Optional

from jinja2 import Environment, FileSystemLoader
from PIL import Image
import io

from generator.state import SceneState
from config import settings

# Jinja2 environment pointing at our templates directory
_TEMPLATE_DIR = Path(__file__).parent / "templates"
_jinja_env = Environment(loader=FileSystemLoader(str(_TEMPLATE_DIR)))


def _build_html(scene: SceneState) -> str:
    """Render the Jinja2 template with scene state variables."""
    template = _jinja_env.get_template("counting_scene.html")
    return template.render(
        canvas_width=scene.canvas_width,
        canvas_height=scene.canvas_height,
        background_color=scene.params.background_color,
        item_color=scene.params.item_color,
        item_size=scene.params.item_size_px,
        item_half_size=scene.params.item_size_px // 2,
        items=scene.items,
        distractors=scene.distractors,
    )


async def _render_async(html: str, width: int, height: int) -> Image.Image:
    """Use Playwright to screenshot rendered HTML."""
    from playwright.async_api import async_playwright

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page(viewport={"width": width, "height": height})
        await page.set_content(html, wait_until="load")
        screenshot_bytes = await page.screenshot(type="png")
        await browser.close()

    return Image.open(io.BytesIO(screenshot_bytes)).convert("RGB")


async def render_scene_async(scene: SceneState) -> Image.Image:
    """
    Render a SceneState to a PIL Image (async version).

    Pipeline: SceneState → Jinja2 HTML → Playwright screenshot → PIL.Image
    """
    html = _build_html(scene)
    return await _render_async(html, scene.canvas_width, scene.canvas_height)


def render_scene(scene: SceneState) -> Image.Image:
    """
    Render a SceneState to a PIL Image (sync wrapper).

    Handles the asyncio event loop for callers that aren't async.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        # We're inside an existing event loop (e.g., Streamlit)
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            future = pool.submit(asyncio.run, render_scene_async(scene))
            return future.result()
    else:
        return asyncio.run(render_scene_async(scene))


def render_scene_to_file(scene: SceneState, path: str | Path) -> Path:
    """Render and save to disk. Returns the output path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    img = render_scene(scene)
    img.save(str(path))
    return path
