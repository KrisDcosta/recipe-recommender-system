"""Browser demo for the recommendation API."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

router = APIRouter()
_ASSET_DIR = Path(__file__).with_name("demo_assets")


@lru_cache(maxsize=1)
def _render_demo_html() -> str:
    """Inline the handoff UI assets so /demo remains a single served document."""
    html = (_ASSET_DIR / "index.html").read_text(encoding="utf-8")
    for name in ("rr-data.jsx", "rr-components.jsx", "rr-tabs.jsx", "rr-app.jsx"):
        src_tag = f'<script type="text/babel" src="{name}"></script>'
        code = (_ASSET_DIR / name).read_text(encoding="utf-8")
        html = html.replace(src_tag, f'<script type="text/babel">\n{code}\n</script>')
    return html


@router.get("/demo", response_class=HTMLResponse)
def demo() -> str:
    """Serve the integrated single-page recommender demo."""
    return _render_demo_html()
