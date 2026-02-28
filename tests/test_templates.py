"""Template compilation and render smoke tests.

Catches Jinja2 TemplateSyntaxError (e.g. corrupted inline scripts in macros
like share_buttons in _macros.html) before deploy. Run with: make test (or
PYTHONPATH=src pytest tests/test_templates.py).

Add any template that contains fragile inline JS/Jinja to
TEMPLATES_USING_SHARE_BUTTONS or add a new parametrized list.
"""

from __future__ import annotations

from pathlib import Path

import pytest

# Templates that use share_buttons (and its inline script). Compiling them
# catches syntax errors in _macros.html (e.g. broken getElementById, bad {{ }}).
TEMPLATES_USING_SHARE_BUTTONS = ["_macros.html", "home.html"]


def _templates_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "src" / "ilga_graph" / "templates"


@pytest.mark.parametrize("template_name", TEMPLATES_USING_SHARE_BUTTONS)
def test_template_compiles_without_syntax_error(template_name: str) -> None:
    """Loading the template compiles it and any included macros; raises on syntax error."""
    from jinja2 import Environment, FileSystemLoader

    templates_dir = _templates_dir()
    assert templates_dir.is_dir(), f"Missing templates dir: {templates_dir}"
    env = Environment(loader=FileSystemLoader(str(templates_dir)))
    env.get_template(template_name)


def test_share_buttons_macro_renders_without_error() -> None:
    """Render the share_buttons macro with minimal context to catch runtime Jinja/JS issues."""
    from jinja2 import Environment, FileSystemLoader

    templates_dir = _templates_dir()
    env = Environment(loader=FileSystemLoader(str(templates_dir)))
    t = env.get_template("_macros.html")
    # Render share_buttons fragment (full variant uses macro defaults for section_id etc).
    out = t.module.share_buttons("full")
    assert "share" in out.lower() or "copy" in out.lower()
    out_sidebar = t.module.share_buttons("sidebar", "test-section", "constituent")
    assert "share" in out_sidebar.lower() or "copy" in out_sidebar.lower()
