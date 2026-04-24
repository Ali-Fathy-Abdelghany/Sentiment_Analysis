from __future__ import annotations

# Backward-compatible shim: keeps old streamlit path working after src-first refactor.
from clean_social.apps.streamlit_app import render_app

render_app()
