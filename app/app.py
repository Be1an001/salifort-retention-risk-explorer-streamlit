from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.pages import (  # noqa: E402
    eda_patterns,
    explainability,
    manager_action_view,
    methods_limitations,
    model_threshold_lab,
    overview,
    pace_navigator,
    workforce_explorer,
)

st.set_page_config(
    page_title="Salifort Motors Retention Risk Explorer",
    page_icon=":material/insights:",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main .block-container {
        padding-top: 1.8rem;
        padding-bottom: 2.5rem;
    }
    [data-testid="stSidebar"] .block-container {
        padding-top: 1.4rem;
    }
    .app-kicker {
        color: #0f766e;
        font-size: 0.85rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown('<div class="app-kicker">Attrition Risk Review</div>', unsafe_allow_html=True)
    st.markdown("## Salifort Motors")
    st.caption("Operational HR Analytics Decision App")
    st.divider()
    st.caption(
        "This app brings together the cleaned HR dataset, generated model outputs, "
        "and project visuals for a practical review of attrition risk."
    )

navigation = st.navigation(
    {
        "Start Here": [
            st.Page(
                overview.render,
                title="Overview",
                icon=":material/home:",
                default=True,
                url_path="overview",
            ),
            st.Page(
                pace_navigator.render,
                title="PACE Navigator",
                icon=":material/map:",
                url_path="pace-navigator",
            ),
        ],
        "Interactive Analysis": [
            st.Page(
                workforce_explorer.render,
                title="Workforce Explorer",
                icon=":material/group:",
                url_path="workforce-explorer",
            ),
            st.Page(
                eda_patterns.render,
                title="EDA & Patterns",
                icon=":material/analytics:",
                url_path="eda-patterns",
            ),
        ],
        "Decision Support": [
            st.Page(
                model_threshold_lab.render,
                title="Model & Threshold Lab",
                icon=":material/tune:",
                url_path="model-threshold-lab",
            ),
            st.Page(
                explainability.render,
                title="Explainability",
                icon=":material/visibility:",
                url_path="explainability",
            ),
            st.Page(
                manager_action_view.render,
                title="Manager Action View",
                icon=":material/badge:",
                url_path="manager-action-view",
            ),
            st.Page(
                methods_limitations.render,
                title="Methods & Limitations",
                icon=":material/rule:",
                url_path="methods-limitations",
            ),
        ],
    }
)

navigation.run()
