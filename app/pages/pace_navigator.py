from __future__ import annotations

import streamlit as st

from app.viewmodels import build_navigator_page_context


def _render_card(title: str, body: str, tone: str = "neutral") -> None:
    tone_styles = {
        "neutral": ("#E5E7EB", "#111827"),
        "primary": ("#0F766E", "#0F766E"),
        "warning": ("#B45309", "#92400E"),
    }
    border_color, title_color = tone_styles.get(tone, tone_styles["neutral"])
    st.markdown(
        f"""
        <div style="
            border: 1px solid {border_color};
            border-radius: 16px;
            padding: 1rem 1rem 0.95rem 1rem;
            background: linear-gradient(180deg, #FFFFFF 0%, #F8FAFC 100%);
            height: 100%;
        ">
            <div style="
                color: {title_color};
                font-size: 0.78rem;
                font-weight: 700;
                letter-spacing: 0.08em;
                text-transform: uppercase;
                margin-bottom: 0.55rem;
            ">{title}</div>
            <div style="color: #111827; line-height: 1.55;">{body}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render() -> None:
    context = build_navigator_page_context()
    identity_card = context["project_identity_card"]
    public_truth = context["public_model_truth_card"]

    st.title(context["page_title"])
    st.caption(context["page_caption"])

    st.markdown(
        "This page is a governed navigator shell. It explains the current project contract, "
        "preserved public truth, and known drift without changing any model, artifact, or runtime behavior."
    )

    st.subheader("What This Navigator Is")
    top_cols = st.columns([1.15, 0.85], gap="large")
    with top_cols[0]:
        _render_card(identity_card["title"], identity_card["summary"], tone="primary")
    with top_cols[1]:
        st.markdown("**Navigator highlights**")
        st.markdown("\n".join(f"- {item}" for item in identity_card["highlights"]))
        st.caption(
            "Supporting truth IDs: "
            + ", ".join(identity_card["supporting_truth_ids"])
        )

    st.subheader("Current Public Truth")
    metric_cols = st.columns(3)
    metric_cols[0].metric("Public Model", public_truth["model_name"])
    metric_cols[1].metric("Selected Threshold", public_truth["selected_threshold"])
    metric_cols[2].metric(
        "Preserve In Upgrade",
        "Yes" if public_truth["preserve_in_upgrade"] else "No",
    )
    st.info(public_truth["summary"])
    st.caption(f"Authority rule: {public_truth['authority_rule']}")

    st.subheader("Runtime Governance")
    governance_cols = st.columns(2, gap="large")
    for column, card in zip(governance_cols, context["runtime_governance_cards"]):
        with column:
            _render_card(card["title"], card["summary"], tone=card["tone"])
            st.caption(card["authority_rule"])

    st.subheader("Known Drift To Preserve And Explain")
    st.caption(
        "The navigator treats drift as governed context that should be surfaced and preserved, not flattened away."
    )
    for drift_card in context["drift_highlight_cards"]:
        with st.expander(
            f"{drift_card['severity']} severity: {drift_card['title']}",
            expanded=False,
        ):
            st.markdown(f"**Status:** {drift_card['status']}")
            st.markdown(f"**User-visible risk:** {drift_card['user_visible_risk']}")
            st.markdown(
                f"**Upgrade handling rule:** {drift_card['upgrade_handling_rule']}"
            )

    st.subheader("PACE Workflow Map")
    st.caption(context["pace_spine_note"])
    phase_cols = st.columns(2, gap="large")
    for index, phase_card in enumerate(context["pace_phase_cards"]):
        with phase_cols[index % 2]:
            _render_card(
                f"{phase_card['phase_title']}",
                (
                    f"{phase_card['phase_goal']}<br><br>"
                    f"<strong>In this repo:</strong> {phase_card['portfolio_meaning']}"
                ),
                tone="neutral",
            )
            st.markdown(
                f"**App pages:** {', '.join(phase_card['app_pages']) if phase_card['app_pages'] else 'None'}"
            )
            st.markdown(
                f"**Known drifts:** {', '.join(phase_card['known_drifts']) if phase_card['known_drifts'] else 'None'}"
            )
            st.caption(phase_card["future_navigator_use"])

    st.subheader("Where To Go In The App")
    st.caption(
        "These deterministic examples preview how later navigator actions can route users through the existing app."
    )
    for item in context["topic_recommendations"]:
        with st.container(border=True):
            st.markdown(f"**Topic:** {item['topic']}")
            st.markdown(
                f"**Recommended page:** `{item['recommended_page_title']}` "
                f"(`/{item['recommended_page_route']}`)"
            )
            st.markdown(
                f"**Supporting phase:** {str(item['supporting_phase']).capitalize()}"
            )
            st.caption(item["reason"])
