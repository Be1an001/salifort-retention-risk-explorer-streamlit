from __future__ import annotations

import pandas as pd
import streamlit as st

from app.viewmodels import (
    build_navigator_page_context,
    build_navigator_topic_drilldown,
    get_drift_records,
)


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


def _render_source_table(source_records: list[dict[str, object]]) -> None:
    if not source_records:
        st.info("No source records are linked to the selected topic.")
        return

    table_rows = []
    for item in source_records:
        consumer_pages = item["consumer_pages"]
        canonical_scope = item["canonical_scope"]
        runtime_scope = item["runtime_scope"]
        table_rows.append(
            {
                "Role": str(item["role"]).capitalize(),
                "Source ID": item["source_id"],
                "Title": item["title"],
                "Layer": item["repo_layer"],
                "Authority": item["authority_level"],
                "Path": item["path"],
                "Consumers": ", ".join(consumer_pages) if consumer_pages else "None",
                "Canonical Scope": ", ".join(canonical_scope) if canonical_scope else "None",
                "Runtime Scope": ", ".join(runtime_scope) if runtime_scope else "None",
            }
        )

    st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)

    for item in source_records:
        with st.expander(f"{str(item['role']).capitalize()} source detail: {item['title']}"):
            st.markdown(f"**Source ID:** `{item['source_id']}`")
            st.markdown(f"**Kind:** `{item['source_kind']}`")
            st.markdown(f"**Path:** `{item['path']}`")
            st.markdown(f"**Notes:** {item['notes']}")


def _render_drift_explorer(drift_records: list[dict[str, object]]) -> None:
    if not drift_records:
        st.info("No drifts match the current filter selection.")
        return

    for item in drift_records:
        with st.expander(
            f"{item['severity']} severity | {item['status']} | {item['title']}",
            expanded=False,
        ):
            st.markdown(f"**Drift ID:** `{item['drift_id']}`")
            st.markdown(f"**Canonical side:** {item['canonical_side']}")
            st.markdown(f"**Current side:** {item['current_side']}")
            st.markdown(f"**Visible risk:** {item['user_visible_risk']}")
            st.markdown(f"**Handling rule:** {item['upgrade_handling_rule']}")
            st.markdown("**Source evidence:**")
            st.markdown("\n".join(f"- `{evidence}`" for evidence in item["source_evidence"]))
            if item["notes"]:
                st.markdown("**Notes:**")
                st.markdown("\n".join(f"- {note}" for note in item["notes"]))


def render() -> None:
    context = build_navigator_page_context()
    identity_card = context["project_identity_card"]
    public_truth = context["public_model_truth_card"]
    topic_options = context["topic_options"]
    topic_label_to_key = {item["label"]: item["topic_key"] for item in topic_options}
    default_label = next(
        (
            item["label"]
            for item in topic_options
            if item["topic_key"] == context["default_topic_key"]
        ),
        topic_options[0]["label"],
    )
    selected_topic_label = st.selectbox(
        "Navigator topic",
        options=[item["label"] for item in topic_options],
        index=[item["label"] for item in topic_options].index(default_label),
        help="Choose a governed topic to see deterministic routing, source-of-truth, and drift context.",
    )
    selected_topic = build_navigator_topic_drilldown(topic_label_to_key[selected_topic_label])

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

    st.subheader("Governed Topic Console")
    console_cols = st.columns([1.1, 0.9], gap="large")
    routing = selected_topic["routing_recommendation"]
    with console_cols[0]:
        _render_card(
            selected_topic["topic_label"],
            (
                f"{selected_topic['topic_summary']}<br><br>"
                f"<strong>Supporting phase:</strong> {selected_topic['supporting_phase']['phase_title']}<br>"
                f"<strong>Phase goal:</strong> {selected_topic['supporting_phase']['phase_goal']}"
            ),
            tone="primary",
        )
    with console_cols[1]:
        st.markdown("**Recommended destination**")
        st.markdown(
            f"`{routing['recommended_page_title']}` (`/{routing['recommended_page_route']}`)"
        )
        st.markdown(
            f"**Why this page:** {routing['reason']}"
        )
        st.markdown(
            f"**Supporting phase:** {str(routing['supporting_phase']).capitalize()}"
        )
        if routing["related_source_ids"]:
            st.caption(
                "Related source IDs: " + ", ".join(routing["related_source_ids"])
            )

    st.markdown("**Relevant truth summaries**")
    for truth in selected_topic["truth_summaries"]:
        with st.container(border=True):
            st.markdown(f"**{truth['title']}**")
            st.markdown(truth["description"])
            st.caption(f"Authority rule: {truth['authority_rule']}")

    st.subheader("Runtime Governance")
    governance_cols = st.columns(2, gap="large")
    for column, card in zip(governance_cols, context["runtime_governance_cards"]):
        with column:
            _render_card(card["title"], card["summary"], tone=card["tone"])
            st.caption(card["authority_rule"])

    st.subheader("Source-Of-Truth Drilldown")
    st.caption(
        "This section shows which files govern the selected topic, how authoritative they are, and which runtime surfaces consume them."
    )
    _render_source_table(selected_topic["source_records"])

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

    st.subheader("Drift Register Explorer")
    filter_cols = st.columns(2)
    severity_options = ["All"] + context["drift_filters"]["severities"]
    status_options = ["All"] + context["drift_filters"]["statuses"]
    selected_severity = filter_cols[0].selectbox(
        "Severity filter",
        options=severity_options,
        index=0,
    )
    selected_status = filter_cols[1].selectbox(
        "Status filter",
        options=status_options,
        index=0,
    )
    drift_records = get_drift_records(
        severity=None if selected_severity == "All" else selected_severity,
        status=None if selected_status == "All" else selected_status,
    )
    _render_drift_explorer(drift_records)

    st.markdown("**Topic-linked drift highlights**")
    _render_drift_explorer(selected_topic["drift_records"])

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
