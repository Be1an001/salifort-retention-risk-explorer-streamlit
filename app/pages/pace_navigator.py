from __future__ import annotations

import pandas as pd
import streamlit as st

from app.viewmodels import (
    build_citation_comparison,
    build_governed_answer_view,
    build_navigator_page_context,
    build_navigator_topic_drilldown,
    build_reviewer_filter_options,
    build_support_quality_review,
    filter_and_sort_retrieval_rows,
    get_governed_answer_query_options,
    get_drift_records,
    get_reviewer_sort_options,
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


def _render_recommended_pages(recommended_pages: list[dict[str, object]]) -> None:
    if not recommended_pages:
        st.info("No governed page recommendations are available for this query.")
        return

    for item in recommended_pages:
        with st.container(border=True):
            st.markdown(f"**{item['title']}**")
            st.markdown(f"`/{item['route']}`")
            st.caption(str(item["reason"]))


def _render_support_quality_review(review: dict[str, object]) -> None:
    _render_card(
        str(review["status_label"]),
        "Reviewer-facing support status based on coverage, truth, drift, page-route, and reference-only signals.",
        tone=str(review["tone"]),
    )
    st.markdown("**Review indicators**")
    indicator_cols = st.columns(4)
    for column, indicator in zip(indicator_cols, review["indicators"]):
        column.metric(str(indicator["label"]), str(indicator["value"]))
    st.markdown("**Review notes**")
    st.markdown("\n".join(f"- {note}" for note in review["review_notes"]))


def _render_citation_detail_card(item: dict[str, object], label: str) -> None:
    st.markdown(f"**{label}: {item['title']}**")
    st.markdown(f"**Chunk ID:** `{item['chunk_id']}`")
    st.markdown(f"**Document ID:** `{item['document_id']}`")
    if item.get("similarity_score") is not None:
        st.markdown(f"**Similarity score:** `{item['similarity_score']}`")
    st.markdown(f"**Retrieval role:** `{item['retrieval_role']}`")
    st.markdown(f"**Authority level:** `{item['authority_level']}`")
    if item["truth_tags"]:
        st.markdown("**Truth tags:** " + ", ".join(f"`{tag}`" for tag in item["truth_tags"]))
    if item["drift_tags"]:
        st.markdown("**Drift tags:** " + ", ".join(f"`{tag}`" for tag in item["drift_tags"]))
    if item["phase_tags"]:
        st.markdown("**Phase tags:** " + ", ".join(f"`{tag}`" for tag in item["phase_tags"]))
    if item.get("page_routes"):
        st.markdown("**Page routes:** " + ", ".join(f"`{route}`" for route in item["page_routes"]))
    st.markdown("**Source paths:**")
    st.markdown("\n".join(f"- `{path}`" for path in item["source_paths"]))
    if item["registry_refs"]:
        st.markdown("**Registry refs:**")
        st.markdown("\n".join(f"- `{ref}`" for ref in item["registry_refs"]))
    if item.get("caveats"):
        st.markdown("**Caveats:**")
        st.markdown("\n".join(f"- {note}" for note in item["caveats"]))
    st.markdown("**Preview:**")
    st.write(item.get("text_preview") or "No preview text available.")
    with st.expander("Full chunk text", expanded=False):
        st.write(item.get("full_text") or "No full chunk text is available.")


def _render_citations(citation_rows: list[dict[str, object]]) -> None:
    if not citation_rows:
        st.info("No citations are available for this answer.")
        return

    for item in citation_rows:
        with st.expander(f"{item['title']} ({item['chunk_id']})", expanded=False):
            st.markdown(f"**Document ID:** `{item['document_id']}`")
            st.markdown(f"**Retrieval role:** `{item['retrieval_role']}`")
            st.markdown(f"**Authority level:** `{item['authority_level']}`")
            if item.get("similarity_score") is not None:
                st.markdown(f"**Similarity score:** `{item['similarity_score']}`")
            if item["truth_tags"]:
                st.markdown("**Truth tags:** " + ", ".join(f"`{tag}`" for tag in item["truth_tags"]))
            if item["drift_tags"]:
                st.markdown("**Drift tags:** " + ", ".join(f"`{tag}`" for tag in item["drift_tags"]))
            if item["phase_tags"]:
                st.markdown("**Phase tags:** " + ", ".join(f"`{tag}`" for tag in item["phase_tags"]))
            st.markdown("**Source paths:**")
            st.markdown("\n".join(f"- `{path}`" for path in item["source_paths"]))
            if item["registry_refs"]:
                st.markdown("**Registry refs:**")
                st.markdown("\n".join(f"- `{ref}`" for ref in item["registry_refs"]))
            if item.get("caveats"):
                st.markdown("**Caveats:**")
                st.markdown("\n".join(f"- {note}" for note in item["caveats"]))
            with st.expander("Chunk text", expanded=False):
                st.write(item.get("full_text") or item.get("text_preview") or "No chunk text is available.")


def _render_retrieval_inspector(retrieval_rows: list[dict[str, object]]) -> None:
    if not retrieval_rows:
        st.info("No retrieval results are available for this query.")
        return

    table_rows = []
    for item in retrieval_rows:
        table_rows.append(
            {
                "Rank": item["rank"],
                "Chunk ID": item["chunk_id"],
                "Score": item["similarity_score"],
                "Role": item["retrieval_role"],
                "Authority": item["authority_level"],
                "Title": item["title"],
                "Truth Tags": ", ".join(item["truth_tags"]) if item["truth_tags"] else "None",
                "Drift Tags": ", ".join(item["drift_tags"]) if item["drift_tags"] else "None",
                "Phase Tags": ", ".join(item["phase_tags"]) if item["phase_tags"] else "None",
                "Routes": ", ".join(item["page_routes"]) if item["page_routes"] else "None",
            }
        )
    st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)

    for item in retrieval_rows:
        with st.expander(f"Rank {item['rank']} | {item['title']}", expanded=False):
            st.markdown(f"**Chunk ID:** `{item['chunk_id']}`")
            st.markdown(f"**Similarity score:** `{item['similarity_score']}`")
            st.markdown(f"**Authority level:** `{item['authority_level']}`")
            st.markdown(f"**Retrieval role:** `{item['retrieval_role']}`")
            st.markdown(f"**Preview:** {item['text_preview']}")
            if item["page_titles"]:
                st.markdown("**Page routes:** " + ", ".join(f"`{route}` ({title})" for route, title in zip(item["page_routes"], item["page_titles"])))
            st.markdown("**Source paths:**")
            st.markdown("\n".join(f"- `{path}`" for path in item["source_paths"]))
            if item["registry_refs"]:
                st.markdown("**Registry refs:**")
                st.markdown("\n".join(f"- `{ref}`" for ref in item["registry_refs"]))
            if item["caveats"]:
                st.markdown("**Caveats:**")
                st.markdown("\n".join(f"- {note}" for note in item["caveats"]))


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
    answer_query_options = get_governed_answer_query_options()
    answer_query_labels = {
        item["query"]: f"{item['query']} — {item['description']}"
        for item in answer_query_options
    }
    selected_answer_query = st.selectbox(
        "Governed answer query",
        options=[item["query"] for item in answer_query_options],
        format_func=lambda query: answer_query_labels[query],
        index=0,
        help="This is a controlled retrieval/answer viewer, not a free-form chatbot.",
    )
    selected_top_k = st.slider(
        "Reviewer top-k retrieval depth",
        min_value=5,
        max_value=12,
        value=8,
        help="Controls how many governed chunks are retrieved for the answer viewer and inspector.",
    )
    answer_view = build_governed_answer_view(selected_answer_query, top_k=selected_top_k)

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

    st.subheader("Governed Answer Viewer")
    st.caption(
        "This section runs the existing governed retrieval and deterministic answer assembly stack for a fixed query set. "
        "It exposes answer text, drift, citations, coverage, and raw retrieval context side by side."
    )

    if answer_view["status"] == "blocked":
        if answer_view["error_kind"] == "missing_api_key":
            st.warning(answer_view["message"])
            st.caption(
                "Set `RAG_STREAMLIT_OPENAI_API_KEY` or `OPENAI_API_KEY` locally. No key is stored in the repo."
            )
        elif answer_view["error_kind"] == "missing_index":
            st.warning(answer_view["message"])
            st.caption(
                "Build the governed local retrieval index before using the answer viewer."
            )
        else:
            st.warning(answer_view["message"])
    elif answer_view["status"] == "error":
        st.error(answer_view["message"])
        st.caption(
            "The viewer did not fabricate an answer. Review the retrieval runtime setup and try again."
        )
    else:
        answer = answer_view["answer"]
        support_review = build_support_quality_review(answer_view)
        answer_tabs = st.tabs(
            [
                "Answer Summary",
                "Support Review",
                "Truth Support",
                "Drift & Caveats",
                "Recommended Pages",
                "Citations",
                "Retrieval Inspector",
            ]
        )

        with answer_tabs[0]:
            metric_cols = st.columns(3)
            metric_cols[0].metric("Assembly Status", str(answer["assembly_status"]).replace("_", " ").title())
            metric_cols[1].metric("Coverage Status", str(answer["coverage_summary"]["status"]).capitalize())
            metric_cols[2].metric("Retrieved Chunks", str(answer["coverage_summary"]["retrieved_result_count"]))
            st.markdown(f"### {answer['answer_title']}")
            st.write(answer["direct_answer"])
            st.markdown("**Governance flags**")
            if answer["governance_flags"]:
                st.markdown("\n".join(f"- `{flag}`" for flag in answer["governance_flags"]))
            else:
                st.caption("No additional governance flags were needed for this answer.")
            with st.expander("Coverage details", expanded=False):
                st.json(answer["coverage_summary"])

        with answer_tabs[1]:
            _render_support_quality_review(support_review)

        with answer_tabs[2]:
            st.markdown("**Supporting points**")
            if answer["supporting_points"]:
                st.markdown("\n".join(f"- {item}" for item in answer["supporting_points"]))
            else:
                st.info("No supporting points were assembled.")
            truth_citations = [
                item for item in answer_view["citation_rows"] if item["truth_tags"]
            ]
            if truth_citations:
                st.markdown("**Canonical truth citations**")
                _render_citations(truth_citations)

        with answer_tabs[3]:
            if answer["drift_and_caveats"]:
                st.markdown("\n".join(f"- {item}" for item in answer["drift_and_caveats"]))
            else:
                st.info("No drift or caveat notes were attached to this answer.")

        with answer_tabs[4]:
            _render_recommended_pages(answer["recommended_pages"])

        with answer_tabs[5]:
            _render_citations(answer_view["citation_rows"])

            if len(answer_view["citation_rows"]) >= 2:
                st.markdown("**Side-by-side citation comparison**")
                citation_options = {
                    f"{item['title']} ({item['chunk_id']})": item["chunk_id"]
                    for item in answer_view["citation_rows"]
                }
                citation_labels = list(citation_options.keys())
                compare_cols = st.columns(2)
                left_label = compare_cols[0].selectbox(
                    "Left citation",
                    options=citation_labels,
                    index=0,
                )
                right_label = compare_cols[1].selectbox(
                    "Right citation",
                    options=citation_labels,
                    index=1,
                )
                comparison = build_citation_comparison(
                    answer_view["citation_rows"],
                    citation_options[left_label],
                    citation_options[right_label],
                )
                if comparison["status"] == "ready":
                    left_col, right_col = st.columns(2, gap="large")
                    with left_col:
                        _render_citation_detail_card(comparison["left"], "Left")
                    with right_col:
                        _render_citation_detail_card(comparison["right"], "Right")
                else:
                    st.warning("Select two valid citations to compare.")
            else:
                st.info("At least two citations are needed for side-by-side comparison.")

        with answer_tabs[6]:
            filter_options = build_reviewer_filter_options(answer_view["retrieval_rows"])
            sort_options = {
                item["label"]: item["key"]
                for item in get_reviewer_sort_options()
            }
            st.markdown("**Reviewer filters**")
            filter_cols = st.columns(3)
            selected_truth_tag = filter_cols[0].selectbox(
                "Truth tag",
                options=filter_options["truth_tags"],
                index=0,
            )
            selected_drift_tag = filter_cols[1].selectbox(
                "Drift tag",
                options=filter_options["drift_tags"],
                index=0,
            )
            selected_phase_tag = filter_cols[2].selectbox(
                "Phase tag",
                options=filter_options["phase_tags"],
                index=0,
            )
            filter_cols = st.columns(3)
            selected_role = filter_cols[0].selectbox(
                "Retrieval role",
                options=filter_options["retrieval_roles"],
                index=0,
            )
            selected_route = filter_cols[1].selectbox(
                "Page route",
                options=filter_options["page_routes"],
                index=0,
            )
            selected_authority = filter_cols[2].selectbox(
                "Authority level",
                options=filter_options["authority_levels"],
                index=0,
            )
            selected_sort_label = st.selectbox(
                "Sort retrieval results",
                options=list(sort_options.keys()),
                index=0,
            )
            filtered_rows = filter_and_sort_retrieval_rows(
                answer_view["retrieval_rows"],
                truth_tag=selected_truth_tag,
                drift_tag=selected_drift_tag,
                phase_tag=selected_phase_tag,
                retrieval_role=selected_role,
                page_route=selected_route,
                authority_level=selected_authority,
                sort_key=sort_options[selected_sort_label],
            )
            st.caption(
                f"Showing {len(filtered_rows)} of {len(answer_view['retrieval_rows'])} retrieved chunks."
            )
            _render_retrieval_inspector(filtered_rows)

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
