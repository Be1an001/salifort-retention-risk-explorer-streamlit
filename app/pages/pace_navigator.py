from __future__ import annotations

import pandas as pd
import streamlit as st

from app.viewmodels import (
    build_agent_shell_context,
    build_agent_shell_preview,
    build_audit_summary_export,
    build_audit_workflow,
    build_audit_checklist,
    build_citation_comparison,
    build_cross_query_audit_export,
    build_eligible_source_index,
    build_final_system_readiness_context,
    build_governed_answer_view,
    build_navigator_page_context,
    build_navigator_topic_drilldown,
    build_orchestration_summary,
    build_orchestration_workflow_detail,
    build_reviewer_filter_options,
    build_source_detail_options,
    build_source_detail_view,
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


def _page_link(title: object, route: object) -> str:
    route_text = str(route).strip("/")
    return f"[{title}](/{route_text})"


def _render_recommended_pages(recommended_pages: list[dict[str, object]]) -> None:
    if not recommended_pages:
        st.info("No page recommendations are available for this query.")
        return

    for item in recommended_pages:
        with st.container(border=True):
            st.markdown(f"**{_page_link(item['title'], item['route'])}**")
            st.caption(f"Route: `/{item['route']}`")
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


def _render_audit_checklist(checklist: dict[str, object]) -> None:
    summary = checklist["summary"]
    metric_cols = st.columns(4)
    metric_cols[0].metric("Checklist Status", str(checklist["status_label"]))
    metric_cols[1].metric("Completeness", f"{float(checklist['completeness_score']) * 100:.0f}%")
    metric_cols[2].metric("Ready Items", str(summary["ready_items"]))
    metric_cols[3].metric("Attention / Blocked", f"{summary['attention_items']} / {summary['blocked_items']}")

    table_rows = [
        {
            "Checklist Item": item["label"],
            "Status": str(item["status"]).replace("_", " ").title(),
            "Detail": item["detail"],
        }
        for item in checklist["items"]
    ]
    st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)
    st.markdown("**Checklist notes**")
    st.markdown("\n".join(f"- {note}" for note in summary["notes"]))


def _render_eligible_source_index(source_index: dict[str, object]) -> None:
    summary = source_index["summary"]
    st.caption(str(summary["governance_note"]))
    metric_cols = st.columns(4)
    metric_cols[0].metric("Governed Sources", str(summary["total_sources"]))
    metric_cols[1].metric("Preview Eligible", str(summary["eligible_sources"]))
    metric_cols[2].metric("Blocked / Unavailable", str(summary["blocked_sources"]))
    metric_cols[3].metric("Preview Cap", f"{summary['preview_max_bytes']} bytes")

    rows = source_index["rows"]
    status_filter = st.selectbox(
        "Source index preview filter",
        options=["All", "Preview eligible", "Blocked or unavailable"],
        index=0,
        help="This filters governed source-registry and retrieval-pack paths only.",
    )
    if status_filter == "Preview eligible":
        filtered_rows = [row for row in rows if row["preview_eligible"]]
    elif status_filter == "Blocked or unavailable":
        filtered_rows = [row for row in rows if not row["preview_eligible"]]
    else:
        filtered_rows = list(rows)

    table_rows = [
        {
            "Source Path": row["source_path"],
            "Label": row["source_label"],
            "Universe": row["source_universe"],
            "Eligible": "Yes" if row["preview_eligible"] else "No",
            "Extension": row["extension"],
            "Size Bucket": row["size_bucket"],
            "Class": row["preview_class"],
            "Blocked Reason": row["blocked_reason"] or "None",
        }
        for row in filtered_rows
    ]
    st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)
    with st.expander("Source preview governance rules", expanded=False):
        st.markdown(
            "- Only governed source-registry and retrieval-pack paths are indexed.\n"
            "- Only small, repo-local, text-like files can be previewed.\n"
            "- Secret-like paths, environment files, binaries, generated arrays, images, and large files remain blocked.\n"
            "- This is not arbitrary repo enumeration or unrestricted file browsing."
        )


def _render_orchestration_summary(orchestration: dict[str, object]) -> None:
    summary = orchestration["summary"]
    st.caption(str(summary["governance_note"]))
    metric_cols = st.columns(4)
    metric_cols[0].metric("Governed Workflows", str(summary["workflow_count"]))
    metric_cols[1].metric("Task Memberships", str(summary["workflow_task_memberships"]))
    metric_cols[2].metric("Human Review", str(summary["human_review_workflows"]))
    metric_cols[3].metric("Mutating Workflows", str(summary["mutating_workflows"]))

    workflow_rows = [
        {
            "Workflow": item["workflow_title"],
            "ID": item["workflow_id"],
            "Kind": item["workflow_kind"],
            "Runtime": item["runtime_mode"],
            "Scheduler": item["scheduler_eligibility"],
            "Human Review": "Yes" if item["human_review_required"] else "No",
            "Mutates Files": "Yes" if item["mutates_repo_files"] else "No",
            "Tasks": item["task_count"],
            "Blocker Status": item["blocker_status"],
        }
        for item in orchestration["workflow_cards"]
    ]
    st.dataframe(pd.DataFrame(workflow_rows), use_container_width=True, hide_index=True)

    workflow_labels = {
        f"{item['workflow_title']} ({item['workflow_id']})": item["workflow_id"]
        for item in orchestration["workflow_cards"]
    }
    selected_label = st.selectbox(
        "Workflow contract detail",
        options=list(workflow_labels.keys()),
        index=0,
        help="Inspect task boundaries and blockers. This does not execute the workflow.",
    )
    detail = build_orchestration_workflow_detail(workflow_labels[selected_label])
    detail_summary = detail["summary"]
    blockers = detail["blockers"]
    detail_cols = st.columns(4)
    detail_cols[0].metric("Runtime", str(detail_summary["runtime_mode"]))
    detail_cols[1].metric("Scheduler Class", str(detail_summary["scheduler_eligibility"]))
    detail_cols[2].metric("Human Review", "Yes" if detail_summary["human_review_required"] else "No")
    detail_cols[3].metric("Writes Files", "Yes" if detail_summary["mutates_repo_files"] else "No")

    st.markdown("**Execution order and task boundaries**")
    task_rows = [
        {
            "Order": row["order"],
            "Task ID": row["task_id"],
            "Kind": row["task_kind"],
            "Runtime": row["runtime_mode"],
            "Scheduler": row["scheduler_eligibility"],
            "Writes Files": "Yes" if row["mutates_repo_files"] else "No",
            "Human Review": "Yes" if row["human_review_required"] else "No",
            "Dependencies": ", ".join(row["dependencies"]) if row["dependencies"] else "None",
            "Command Hint": row["command_hint"] or "Contract only",
        }
        for row in detail["task_rows"]
    ]
    st.dataframe(pd.DataFrame(task_rows), use_container_width=True, hide_index=True)

    blocker_cols = st.columns(2)
    blocker_cols[0].markdown("**Missing artifacts**")
    blocker_cols[0].markdown(
        "\n".join(f"- `{item}`" for item in blockers["artifact_status"]["missing"])
        or "- None detected"
    )
    blocker_cols[1].markdown("**Missing environment requirements**")
    blocker_cols[1].markdown(
        "\n".join(f"- `{item}`" for item in blockers["env_status"]["missing"])
        or "- None detected"
    )

    with st.expander("Blocked states and workflow notes", expanded=False):
        st.markdown("**Blocked states:**")
        st.markdown("\n".join(f"- {item}" for item in blockers["blocked_states"]))
        st.markdown("**Workflow notes:**")
        st.write(detail["workflow"]["notes"])


def _render_agent_shell(agent_context: dict[str, object]) -> None:
    summary = agent_context["summary"]
    st.caption(str(summary["governance_note"]))
    metric_cols = st.columns(4)
    metric_cols[0].metric("Controlled Requests", str(summary["controlled_request_count"]))
    metric_cols[1].metric("Supported Intents", str(summary["supported_intent_count"]))
    metric_cols[2].metric("Blocked Behaviors", str(summary["disallowed_behavior_count"]))
    metric_cols[3].metric("Execution", "Preview only")

    request_labels = {
        f"{request['label']} ({request['request_id']})": request["request_id"]
        for request in agent_context["requests"]
    }
    selected_label = st.selectbox(
        "Controlled agent request",
        options=list(request_labels.keys()),
        index=0,
        help="Choose from governed request intents only. This is not a free-text agent prompt.",
    )
    preview = build_agent_shell_preview(request_labels[selected_label])
    if preview["status"] != "preview_ready":
        st.warning(preview["classification_note"])
        return

    route = preview["route_summary"]
    route_cols = st.columns(4)
    route_cols[0].markdown("**Intent**")
    route_cols[0].write(str(route["intent_title"]))
    route_cols[1].markdown("**PACE phase**")
    route_cols[1].write(str(route["pace_phase"]).capitalize())
    route_cols[2].markdown("**Workflow**")
    route_cols[2].code(str(route["workflow_id"]))
    route_cols[3].markdown("**Execution**")
    route_cols[3].write("Not allowed from the app")

    st.info(preview["execution_policy_note"])
    st.markdown("**Route explanation**")
    st.write(
        f"`{route['label']}` maps to `{route['intent_id']}` and workflow "
        f"`{route['workflow_title']}`. Recommended page routes: "
        + ", ".join(f"`/{page}`" for page in route["page_routes"])
    )
    if route["governed_query"]:
        st.caption(f"Governed fixed query: `{route['governed_query']}`")

    st.markdown("**Plan preview and task contract mapping**")
    plan_rows = [
        {
            "Order": row["order"],
            "Task ID": row["task_id"],
            "Included": "Yes" if row["included_for_request"] else "Context",
            "Kind": row["task_kind"],
            "Runtime": row["runtime_mode"],
            "Writes Files": "Yes" if row["mutates_repo_files"] else "No",
            "Human Review": "Yes" if row["human_review_required"] else "No",
            "Dependencies": ", ".join(row["dependencies"]) if row["dependencies"] else "None",
            "Note": row["preview_note"],
        }
        for row in preview["plan_rows"]
    ]
    st.dataframe(pd.DataFrame(plan_rows), use_container_width=True, hide_index=True)

    info_cols = st.columns(2)
    with info_cols[0]:
        st.markdown("**Required inputs**")
        st.markdown(
            "\n".join(
                f"- {item}" for item in preview["required_inputs"]["request_inputs"]
            )
        )
        st.markdown("**Missing artifacts**")
        st.markdown(
            "\n".join(f"- `{item}`" for item in route["missing_artifacts"])
            or "- None detected"
        )
    with info_cols[1]:
        st.markdown("**Expected outputs**")
        st.markdown(
            "\n".join(
                f"- {item}"
                for item in preview["expected_outputs"]["request_expected_outputs"]
            )
        )
        st.markdown("**Missing environment requirements**")
        st.markdown(
            "\n".join(f"- `{item}`" for item in route["missing_env_vars"])
            or "- None detected"
        )

    st.markdown("**Human-review checkpoints**")
    st.markdown("\n".join(f"- {item}" for item in preview["review_checkpoints"]))

    with st.expander("Agent shell guardrails and disallowed behavior policy", expanded=False):
        st.markdown("**Guardrails:**")
        st.markdown("\n".join(f"- {item}" for item in preview["guardrails"]["guardrails"]))
        st.markdown("**Explicitly disallowed behavior classes:**")
        st.markdown(
            "\n".join(
                f"- `{item['behavior_id']}`: {item['handling_rule']}"
                for item in agent_context["disallowed_behaviors"]
            )
        )


def _render_final_system_readiness(readiness: dict[str, object]) -> None:
    summary = readiness["summary"]
    counts = readiness["combined_status_counts"]
    st.caption(str(summary["governance_note"]))
    metric_cols = st.columns(5)
    metric_cols[0].metric("Demo", "Ready" if readiness["demo_ready"] else "Blocked")
    metric_cols[1].metric("Ready", str(counts["ready"]))
    metric_cols[2].metric("Needs Review", str(counts["review_needed"]))
    metric_cols[3].metric("Preview Only", str(counts["preview_only"]))
    metric_cols[4].metric("Blocked", str(counts["blocked"]))
    st.caption(
        "Status counts combine the readiness matrix and approval gates shown below. "
        f"Total counted items: {readiness['combined_status_total']}."
    )
    st.caption(f"Demo status detail: {summary['demo_status']}")

    component_rows = [
        {
            "Component": item["component_title"],
            "Status": item["status_label"],
            "Kind": item["readiness_kind"],
            "Human Review": "Yes" if item["human_review_required"] else "No",
            "Streamlit Execution": "No",
            "Missing Artifacts": ", ".join(item["missing_artifacts"]) or "None",
            "Missing Env": ", ".join(item["missing_env_vars"]) or "None",
        }
        for item in readiness["component_cards"]
    ]
    st.markdown("**Integrated readiness matrix**")
    st.dataframe(pd.DataFrame(component_rows), use_container_width=True, hide_index=True)

    checklist = readiness["demo_checklist"]
    st.markdown("**Demo checklist**")
    checklist_rows = [
        {
            "Check": item["label"],
            "Status": str(item["status"]).replace("_", " ").title(),
        }
        for item in checklist["items"]
    ]
    st.dataframe(pd.DataFrame(checklist_rows), use_container_width=True, hide_index=True)
    st.markdown("\n".join(f"- {note}" for note in checklist["notes"]))

    execution = readiness["execution_eligibility"]
    st.markdown("**Execution eligibility boundary**")
    st.info(str(execution["summary"]["note"]))
    workflow_rows = [
        {
            "Workflow": row["workflow_id"],
            "Runtime": row["runtime_mode"],
            "Scheduler": row["scheduler_eligibility"],
            "Writes Files": "Yes" if row["mutates_repo_files"] else "No",
            "Human Review": "Yes" if row["human_review_required"] else "No",
            "Streamlit Execution": "No",
            "Blocker Status": row["blocker_status"],
        }
        for row in execution["workflow_rows"]
    ]
    st.dataframe(pd.DataFrame(workflow_rows), use_container_width=True, hide_index=True)

    with st.expander("Approval gates and final guardrails", expanded=False):
        st.markdown("**Approval gates:**")
        for gate in readiness["approval_gates"]:
            st.markdown(
                f"- `{gate['gate_id']}` | {gate['gate_title']} | status={gate['status']} | "
                f"Streamlit execution allowed={gate['execution_allowed_in_streamlit']}"
            )
        st.markdown("**Shared guardrails:**")
        st.markdown("\n".join(f"- {item}" for item in readiness["guardrails"]))


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


def _render_source_detail(detail: dict[str, object]) -> None:
    if detail["status"] != "ready":
        st.warning(str(detail["browser_note"]))
        return

    selected = detail["selected"]
    st.info(str(detail["browser_note"]))
    _render_citation_detail_card(selected, "Selected governed chunk")

    st.markdown("**Evidence trace**")
    trace = detail["evidence_trace"]
    trace_cols = st.columns(4)
    trace_cols[0].metric("Selected Chunk", str(trace["selected_chunk_id"]))
    trace_cols[1].metric("Related Chunks", str(trace["related_chunk_count"]))
    trace_cols[2].metric("Preview Status", str(trace["preview_status"]).capitalize())
    trace_cols[3].metric("Browser Level", str(trace["browser_level"]).replace("_", " ").title())
    with st.expander("Trace chain details", expanded=False):
        st.markdown(f"**Query:** {trace['query']}")
        st.markdown(f"**Answer title:** {trace['answer_title']}")
        st.markdown(f"**Document ID:** `{trace['document_id']}`")
        st.markdown("**Source paths:**")
        st.markdown("\n".join(f"- `{path}`" for path in trace["source_paths"]))
        st.markdown("**Registry refs:**")
        st.markdown("\n".join(f"- `{ref}`" for ref in trace["registry_refs"]) or "- None")
        st.markdown(f"**Preview rationale:** {trace['preview_rationale']}")

    st.markdown("**Eligible source-file preview**")
    preview_options = detail["preview_options"]
    if preview_options:
        preview_labels = {
            f"{option['source_path']} ({option['status']})": option
            for option in preview_options
        }
        selected_preview_label = st.selectbox(
            "Source path preview candidate",
            options=list(preview_labels.keys()),
            index=0,
            help="Only governed, repo-local, small text-like files can be previewed.",
        )
        preview = preview_labels[selected_preview_label]
        if preview["eligible"]:
            st.success(preview["reason"])
            st.caption(
                f"Size: {preview['file_size_bytes']} bytes | Extension: `{preview['extension']}` | "
                f"Truncated: {'Yes' if preview['is_truncated'] else 'No'} | {preview['limit_note']}"
            )
            st.text_area(
                "Governed read-only source preview",
                value=str(preview["preview_text"]),
                height=320,
            )
        else:
            st.warning(preview["reason"])
            st.caption(
                "Use the governed chunk text and related chunks above as the fallback context."
            )
    else:
        st.info("No source paths are available for preview eligibility evaluation.")

    related_chunks = detail["related_chunks"]
    st.markdown("**Related governed chunks from the same document or source path**")
    if not related_chunks:
        st.caption("No related chunks were found in the governed retrieval pack.")
        return

    table_rows = []
    for item in related_chunks:
        table_rows.append(
            {
                "Relationship": item["relationship"],
                "Chunk ID": item["chunk_id"],
                "Kind": item["chunk_kind"],
                "Role": item["retrieval_role"],
                "Authority": item["authority_level"],
                "Title": item["title"],
            }
        )
    st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)
    for item in related_chunks:
        with st.expander(f"{item['relationship']}: {item['title']}", expanded=False):
            st.markdown(f"**Chunk ID:** `{item['chunk_id']}`")
            st.markdown(f"**Document ID:** `{item['document_id']}`")
            st.markdown(f"**Chunk kind:** `{item['chunk_kind']}`")
            if item["truth_tags"]:
                st.markdown("**Truth tags:** " + ", ".join(f"`{tag}`" for tag in item["truth_tags"]))
            if item["drift_tags"]:
                st.markdown("**Drift tags:** " + ", ".join(f"`{tag}`" for tag in item["drift_tags"]))
            if item["phase_tags"]:
                st.markdown("**Phase tags:** " + ", ".join(f"`{tag}`" for tag in item["phase_tags"]))
            st.markdown("**Source paths:**")
            st.markdown("\n".join(f"- `{path}`" for path in item["source_paths"]))
            st.markdown("**Preview:**")
            st.write(item["text_preview"])


def _render_audit_export(export_payload: dict[str, object]) -> None:
    if export_payload["status"] != "ready":
        st.warning(str(export_payload["message"]))
        return

    filenames = export_payload["filenames"]
    st.markdown("**Copyable governed markdown summary**")
    st.text_area(
        "Review summary markdown",
        value=str(export_payload["markdown"]),
        height=360,
        help="Copy this governed review block into an audit note or PR review.",
    )

    download_cols = st.columns(3)
    download_cols[0].download_button(
        "Download Markdown",
        data=str(export_payload["markdown"]),
        file_name=str(filenames["markdown"]),
        mime="text/markdown",
    )
    download_cols[1].download_button(
        "Download JSON Packet",
        data=str(export_payload["json"]),
        file_name=str(filenames["json"]),
        mime="application/json",
    )
    download_cols[2].download_button(
        "Download Text Summary",
        data=str(export_payload["text"]),
        file_name=str(filenames["text"]),
        mime="text/plain",
    )
    with st.expander("JSON packet preview", expanded=False):
        st.json(export_payload["packet"])


def _render_cross_query_export(export_payload: dict[str, object]) -> None:
    if export_payload["status"] != "ready":
        st.warning(str(export_payload["message"]))
        return

    filenames = export_payload["filenames"]
    st.markdown("**Copyable cross-query audit markdown**")
    st.text_area(
        "Cross-query audit markdown",
        value=str(export_payload["markdown"]),
        height=380,
        help="Copy this governed multi-query review block into an audit note or PR review.",
    )
    download_cols = st.columns(3)
    download_cols[0].download_button(
        "Download Cross-Query Markdown",
        data=str(export_payload["markdown"]),
        file_name=str(filenames["markdown"]),
        mime="text/markdown",
    )
    download_cols[1].download_button(
        "Download Cross-Query JSON",
        data=str(export_payload["json"]),
        file_name=str(filenames["json"]),
        mime="application/json",
    )
    download_cols[2].download_button(
        "Download Cross-Query Text",
        data=str(export_payload["text"]),
        file_name=str(filenames["text"]),
        mime="text/plain",
    )
    with st.expander("Cross-query JSON packet preview", expanded=False):
        st.json(export_payload["packet"])


def _render_audit_workflow(workflow: dict[str, object]) -> None:
    summary = workflow["summary"]
    metric_cols = st.columns(4)
    metric_cols[0].metric("Workflow Status", str(summary["workflow_status"]))
    metric_cols[1].metric("Selected Queries", str(summary["total_queries"]))
    metric_cols[2].metric("Ready", str(summary["ready_queries"]))
    metric_cols[3].metric("Blocked", str(summary["blocked_queries"]))

    st.markdown("**Workflow review notes**")
    st.markdown("\n".join(f"- {note}" for note in summary["review_notes"]))

    if not workflow["comparison_rows"]:
        st.info("No queries are selected for comparison.")
        return

    table_rows = []
    for row in workflow["comparison_rows"]:
        table_rows.append(
            {
                "Query": row["query"],
                "Answer Title": row["answer_title"],
                "Support": row["support_quality_status"],
                "Coverage": row["coverage_status"],
                "Truth": "Yes" if row["canonical_truth_present"] else "No",
                "Drift": "Yes" if row["drift_context_present"] else "No",
                "Page Route": "Yes" if row["page_route_support_present"] else "No",
                "Reference-only": row["reference_only_count"],
                "Citations": row["citation_count"],
                "Source Detail": "Yes" if row["selected_source_detail_present"] else "No",
                "Checklist": row.get("checklist_status", "Not checked"),
                "Checklist Score": f"{float(row.get('checklist_score', 0)) * 100:.0f}%",
            }
        )
    st.markdown("**Cross-query comparison matrix**")
    st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)

    grouped_cols = st.columns(3)
    grouped_cols[0].markdown("**Strong governed support**")
    grouped_cols[0].markdown(
        "\n".join(f"- {query}" for query in summary["strong_queries"])
        or "- None"
    )
    grouped_cols[1].markdown("**Needs attention / partial**")
    grouped_cols[1].markdown(
        "\n".join(f"- {query}" for query in summary["partial_or_attention_queries"])
        or "- None"
    )
    grouped_cols[2].markdown("**Drift or caveat context**")
    grouped_cols[2].markdown(
        "\n".join(f"- {query}" for query in summary["drift_heavy_queries"])
        or "- None"
    )
    checklist_cols = st.columns(3)
    checklist_cols[0].markdown("**Checklist ready**")
    checklist_cols[0].markdown(
        "\n".join(f"- {query}" for query in summary.get("checklist_ready_queries", []))
        or "- None"
    )
    checklist_cols[1].markdown("**Checklist attention**")
    checklist_cols[1].markdown(
        "\n".join(f"- {query}" for query in summary.get("checklist_attention_queries", []))
        or "- None"
    )
    checklist_cols[2].markdown("**Checklist blocked**")
    checklist_cols[2].markdown(
        "\n".join(f"- {query}" for query in summary.get("checklist_blocked_queries", []))
        or "- None"
    )

    st.markdown("**Per-query review cards**")
    for item in workflow["items"]:
        row = item["comparison_row"]
        with st.expander(f"{row['support_quality_status']} | {row['query']}", expanded=False):
            st.markdown(f"**Answer title:** {row['answer_title']}")
            if row.get("direct_answer"):
                st.markdown("**Direct governed answer:**")
                st.write(row["direct_answer"])
            if row.get("message"):
                st.warning(row["message"])
            st.markdown("**Review notes:**")
            st.markdown("\n".join(f"- {note}" for note in row["review_notes"]))
            if row["drift_and_caveats"]:
                st.markdown("**Drift and caveats:**")
                st.markdown("\n".join(f"- {note}" for note in row["drift_and_caveats"]))
            if row["recommended_pages"]:
                st.markdown("**Recommended pages:**")
                st.markdown(
                    "\n".join(
                        f"- {_page_link(page['title'], page['route'])}: {page['reason']}"
                        for page in row["recommended_pages"]
                    )
                )
            st.caption(
                f"Citations: {row['citation_count']} | Reference-only chunks: {row['reference_only_count']} | "
                f"Checklist: {row.get('checklist_status', 'Not checked')} ({float(row.get('checklist_score', 0)) * 100:.0f}%)"
            )
            checklist = item.get("audit_checklist")
            if checklist:
                with st.expander("Checklist item detail", expanded=False):
                    _render_audit_checklist(checklist)


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
    answer_query_options = get_governed_answer_query_options()
    answer_query_labels = {
        item["query"]: f"{item['query']} — {item['description']}"
        for item in answer_query_options
    }
    eligible_source_index = build_eligible_source_index()
    orchestration_summary = build_orchestration_summary()
    agent_shell_context = build_agent_shell_context()
    final_readiness = build_final_system_readiness_context()

    st.title(context["page_title"])
    st.caption(context["page_caption"])

    st.markdown(
        "Use this page as the project map. Start with the public model truth and page recommendations, "
        "then open the advanced review sections if you want to inspect citations, retrieval evidence, workflow contracts, "
        "or demo readiness. Nothing on this page retrains a model or runs background jobs."
    )

    st.subheader("What This Page Is For")
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

    st.subheader("Current Public Model Truth")
    metric_cols = st.columns(3)
    metric_cols[0].metric("Public Model", public_truth["model_name"])
    metric_cols[1].metric("Selected Threshold", public_truth["selected_threshold"])
    metric_cols[2].metric(
        "Preserve In Upgrade",
        "Yes" if public_truth["preserve_in_upgrade"] else "No",
    )
    st.info(public_truth["summary"])
    st.caption(f"Authority rule: {public_truth['authority_rule']}")

    st.subheader("Guided Topic Explorer")
    selected_topic_label = st.selectbox(
        "Project topic",
        options=[item["label"] for item in topic_options],
        index=[item["label"] for item in topic_options].index(default_label),
        help="Choose a topic to see where it fits in the project and which page to visit next.",
    )
    selected_topic = build_navigator_topic_drilldown(topic_label_to_key[selected_topic_label])
    console_cols = st.columns([1.1, 0.9], gap="large")
    routing = selected_topic["routing_recommendation"]
    with console_cols[0]:
        _render_card(
            selected_topic["topic_label"],
            (
                f"{selected_topic['topic_summary']}<br><br>"
                f"<strong>Topic lens:</strong> {selected_topic['supporting_phase']['phase_title']}<br>"
                f"<strong>Phase goal:</strong> {selected_topic['supporting_phase']['phase_goal']}"
            ),
            tone="primary",
        )
    with console_cols[1]:
        st.markdown("**Recommended destination**")
        st.markdown(_page_link(routing["recommended_page_title"], routing["recommended_page_route"]))
        st.caption(f"Route: `/{routing['recommended_page_route']}`")
        st.markdown(
            f"**Why this page:** {routing['reason']}"
        )
        st.markdown(
            f"**Recommended page phase:** {str(routing['supporting_phase']).capitalize()}"
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

    st.subheader("Source-of-Truth Drilldown")
    st.caption(
        "This section shows which files govern the selected topic, how authoritative they are, and which runtime surfaces consume them."
    )
    _render_source_table(selected_topic["source_records"])

    st.subheader("Known Differences to Preserve and Explain")
    st.caption(
        "Some project layers intentionally differ. This section keeps those differences visible instead of smoothing them over."
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

    st.subheader("Where to Go Next")
    st.caption(
        "Use these page recommendations as a reviewer-friendly route through the app. "
        "They do not run automation or change any data."
    )
    for item in context["topic_recommendations"]:
        with st.container(border=True):
            st.markdown(f"**Topic:** {item['topic']}")
            st.markdown(f"**Recommended page:** {_page_link(item['recommended_page_title'], item['recommended_page_route'])}")
            st.caption(f"Route: `/{item['recommended_page_route']}`")
            st.markdown(
                f"**Recommended page phase:** {str(item['supporting_phase']).capitalize()}"
            )
            st.caption(item["reason"])

    st.subheader("PACE Workflow Map")
    st.caption(
        "PACE means Plan, Analyze, Construct, and Execute. In this app it is a simple project map, "
        "not a separate framework a visitor needs to know ahead of time."
    )
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
            st.caption(phase_card["future_navigator_use"])

    st.divider()
    st.markdown("## Advanced Review Tools")
    st.caption(
        "The sections below are optional reviewer tools. They use fixed questions, prepared retrieval evidence, "
        "source traces, workflow-readiness summaries, and plan previews. They are not a chatbot and they do not execute jobs."
    )

    st.subheader("Advanced Answer Viewer")
    st.caption(
        "This reviewer tool runs fixed questions through the prepared retrieval and answer-assembly stack. "
        "It shows answer text, caveats, citations, coverage, and retrieved evidence side by side. "
        "When retrieval is used, the OpenAI API supplies embeddings from a local key; no key is stored in the repo."
    )
    control_cols = st.columns([1.3, 0.7], gap="large")
    with control_cols[0]:
        selected_answer_query = st.selectbox(
            "Advanced review question",
            options=[item["query"] for item in answer_query_options],
            format_func=lambda query: answer_query_labels[query],
            index=0,
            help="Choose from fixed review questions. This is not a free-form chatbot.",
        )
    with control_cols[1]:
        selected_top_k = st.slider(
            "Evidence chunks to retrieve",
            min_value=5,
            max_value=12,
            value=8,
            help=(
                "This is the retrieval depth, often called top-k. Higher values show more prepared evidence "
                "in the answer viewer and inspector, but they may include more reference-only context."
            ),
        )
    st.caption(
        f"This answer area will use the selected fixed question and retrieve up to {selected_top_k} prepared evidence chunks."
    )
    answer_view = build_governed_answer_view(selected_answer_query, top_k=selected_top_k)

    if answer_view["status"] == "blocked":
        if answer_view["error_kind"] == "missing_api_key":
            st.warning(answer_view["message"])
            st.caption(
                "Set `RAG_STREAMLIT_OPENAI_API_KEY` or `OPENAI_API_KEY` locally. No key is stored in this repo."
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
        source_detail_options = build_source_detail_options(answer_view)
        st.success(
            f"Answer assembled for: {selected_answer_query}. "
            f"Retrieved chunks available for review: {len(answer_view['retrieval_rows'])}."
        )
        answer_tabs = st.tabs(
            [
                "Answer Summary",
                "Support Review",
                "Truth Support",
                "Drift & Caveats",
                "Recommended Pages",
                "Citations",
                "Source Detail",
                "Audit Checklist",
                "Export Review",
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
            if source_detail_options:
                source_option_labels = {
                    item["label"]: item["chunk_id"]
                    for item in source_detail_options
                }
                selected_source_label = st.selectbox(
                    "Source detail target",
                    options=list(source_option_labels.keys()),
                    index=0,
                    help="Choose a citation or retrieved chunk to inspect at governed chunk level.",
                )
                source_detail = build_source_detail_view(
                    answer_view,
                    source_option_labels[selected_source_label],
                )
                _render_source_detail(source_detail)
            else:
                source_detail = {
                    "status": "blocked",
                    "browser_note": "No governed source detail options are available.",
                }
                st.info("No governed source detail options are available.")

        with answer_tabs[7]:
            preliminary_export = build_audit_summary_export(
                answer_view,
                support_review,
                source_detail if source_detail_options else None,
            )
            checklist = build_audit_checklist(
                answer_view,
                support_review,
                source_detail if source_detail_options else None,
                preliminary_export,
            )
            _render_audit_checklist(checklist)

        with answer_tabs[8]:
            if source_detail_options:
                export_source_detail = source_detail
            else:
                export_source_detail = None
            export_payload = build_audit_summary_export(
                answer_view,
                support_review,
                export_source_detail,
                checklist,
                eligible_source_index,
            )
            _render_audit_export(export_payload)

        with answer_tabs[9]:
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

    st.subheader("Advanced Source Preview Index")
    st.caption(
        "This index shows which prepared project sources can be previewed safely. "
        "It is a curated review surface, not a general repository browser."
    )
    _render_eligible_source_index(eligible_source_index)

    st.subheader("Advanced Multi-Query Audit")
    st.caption(
        "Compare multiple fixed review questions in one workspace. This is controlled review, not free-form chat."
    )
    default_workflow_queries = [
        "what is the public model truth",
        "how is fallback different from final model truth",
        "why is threshold 0.29 used",
    ]
    selected_workflow_queries = st.multiselect(
        "Fixed governed queries for audit workflow",
        options=[item["query"] for item in answer_query_options],
        default=[
            query
            for query in default_workflow_queries
            if query in {item["query"] for item in answer_query_options}
        ],
        format_func=lambda query: answer_query_labels[query],
        help="Select from the existing fixed governed query set only.",
    )
    run_audit_workflow = st.checkbox(
        "Run multi-query audit",
        value=False,
        help="Runs one retrieval and answer-assembly pass per selected fixed question.",
    )
    if run_audit_workflow:
        workflow = build_audit_workflow(
            selected_workflow_queries,
            top_k=selected_top_k,
        )
        _render_audit_workflow(workflow)
        st.markdown("**Combined audit packet export**")
        combined_export = build_cross_query_audit_export(workflow)
        _render_cross_query_export(combined_export)
    else:
        st.info(
            "Select the fixed queries to compare, then enable the workflow when you are ready to run the controlled multi-query review."
        )

    st.subheader("Advanced Workflow Contracts")
    st.caption(
        "Inspect workflow and task boundaries, including local Airflow-readiness notes. "
        "This area is informational only; it does not execute jobs."
    )
    _render_orchestration_summary(orchestration_summary)

    st.subheader("Advanced Plan Preview")
    st.caption(
        "Preview how a controlled request maps to intents, workflows, tasks, blockers, and review checkpoints. "
        "This shell does not execute workflows, trigger Airflow, or accept free-form prompts."
    )
    _render_agent_shell(agent_shell_context)

    st.subheader("Advanced Demo Readiness")
    st.caption(
        "Technical readiness status across registries, retrieval, reviewer surfaces, workflow contracts, "
        "Airflow scaffold, and the plan-preview shell. This section is read-only and does not execute workflows."
    )
    _render_final_system_readiness(final_readiness)
