from __future__ import annotations

import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MLOPS_LAB_PAGE = REPO_ROOT / "app" / "pages" / "mlops_lab.py"
REQUIREMENTS = REPO_ROOT / "requirements.txt"
MLOPS_REQUIREMENTS = REPO_ROOT / "requirements-mlops.txt"


def _page_text() -> str:
    return MLOPS_LAB_PAGE.read_text(encoding="utf-8")


def test_mlops_lab_has_no_shell_or_workflow_execution_calls() -> None:
    text = _page_text()

    forbidden = [
        "subprocess.run",
        "os.system",
        "Popen(",
        "airflow dags trigger",
        "git push",
    ]
    for phrase in forbidden:
        assert phrase not in text


def test_external_api_and_openai_calls_are_user_triggered() -> None:
    text = _page_text()

    assert 'st.button("Run external API batch scoring")' in text
    assert 'st.button("Generate AI briefing")' in text
    assert '"/batch-predict"' in text
    assert "client.responses.create" in text
    assert "Uploaded CSVs are kept in memory" in text


def test_no_api_or_openai_calls_at_module_import_level() -> None:
    tree = ast.parse(_page_text())
    top_level_calls = [
        node
        for statement in tree.body
        for node in ast.walk(statement)
        if isinstance(node, ast.Call) and not isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]

    rendered_names = {getattr(call.func, "id", "") for call in top_level_calls}
    rendered_attrs = {getattr(call.func, "attr", "") for call in top_level_calls}
    assert "_api_json_request" not in rendered_names
    assert "_generate_ai_briefing" not in rendered_names
    assert "create" not in rendered_attrs


def test_payload_and_summary_helpers_exclude_pii_like_fields() -> None:
    text = _page_text()

    assert "PII_COLUMN_HINTS" in text
    assert '"email"' in text
    assert '"employee"' in text
    assert '"phone"' in text
    assert '"address"' in text
    assert 'normalized[API_FEATURE_COLUMNS]' in text
    assert '"left"' in text
    assert "Target column `left` was excluded from API payload." in text
    assert "compact aggregate JSON" in text


def test_requirements_define_openai_only_for_streamlit_runtime() -> None:
    runtime_requirements = REQUIREMENTS.read_text(encoding="utf-8")
    mlops_requirements = MLOPS_REQUIREMENTS.read_text(encoding="utf-8")

    assert "openai>=2,<3" in runtime_requirements
    assert "openai" not in mlops_requirements.lower()


def test_salifort_api_url_is_optional_in_ui() -> None:
    text = _page_text()

    assert 'SALIFORT_API_URL' in text
    assert "No external FastAPI endpoint is configured." in text
    assert "_config_value(\"SALIFORT_API_URL\")" in text
