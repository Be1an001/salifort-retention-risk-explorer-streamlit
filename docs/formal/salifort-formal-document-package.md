# Salifort Formal Documentation Package

## Document Control

| Field | Value |
| --- | --- |
| Document number | SALIFORT-DOC-PACK-001 |
| Project | Salifort Motors Retention Risk Explorer |
| Repository branch | `main` |
| Version | 1.1 |
| Status | Portfolio documentation package; formal docs and agent/eval guidance completed on main |
| Last reviewed | 2026-05-07 |

## Version Table

| Version | Date | Change Summary | Owner |
| --- | --- | --- | --- |
| 1.0 | 2026-05-07 | 建立正式文件索引、business-facing summary、HR quick start 與 maintenance rules | Project owner |
| 1.1 | 2026-05-07 | Updated document-control status after `docs/formal-docs-refresh` and `docs/add-agent-guidance-evals` were merged into `main` | Project owner |

## Prepared / Reviewed / Approved

| Role | Name | Status | Notes |
| --- | --- | --- | --- |
| Prepared by | Project owner | Drafted | Portfolio documentation refresh |
| Reviewed by | Not confirmed from project files | Not confirmed | 可於 portfolio review 時補齊；不代表真實公司審核 |
| Approved by | Not confirmed from project files | Not applicable | Portfolio demo 文件，不代表真實公司核准 |

## Purpose

本文件是正式文件包索引，協助 reviewer 快速找到產品、技術、部署、使用者、主管摘要與 MLOps evidence 文件。此文件不取代各 canonical document，而是提供文件治理與維護入口。

## Scope Boundary

此 repository 是 portfolio-grade decision-support demo：

- Not a production HR system。
- Not an employment decision system。
- Human review support only。
- Public app truth remains Weighted XGBoost at threshold `0.29`。
- MLOps Lab packaged demo model uses separate threshold `0.60`。
- OpenAI optional briefing receives compact aggregate JSON only, not raw CSV rows or PII。
- FastAPI、Docker Compose、MLflow、Airflow DAG 是 local/dev components，不在 hosted Streamlit Cloud 內執行。

## Documentation Index

| Document | Canonical Path | Primary Audience | Purpose |
| --- | --- | --- | --- |
| Product Requirements Document | [docs/product/product-requirements-document.md](../product/product-requirements-document.md) | Product, HR, reviewers | 背景、目標、使用者、功能、non-goals、roadmap |
| Technical Design and Architecture | [docs/technical/technical-design-and-architecture.md](../technical/technical-design-and-architecture.md) | Technical reviewers | 架構、artifact flow、MLOps flow、OpenAI boundaries |
| Environment Setup and Deployment Guide | [docs/deployment/environment-setup-and-deployment-guide.md](../deployment/environment-setup-and-deployment-guide.md) | Developers, reviewers | local setup、Streamlit Cloud、secrets、local/dev MLOps commands |
| User Manual | [docs/user-guide/user-manual.md](../user-guide/user-manual.md) | HR, managers, interviewers | 九頁 app 使用方式、business interpretation、FAQ |
| HR Quick Start | [docs/user-guide/hr-quick-start.md](../user-guide/hr-quick-start.md) | HR, HRBP, department managers | 一頁快速使用指南與 responsible-use reminders |
| Executive Summary | [docs/executive/executive-summary.md](../executive/executive-summary.md) | Executives, senior stakeholders | 非技術管理摘要與治理邊界 |
| Docs Index | [docs/README.md](../README.md) | All readers | 中央文件導覽 |
| MLOps Demo Guide | [docs/mlops-demo-guide.md](../mlops-demo-guide.md) | Technical reviewers | Hosted and local/dev MLOps demonstration path |
| MLOps Evidence Pack | [docs/demo-assets/mlops-evidence/README.md](../demo-assets/mlops-evidence/README.md) | Online reviewers | Sanitized local/dev evidence snapshots |

## Screenshot Placeholder Table

| Screenshot Needed | Suggested Page | Purpose | Status |
| --- | --- | --- | --- |
| App landing view | Overview | Show product framing and public model truth | Placeholder |
| Workforce filters | Workforce Explorer | Show interactive review flow | Placeholder |
| Threshold trade-off | Model & Threshold Lab | Show recall/precision/review workload discussion | Placeholder |
| SHAP explanation | Explainability | Show explainability without causal overclaiming | Placeholder |
| Manager summary | Manager Action View | Show department review prioritization | Placeholder |
| Online CSV Insight | MLOps Lab | Show hosted CSV sandbox and scoring modes | Placeholder |
| MLOps Evidence Pack | MLOps Lab | Show local/dev evidence snapshots | Placeholder |

## Canonical Document Rules

- PRD canonical location: `docs/product/product-requirements-document.md`。
- TDD canonical location: `docs/technical/technical-design-and-architecture.md`。
- Deployment canonical location: `docs/deployment/environment-setup-and-deployment-guide.md`。
- User manual canonical location: `docs/user-guide/user-manual.md`。
- HR quick start canonical location: `docs/user-guide/hr-quick-start.md`。
- Executive summary canonical location: `docs/executive/executive-summary.md`。
- Formal package canonical location: `docs/formal/salifort-formal-document-package.md`。
- Docs index canonical location: `docs/README.md`。

## Duplicate and Overlap Handling

- Root `README.md` remains the public entry point and should stay concise.
- `docs/README.md` is the documentation index and should link all important docs.
- `docs/user-guide/streamlit-app-walkthrough.md` is the canonical page-by-page walkthrough.
- `docs/streamlit-app-walkthrough.md` is a redirect stub only.
- `docs/navigator/wp*.md` files are implementation-history notes, not first-stop user documentation.
- MLOps runbooks remain scoped to local/dev details and should cross-link rather than duplicate the deployment guide.

## Maintenance Rules

- Update this package when canonical docs are created, renamed, or materially changed.
- Keep public model truth and MLOps Lab demo model truth separate.
- Do not add claims of production HR deployment, automated employment decisions, or hosted FastAPI/Docker/MLflow/Airflow services.
- Do not document secrets, local absolute paths, uploaded CSVs, `mlruns/`, generated model binaries, or private environment values.
- Keep OpenAI privacy language explicit: aggregate-only briefing payloads, no raw CSV rows, no PII.
- Prefer cross-links over duplicated prose when a topic already has a canonical document.
