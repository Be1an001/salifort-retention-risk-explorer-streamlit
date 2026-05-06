# Executive Summary

## 商業問題

Salifort Motors Retention Risk Explorer 用 portfolio demo 的形式回答一個管理問題：如何及早辨識員工留任風險、集中主管 review 注意力，同時避免把模型分數當成自動化 HR 決策。

## 產品價值

此專案把 HR attrition analysis 包裝成九頁 Streamlit app，讓非技術與技術 reviewer 都能快速理解資料、模型、threshold trade-off、SHAP explanation、部門暴露情形、負責任使用邊界與 local/dev MLOps evidence。

最重要的價值不是「自動做 HR 決策」，而是協助管理者更有結構地討論風險訊號。

## 主要能力

- **Workforce exploration:** 依部門、薪資、年資與風險旗標檢視不同人力切片。
- **Model and threshold review:** 說明 public model truth 為 Weighted XGBoost at threshold `0.29`，並以 recall、precision 與 review workload 說明門檻取捨。
- **Explainability:** 使用 SHAP 解釋模型行為，但不做因果宣稱。
- **Manager Action View:** 將部門暴露情形轉成管理討論優先順序。
- **PACE Navigator:** 提供 evidence、citations、source preview、workflow contracts 與 preview-only planning surface。
- **MLOps Lab:** 提供 Online CSV Insight、heuristic review scoring、packaged demo model inference、MLOps Evidence Pack 與 local/dev FastAPI、Docker Compose、MLflow、Airflow DAG、GitHub Actions CI 證據路徑。

## 管理層如何解讀

- 模型分數是 review cue，不是 final decision。
- High / Medium / Low band 是優先順序語言，不是員工價值判斷。
- Public app model 使用 threshold `0.29`；MLOps Lab packaged demo model 使用 separate lab threshold `0.60`。
- 部門暴露情形應用於安排人工 review、討論工作負荷與管理情境，而不是自動化處置員工。

## 風險與治理

此專案明確保留以下邊界：

- 不是 production HR system。
- 不是 employment decision system。
- Streamlit hosted app 不會 retrain model、run FastAPI、start Docker、run MLflow、trigger Airflow 或執行 CI。
- FastAPI、Docker Compose、MLflow、Airflow DAG 是 local/dev MLOps components。
- Optional OpenAI briefing 只接收 compact aggregate JSON；raw CSV rows 與 PII 不會送出。
- SHAP 只說明 model behavior，不證明 attrition cause。

## One-Slide Style Summary

| Area | Summary |
| --- | --- |
| Problem | Identify retention-risk patterns responsibly |
| Product | Nine-page Streamlit decision-support portfolio app |
| Public model truth | Weighted XGBoost, threshold `0.29` |
| Management use | Prioritize human review and department-level discussion |
| Technical proof | Artifact-backed app, PACE Navigator, MLOps Evidence Pack, CI |
| MLOps boundary | Hosted CSV insight plus local/dev pipeline/API/Docker/MLflow/Airflow |
| Responsible-use boundary | Human review support only, not employment decisions |

## Executive Takeaway

This project demonstrates how an analytics model can be translated into a review-ready product with governance language, stakeholder interpretation, and technical evidence. It should be presented as a portfolio-grade HR analytics decision-support demo, not a deployed HR operating system.
