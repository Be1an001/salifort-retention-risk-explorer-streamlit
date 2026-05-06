# HR 快速上手一頁指南

本頁提供 HR、HRBP、部門主管與非技術主管快速理解 **Salifort Motors Retention Risk Explorer** 的方式。

## 這個系統做什麼

這是一個 portfolio-grade decision-support demo，用來協助檢視員工留任風險模式、部門暴露情形、模型門檻取捨與負責任使用邊界。它不是 production HR system，也不是 employment decision system。

## 建議頁面順序

1. **Overview:** 先了解商業問題、資料、模型與負責任使用邊界。
2. **Workforce Explorer:** 用部門、薪資、年資與風險旗標篩選人力切片。
3. **EDA & Patterns:** 檢視工作量、滿意度、薪資、部門與年資模式。
4. **Model & Threshold Lab:** 用商業語言理解 recall、precision、threshold 與 review workload。
5. **Explainability:** 用 SHAP 理解模型依據哪些特徵產生訊號。
6. **Manager Action View:** 將部門暴露情形轉成主管可討論的優先順序。
7. **Methods & Limitations:** 確認限制、資料邊界、fallback 邏輯與 responsible-use 規則。
8. **PACE Navigator:** 技術審查時再看，可檢查 evidence、citations、workflow readiness。
9. **MLOps Lab:** 技術展示時再看，可檢查 Online CSV Insight、packaged demo model inference、MLOps Evidence Pack 與 local/dev MLOps 路徑。

## 各頁用途

| Page | HR 用途 |
| --- | --- |
| Overview | 快速建立專案背景與風險邊界 |
| Workforce Explorer | 篩選部門與族群，找出需要人工 review 的區域 |
| EDA & Patterns | 用圖表理解工作量、滿意度與離職模式 |
| Model & Threshold Lab | 理解門檻調整如何影響捕捉率與誤報 |
| Explainability | 看模型主要依據哪些特徵，不做因果宣稱 |
| Manager Action View | 將模型訊號轉成管理討論與優先 review |
| Methods & Limitations | 確認哪些事情不能宣稱或自動化 |
| PACE Navigator | 給 reviewer 檢查證據、引用與治理結構 |
| MLOps Lab | 展示 hosted CSV insight 與 local/dev MLOps evidence |

## 如何解讀 High / Medium / Low

- **High:** 優先安排人工 review，搭配工作量、主管情境、升遷紀錄與團隊情況判讀。
- **Medium:** 持續觀察，與部門基準和主管情境一起討論。
- **Low:** 沒有立即模型訊號，但仍應遵循一般 HR review 流程。

重要：High / Medium / Low 是 review priority，不是 HR 決策，也不是員工排名。

## HR 下一步可以做什麼

- 從部門層級開始，而不是直接跳到個別員工判斷。
- 把模型訊號與工作量、滿意度、升遷、薪資與主管觀察一起討論。
- 對 High band 或高暴露部門安排人工 review。
- 使用 Manager Action View 準備管理層討論。
- 在對外展示前先閱讀 Methods & Limitations。

## HR 不應該做什麼

- 不要把模型分數當成解雇、懲處、升遷或薪資決策。
- 不要宣稱這是 production HR platform。
- 不要宣稱這能預測真實公司員工未來行為。
- 不要把 SHAP 或 feature importance 說成離職原因。
- 不要把 Online CSV Insight 的 heuristic score 當成 public weighted XGBoost model probability。
- 不要把 MLOps Lab packaged demo model threshold `0.60` 和 public app threshold `0.29` 混在一起。

## 隱私與 OpenAI 邊界

Online CSV Insight 的 optional OpenAI briefing 只使用 compact aggregate JSON。Raw CSV rows、PII 與 identifier-like fields 不會送到 OpenAI。

## 最安全的一句話

這個專案最適合被說明為一個 Streamlit portfolio decision-support app，用來展示 retention-risk analytics、threshold trade-off、SHAP explainability、responsible-use framing 與 local/dev MLOps evidence，而不是 production HR automation。
