# Zepto Data & AI Platform

A complete AI/ML project covering data collection and SQL analytics, exploratory data analysis and machine learning, and a local RAG-based customer support assistant.

The project is organized into three independent modules:

```text
zepto-data-ai-platform/
│
├── data_pipeline/
│   ├── scrape_and_load.py
│   ├── queries.py
│   ├── queries.sql
│   ├── query_6_output.csv
│   └── zepto_books.db
│
├── analytics/
│   ├── eda.py
│   ├── visual_analysis.py
│   ├── standardization.py
│   ├── modeling.py
│   ├── regression.py
│   ├── comparison.py
│   ├── persistence.py
│   ├── titanic.csv
│   ├── titanic_cleaned.csv
│   └── titanic_rf_pipeline.joblib
│
├── support_assistant/
│   ├── docs/
│   │   ├── doc_01_delivery.txt
│   │   ├── doc_02_returns_refunds.txt
│   │   ├── doc_03_membership.txt
│   │   ├── doc_04_tracking.txt
│   │   ├── doc_05_cancellation.txt
│   │   ├── doc_06_damaged_missing.txt
│   │   ├── doc_07_gift_cards.txt
│   │   └── doc_08_support_hours.txt
│   ├── ingest.py
│   ├── rag.py
│   ├── graph.py
│   ├── api.py
│   ├── chroma_db/
│   └── ingestion_report.txt
│
├── Dockerfile
├── requirements.txt
└── README.md
```

---

# 1. Setup

## Clone the repository

```bash
git clone https://github.com/irenejoy5349/zepto-data-ai-platform.git
cd zepto-data-ai-platform
```

## Create a virtual environment

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## Install dependencies

```bash
pip install -r requirements.txt
```

The repository uses one consolidated `requirements.txt` for all three modules.

---

# 2. Module 1 — Zepto Book Data Pipeline

## Objective

Scrape book information from the Books to Scrape website, clean the extracted data, convert prices from GBP to INR using a fixed exchange rate, store the data in normalized SQLite tables, and perform SQL and pandas analysis.

## Technologies

* Python
* requests
* BeautifulSoup
* pandas
* SQLite

## Scraping

The scraper collects books from three categories:

* Travel
* Mystery
* Historical Fiction

The project contains **69 book records** across these three categories.

The scraper captures:

* title
* price in GBP
* star rating text
* availability
* category

## Data cleaning

The raw values are converted into structured fields:

| Field                   | Cleaning decision                   |
| ----------------------- | ----------------------------------- |
| `price_gbp`             | Parsed into numeric float           |
| `rating`                | Converted into integer from 1 to 5  |
| `in_stock`              | Converted into boolean              |
| Numeric parsing failure | Median imputation where appropriate |
| Category                | Preserved from the source category  |

## Currency conversion

The required fixed conversion rate is used:

```text
1 GBP = 105.50 INR
```

No live currency API is required.

## SQLite schema

The cleaned data is stored in normalized SQLite tables:

### `categories`

```text
category_id  PRIMARY KEY
category_name
```

### `books`

```text
book_id      PRIMARY KEY
category_id  FOREIGN KEY → categories.category_id
title
price_gbp
price_inr
rating
in_stock
```

This provides a normalized primary-key / foreign-key relationship.

## Running the pipeline

```bash
python data_pipeline/scrape_and_load.py
```

The script creates or refreshes:

```text
data_pipeline/zepto_books.db
```

## SQL analysis

The repository contains SQL queries covering:

* SELECT / WHERE
* ORDER BY
* LIMIT
* DISTINCT
* IN
* BETWEEN
* JOIN

The queries are saved in:

```text
data_pipeline/queries.sql
```

Example query output is also stored in:

```text
data_pipeline/query_6_output.csv
```

## SQL + pandas equivalence

The project uses `pd.read_sql()` for SQL results and reproduces the relational JOIN in memory using `pd.merge()`.

Run:

```bash
python data_pipeline/queries.py
```

---

# 3. Module 2 — Titanic Analytics and Machine Learning

## Objective

Perform end-to-end exploratory data analysis, preprocessing, classification, imbalance handling, hyperparameter tuning, regression, and model persistence using the Titanic dataset.

## Dataset loading

The Titanic dataset is loaded from seaborn once:

```python
sns.load_dataset("titanic")
```

The raw dataset is immediately cached as:

```text
analytics/titanic.csv
```

Subsequent modeling and analysis use the saved local dataset for reproducibility.

## Data cleaning

The cleaned dataset contains **889 rows**.

Missing-value decisions:

| Column        | Missing percentage | Decision                                 |
| ------------- | -----------------: | ---------------------------------------- |
| `deck`        |             77.22% | Dropped because of very high missingness |
| `age`         |             19.87% | Median imputation                        |
| `embarked`    |              0.22% | Affected rows dropped                    |
| `embark_town` |              0.22% | Affected rows dropped                    |

The cleaned dataset is saved as:

```text
analytics/titanic_cleaned.csv
```

---

# 4. Univariate Analysis and Outliers

## Age

Age is examined using:

* histogram
* boxplot
* IQR-based outlier detection

Number of detected age outliers:

```text
65
```

## Fare

Fare is examined using:

* histogram
* boxplot
* IQR-based outlier detection

Number of detected fare outliers:

```text
114
```

Fare statistics:

| Statistic |   Value |
| --------- | ------: |
| Mean      | 32.0967 |
| Median    | 14.4542 |
| Mode      |  8.0500 |
| Skewness  |  4.8014 |

The mean is much higher than the median and the mode, which is consistent with a **strong right-skewed distribution** caused by high-fare observations.

---

# 5. Survival Analysis

## Survival by sex

```text
Female: 74.04%
Male:   18.89%
```

Female passengers had substantially higher observed survival than male passengers.

## Survival by passenger class

```text
1st class: 62.62%
2nd class: 47.28%
3rd class: 24.24%
```

Survival decreased from first to third class.

## Survival by sex and passenger class

The combined analysis shows that sex and passenger class together provide a stronger separation of survival outcomes than either variable alone.

---

# 6. Correlation Analysis

The required six-column correlation matrix contains exactly:

```text
survived
pclass
age
sibsp
parch
fare
```

The following columns are intentionally excluded:

```text
adult_male
alone
```

Top two absolute off-diagonal correlation pairs:

| Pair              | Correlation |
| ----------------- | ----------: |
| `pclass` ↔ `fare` |     -0.5482 |
| `sibsp` ↔ `parch` |      0.4145 |

### Interpretation

The negative correlation between passenger class and fare shows that higher-class passengers generally paid higher fares.

The positive relationship between `sibsp` and `parch` indicates that passengers traveling with siblings/spouses were also more likely to travel with parents/children.

---

# 7. Visual Exploratory Analysis

The project produces **nine distinct charts**, including **four clearly multivariate charts**, with written interpretations.

## Chart 1 — Survival Rate by Sex

Female passengers had a survival rate of approximately 74.04%, compared with 18.89% for male passengers. This large difference indicates that sex was strongly associated with survival.

## Chart 2 — Survival Rate by Passenger Class

Survival decreases from 62.62% in first class to 47.28% in second class and 24.24% in third class. This shows a strong relationship between passenger class and survival.

## Chart 3 — Survival Rate by Sex and Class

Combining sex and passenger class reveals a stronger survival pattern than either variable alone. Female passengers generally had higher survival rates within each class, while third-class passengers had lower survival rates.

## Chart 4 — Fare Distribution by Survival

Surviving passengers generally show a higher fare distribution than non-survivors. The plot also highlights the high-fare outliers identified during IQR analysis.

## Chart 5 — Age Distribution by Survival

The age distributions overlap substantially, suggesting that age alone does not separate survival as clearly as sex or passenger class. Age can still contribute useful information when combined with other features.

## Chart 6 — Age vs Fare by Survival Status

This multivariate chart combines age and fare while separating passengers by survival status. Survivors are more frequently observed among passengers with higher fares, while age shows substantial overlap between the groups. The combined view demonstrates the value of considering multiple numerical features together.

## Chart 7 — Fare by Passenger Class and Survival Status

This multivariate chart combines passenger class, fare, and survival status. Fare distributions differ substantially across classes, while comparing survival within each class provides a more detailed view than examining fare or class alone.

## Chart 8 — Age by Passenger Class and Survival Status

This multivariate chart combines passenger class, age, and survival status. Age distributions vary across passenger classes and also differ between survivors and non-survivors, demonstrating why age should be interpreted together with other passenger characteristics.

## Chart 9 — Family Relationships and Survival Status

This multivariate chart combines `sibsp`, `parch`, and survival status. Most passengers are concentrated at low family-count values, while larger family structures are less common. The overlap between survival groups indicates that family variables alone do not explain survival but can contribute useful predictive information.

---

# 8. Standardization

Exploratory standardization is performed for:

```text
age
fare
```

Z-score standardization is applied using `StandardScaler`.

Before and after scaling, the project checks the mean and standard deviation to demonstrate the effect of standardization.

Importantly, this exploratory standardization is separate from the predictive preprocessing pipeline.

---

# 9. Train/Test Split

The dataset is split before fitting preprocessing components.

The split is stratified on the target:

```python
train_test_split(
    ...,
    stratify=y
)
```

Final split:

```text
Training rows: 711
Test rows:     178
```

Class distribution is preserved:

```text
Not survived: 61.74%
Survived:     38.26%
```

---

# 10. Classification Models

The following classifiers are evaluated on the same train/test split:

* Logistic Regression
* Decision Tree
* Random Forest

Preprocessing is implemented using `Pipeline` and `ColumnTransformer` so that imputation, encoding, and scaling are fitted only on training data.

## Baseline results

| Model               | Accuracy | Precision | Recall |     F1 | ROC-AUC |
| ------------------- | -------: | --------: | -----: | -----: | ------: |
| Logistic Regression |   0.8090 |    0.7833 | 0.6912 | 0.7344 |  0.8610 |
| Decision Tree       |   0.7640 |    0.7600 | 0.5588 | 0.6441 |  0.8374 |
| Random Forest       |   0.8090 |    0.7656 | 0.7206 | 0.7424 |  0.8196 |

The Decision Tree produced the weakest recall and F1 among the three baseline models.

Logistic Regression produced the strongest ROC-AUC, while Random Forest produced the highest baseline F1 and recall.

---

# 11. Class Imbalance Analysis

Three approaches are compared:

1. Baseline model
2. `class_weight="balanced"`
3. SMOTE applied only to training data

| Variant               | Precision | Recall |     F1 |
| --------------------- | --------: | -----: | -----: |
| Baseline              |    0.7833 | 0.6912 | 0.7344 |
| Class Weight Balanced |    0.7500 | 0.7338 | 0.7338 |
| SMOTE                 |    0.7353 | 0.7353 | 0.7353 |

The balanced approaches improve recall but reduce precision.

SMOTE is applied only to the training portion to avoid leaking synthetic samples into the test set.

The final choice therefore considers the trade-off between detecting survivors and maintaining precision.

---

# 12. Random Forest Hyperparameter Tuning

GridSearchCV is used to tune the Random Forest model over:

```python
{
    "model__n_estimators": [100, 200],
    "model__max_depth": [None, 5, 10],
    "model__max_features": ["sqrt", "log2"]
}
```

## Best parameters

```text
n_estimators = 200
max_depth = 5
max_features = sqrt
```

Best cross-validation F1:

```text
0.7408
```

Final tuned Random Forest uses:

```python
oob_score=True
```

OOB score:

```text
0.8214
```

## Tuned test performance

| Metric    |  Value |
| --------- | -----: |
| Accuracy  | 0.8315 |
| Precision | 0.8654 |
| Recall    | 0.6618 |
| F1        | 0.7500 |
| ROC-AUC   | 0.8389 |

Confusion matrix:

```text
TN = 103
FP = 7
FN = 23
TP = 45
```

The tuned Random Forest improves accuracy, precision and F1 compared with the baseline Random Forest.

---

# 13. Regression

A Linear Regression model is used to predict:

```text
fare
```

Regression performance:

| Metric      |   Value |
| ----------- | ------: |
| MAE         | 21.0986 |
| RMSE        | 41.7021 |
| R²          |  0.3482 |
| Adjusted R² |  0.3091 |

The residual analysis shows that the spread of residuals changes across predicted values, indicating **possible heteroscedasticity**.

The model explains a meaningful but limited portion of fare variance, so fare prediction remains challenging.

---

# 14. Final Model Recommendation

The tuned Random Forest is selected as the preferred classifier because it achieved the highest test F1 score and strong accuracy and precision while maintaining reasonable recall.

Logistic Regression achieved the strongest ROC-AUC among the baseline models, making it a useful interpretable benchmark.

Decision Tree performed worse than the other classifiers on the main evaluation metrics.

For this project, the tuned Random Forest provides the best overall classification balance and is therefore used as the persisted final model.

---

# 15. Model Persistence

The complete preprocessing + Random Forest estimator pipeline is saved using `joblib`.

Saved artifact:

```text
analytics/titanic_rf_pipeline.joblib
```

The project also demonstrates:

1. loading the saved pipeline,
2. reloading it from disk,
3. passing a raw passenger record,
4. generating a prediction.

This ensures the saved artifact includes the preprocessing steps required before inference.

---

# 16. Module 3 — Zepto Customer Support Assistant

## Objective

Build a local Retrieval-Augmented Generation-style customer support assistant using policy documents, local embeddings, ChromaDB, LangGraph and FastAPI.

The system supports a deterministic `MOCK_LLM=1` baseline so that the project can run locally without a paid API.

---

# 17. Support Knowledge Base

The knowledge base contains eight focused policy documents:

```text
doc_001 – Delivery
doc_002 – Returns and Refunds
doc_003 – Membership
doc_004 – Tracking
doc_005 – Cancellation
doc_006 – Damaged / Missing Items
doc_007 – Gift Cards
doc_008 – Customer Support Hours
```

Each short policy document is kept as a focused knowledge chunk.

---

# 18. Embeddings

The embedding model is:

```text
all-MiniLM-L6-v2
```

Embedding dimension:

```text
384
```

Embeddings are normalized before storage and retrieval.

---

# 19. ChromaDB

ChromaDB is stored persistently under:

```text
support_assistant/chroma_db/
```

Collection name:

```text
zepto_support
```

The collection explicitly uses cosine distance:

```python
"hnsw:space": "cosine"
```

Documents and query embeddings are normalized, and the retrieval step returns the top three results.

The ingestion script also creates an ingestion report containing the document count, embedding dimension, collection name, distance metric, and sample retrieval results.

Run ingestion with:

```bash
python support_assistant/ingest.py
```

---

# 20. Prompt Design

The support assistant prompt includes:

* Role
* Context
* Task
* Output format
* Length constraint
* Negative constraint
* Few-shot examples

The negative constraint instructs the model not to invent unsupported policy information.

This keeps generated answers grounded in retrieved support content.

---

# 21. LangGraph Workflow

The support assistant uses a `StateGraph` with three required nodes:

```text
classify_intent
retrieve_and_answer
direct_answer
```

Flow:

```text
                ┌─────────────────────────┐
                │     classify_intent     │
                └────────────┬────────────┘
                             │
                 ┌───────────┴───────────┐
                 │                       │
        policy_question          general_question
                 │                       │
                 ▼                       ▼
     retrieve_and_answer          direct_answer
                 │                       │
                 └───────────┬───────────┘
                             ▼
                          Response
```

---

# 22. Mock Mode

The default baseline uses:

```text
MOCK_LLM=1
```

In mock mode, intent classification uses deterministic keyword matching.

Policy-related keywords include:

```text
delivery
return
refund
membership
tracking
cancel
gift card
support hours
```

Policy queries are routed to retrieval.

General questions are routed to the direct-answer path.

The retrieval path still performs the actual embedding query and top-3 ChromaDB retrieval in mock mode.

The mock response is deterministic and based on the retrieved context.

---

# 23. Structured Output

The API response is validated with Pydantic.

The response contains:

```text
answer
sources
confidence
```

`confidence` is constrained to the range:

```text
0 to 1
```

In real-LLM mode, invalid structured output can trigger corrective retries before falling back.

---

# 24. FastAPI

The support assistant exposes:

```text
POST /ask
```

Request format:

```json
{
  "query": "How long does a refund take?"
}
```

The local server can be started using:

```bash
uvicorn support_assistant.api:app --host 0.0.0.0 --port 8000
```

Swagger UI:

```text
http://localhost:8000/docs
```

Example API endpoint:

```text
http://localhost:8000/ask
```

---

# 25. Example Support Queries

## Example 1 — Policy question

```json
{
  "query": "How long does a refund take?"
}
```

In mock mode, the classifier identifies this as a policy question and the workflow performs ChromaDB retrieval before returning a grounded response.

Retrieved sources include support policy documents related to refunds.

## Example 2 — General question

```json
{
  "query": "Hello, how are you?"
}
```

This is classified as a general question and routed to the direct-answer node without policy retrieval.

## Support-hours query

```json
{
  "query": "What are the customer support hours?"
}
```

Because `support hours` is a policy keyword, this query is classified as a policy question and routed through:

```text
classify_intent
      ↓
retrieve_and_answer
```

The relevant support-hours document is therefore retrieved from ChromaDB.

---

# 26. RAG Architecture

The overall support-assistant architecture is:

```text
Policy Documents
       ↓
Document Ingestion
       ↓
MiniLM Embeddings
       ↓
Persistent ChromaDB
       ↓
User Query
       ↓
Query Embedding
       ↓
Top-3 Similarity Retrieval
       ↓
Prompt + Retrieved Context
       ↓
Answer Generation
       ↓
Pydantic Structured Response
       ↓
FastAPI
```

The deterministic mock mode demonstrates the complete retrieval and routing flow without requiring an external paid LLM service.

---

# 27. Docker

The project includes a Dockerfile for local execution.

Build:

```bash
docker build -t zepto-data-ai:latest .
```

Run:

```bash
docker run --rm -p 8000:8000 zepto-data-ai:latest
```

The Docker baseline uses:

```text
MOCK_LLM=1
```

so the application can run locally without external API credentials.

---

# 28. Useful Commands

## Data pipeline

```bash
python data_pipeline/scrape_and_load.py
python data_pipeline/queries.py
```

## Titanic analytics

```bash
python analytics/eda.py
python analytics/visual_analysis.py
python analytics/standardization.py
python analytics/modeling.py
python analytics/regression.py
python analytics/comparison.py
python analytics/persistence.py
```

## Support assistant

```bash
python support_assistant/ingest.py
uvicorn support_assistant.api:app --host 0.0.0.0 --port 8000
```

## Docker

```bash
docker build -t zepto-data-ai:latest .
docker run --rm -p 8000:8000 zepto-data-ai:latest
```

---

# 29. Reproducibility Notes

The project is designed so that the major datasets and generated artifacts can be reproduced locally.

Important reproducibility decisions include:

* fixed GBP → INR conversion rate of 105.50,
* cached Titanic dataset,
* deterministic train/test split,
* train-only preprocessing,
* deterministic mock support-assistant baseline,
* explicit ChromaDB cosine distance,
* persistent local vector database,
* saved fitted machine-learning pipeline.

The support assistant can regenerate its ChromaDB index using:

```bash
python support_assistant/ingest.py
```

---

# 30. Git Workflow

The repository development history includes a feature branch workflow:

```text
feature/capstone-final
        ↓
multiple commits
        ↓
merged/integrated into main
```

The repository therefore preserves the required feature-branch development workflow rather than showing only a single direct development path.

---

# 31. Final Checklist

## Module 1 — Data Pipeline

* [x] Requests + BeautifulSoup scraping
* [x] 69 books
* [x] 3 categories
* [x] Required fields
* [x] Data cleaning
* [x] Fixed 105.50 GBP→INR conversion
* [x] Normalized SQLite schema
* [x] PK/FK relationship
* [x] Required SQL operations
* [x] JOIN
* [x] `pd.read_sql`
* [x] `pd.merge`

## Module 2 — Analytics

* [x] Titanic dataset cached locally
* [x] Missing-value analysis
* [x] IQR outliers
* [x] Fare statistics and skewness
* [x] Survival analysis
* [x] Six-column correlation matrix
* [x] Nine visual charts
* [x] Four clearly multivariate charts
* [x] Standardization
* [x] Stratified split
* [x] Train-only preprocessing
* [x] Logistic Regression
* [x] Decision Tree
* [x] Random Forest
* [x] Required classification metrics
* [x] Imbalance comparison
* [x] SMOTE
* [x] GridSearchCV
* [x] OOB score
* [x] Linear Regression
* [x] Regression metrics
* [x] Residual analysis
* [x] Final model comparison
* [x] Persisted complete pipeline
* [x] Reload + prediction

## Module 3 — Support Assistant

* [x] Eight policy documents
* [x] MiniLM embeddings
* [x] 384-dimensional vectors
* [x] ChromaDB
* [x] Explicit cosine distance
* [x] Top-3 retrieval
* [x] Prompt anatomy
* [x] Negative constraint
* [x] Few-shot examples
* [x] LangGraph
* [x] Required graph nodes
* [x] Intent routing
* [x] Mock baseline
* [x] Structured Pydantic output
* [x] Confidence validation
* [x] Real-LLM retry path
* [x] FastAPI `/ask`
* [x] Swagger documentation
* [x] Docker
* [x] Reproducible local execution

---

# Conclusion

This repository demonstrates an end-to-end progression from data ingestion and SQL analysis to machine learning, model persistence, semantic retrieval, LangGraph orchestration, structured LLM responses, FastAPI serving, and Dockerized local deployment.

The project is designed to be reproducible locally and to demonstrate practical AI/ML engineering decisions across the full data and AI application lifecycle.
