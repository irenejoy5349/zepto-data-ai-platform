# Zepto Data & AI Platform

This project combines three practical components into one local AI and data platform:

1. A web data pipeline that collects and stores book information.
2. An analytics and machine learning workflow built around the Titanic dataset.
3. A local customer-support assistant using semantic retrieval and LangGraph.

The workflow is designed to run locally without requiring a paid AI API.

---

## Setup

This repository uses **one consolidated `requirements.txt`** for all three modules.

Install the dependencies with:

```bash
pip install -r requirements.txt
```

The same environment is used for the `/data_pipeline`, `/analytics`, and `/support_assistant` modules.

---

## Project Layout

```text
zepto-data-ai-platform/
├── data_pipeline/
├── analytics/
├── support_assistant/
├── Dockerfile
├── requirements.txt
└── README.md
```

---

# 1. Data Pipeline

The data pipeline collects book information from the Books to Scrape website using Python.

The scraper uses `requests` to retrieve pages and `BeautifulSoup` to extract the required fields.

## Categories Collected

The implementation covers:

* Travel
* Mystery
* Historical Fiction

The final cleaned dataset contains **69 records across 3 categories**.

## Fields Captured

Each record includes:

* title
* price in GBP
* star rating
* availability
* category

The source values are cleaned into useful types:

* price → floating-point `price_gbp`
* rating → integer from 1 to 5
* availability → Boolean `in_stock`

The pipeline handles unexpected parsing failures without stopping the complete run. Numeric parsing failures are handled with a median-based fallback where applicable, while unrecoverable invalid rows are dropped rather than allowing malformed data to stop the pipeline.

## Currency Conversion

A fixed project-defined conversion rate is used:

```text
1 GBP = 105.50 INR
```

The INR value is calculated locally:

```text
price_inr = price_gbp * 105.50
```

No external currency service is required.

## SQLite Design

The normalized database is:

```text
data_pipeline/zepto_books.db
```

It contains two related tables:

```text
categories
    category_id (Primary Key)
    category_name (UNIQUE)

books
    book_id (Primary Key)
    title
    price_gbp
    price_inr
    rating
    in_stock
    category_id (Foreign Key)
```

Separating categories into their own table avoids storing the same category text repeatedly and provides the required primary-key/foreign-key relationship.

## SQL Analysis

The SQL workflow demonstrates:

* `SELECT` and `WHERE`
* `ORDER BY`
* `LIMIT`
* `DISTINCT`
* `IN`
* `BETWEEN`
* `JOIN`

The executed queries and their outputs are saved by the project scripts.

SQL results are also loaded through pandas using `pd.read_sql(...)`.

The category-book relationship is independently reproduced using `pd.merge(...)` on in-memory DataFrames, and the results are compared for equivalence.

## Running the Pipeline

The data pipeline script performs the scraping, cleaning, fixed-rate currency conversion, normalized SQLite loading, and query-output generation.

```bash
python data_pipeline/scrape_and_load.py
python data_pipeline/queries.py
```

The repository also contains the generated SQLite database and saved query outputs for reproducibility.

---

# 2. Analytics and Machine Learning

The analytics workflow uses the Titanic dataset for exploratory analysis and predictive modeling.

The original dataset is cached locally as:

```text
analytics/titanic.csv
```

The cleaned dataset is:

```text
analytics/titanic_cleaned.csv
```

The modeling workflow includes Logistic Regression, Decision Tree, and Random Forest classifiers with training-only preprocessing, confusion matrices, ROC/AUC, and a side-by-side metrics table.

Class imbalance is compared with a baseline, `class_weight="balanced"`, and training-only SMOTE.

Random Forest tuning uses GridSearchCV with an OOB-enabled estimator.

The cleaned dataset contains **889 rows**.

## Data Cleaning

The cleaning decisions are based on the measured missing-value percentages and the required threshold rule:

| Column        | Missing | Decision           | Reason                                                        |
| ------------- | ------: | ------------------ | ------------------------------------------------------------- |
| `deck`        |  77.22% | Drop column        | The missing rate is too high for reliable imputation.         |
| `age`         |  19.87% | Median imputation  | This falls in the 5%–30% range, so imputation is appropriate. |
| `embarked`    |   0.22% | Drop affected rows | The missing rate is below 5%, so affected rows are removed.   |
| `embark_town` |   0.22% | Drop affected rows | The missing rate is below 5%, so affected rows are removed.   |

After cleaning, the dataset contains no remaining missing values.

## Exploratory Analysis

Histograms and boxplots are used to examine `age` and `fare`.

Using the IQR rule:

* Age outliers: **65**
* Fare outliers: **114**

Fare statistics:

```text
Mean     = 32.0967
Median   = 14.4542
Mode     = 8.0500
Skewness = 4.8014
```

The ordering `mean > median > mode`, together with the strongly positive skewness, indicates that the Fare distribution is strongly **right-skewed**. A relatively small number of high-fare observations pulls the mean substantially above the median.

## Survival Analysis

Boolean masking with `&` and `|` is used for combined survival analysis.

Observed survival rates:

```text
Female = 74.04%
Male   = 18.89%
```

By passenger class:

```text
1st class = 62.62%
2nd class = 47.28%
3rd class = 24.24%
```

The sex breakdown shows a large difference in observed survival, with female passengers surviving at a much higher rate than male passengers.

Passenger class also shows a strong relationship with survival: first-class passengers had the highest observed survival rate, while third-class passengers had the lowest.

Combining sex and passenger class gives an even stronger separation of survival outcomes. Female passengers in higher classes had the strongest observed survival, while male passengers in lower classes had substantially lower survival.

## Correlation Analysis

The required correlation matrix contains exactly these six columns:

```text
survived
pclass
age
sibsp
parch
fare
```

The boolean-derived columns `adult_male` and `alone` are excluded because they are derived/redundant flags rather than independent measured features.

The two strongest absolute off-diagonal correlations are:

```text
pclass ↔ fare = -0.5482
sibsp  ↔ parch = 0.4145
```

These are the two feature pairs with the largest absolute off-diagonal correlation coefficients.

The negative `pclass`–`fare` correlation indicates that higher fares are associated with lower numeric class values, where first class is encoded as 1. The positive `sibsp`–`parch` correlation suggests that passengers traveling with siblings/spouses also tended to travel with parents/children, reflecting family-group structure.

A correlation heatmap is also generated.

## Visual EDA

The project produces five distinct charts, each with an accompanying written interpretation.

### 1. Survival Rate by Sex

Female passengers had a survival rate of approximately 74.04%, compared with 18.89% for male passengers. This large difference indicates that sex was strongly associated with survival in the Titanic dataset.

### 2. Survival Rate by Passenger Class

Survival decreases from 62.62% in first class to 47.28% in second class and 24.24% in third class. This shows a strong relationship between passenger class and survival, with higher-class passengers having better observed survival rates.

### 3. Survival Rate by Sex and Class

The combined sex-and-class analysis separates passenger groups more clearly than either variable alone. Female passengers in higher classes had the strongest observed survival outcomes, while male passengers in lower classes had substantially lower survival.

### 4. Fare Distribution by Survival

Surviving passengers generally show a higher fare distribution than non-survivors. The plot also highlights the extreme high-fare observations identified by the IQR outlier analysis.

### 5. Age Distribution by Survival

The age distributions overlap substantially, but survival outcomes vary across different age ranges. This suggests that age contributes useful information, although its visual separation is weaker than the patterns observed for sex and passenger class.

## Standardization

Age and Fare are standardized using `StandardScaler`.

The z-score transformation is:

```text
z = (x - mean) / std
```

After standardization, both features are centered around approximately zero with a standard deviation close to one.

This exploratory experiment is kept separate from the final predictive workflow. The actual model pipeline performs preprocessing only after the train/test split.

---

# 3. Classification

The classification target is `survived`.

An 80/20 **stratified train-test split** is used:

```text
Training rows = 711
Testing rows  = 178
```

The survived/not-survived class balance is approximately:

```text
Not survived = 61.74%
Survived     = 38.26%
```

Because the target classes are not perfectly balanced, stratification is used to preserve approximately the same class proportions in both the training and test sets.

## Training Preprocessing

Preprocessing is performed only after the train/test split.

The predictive workflow ensures that:

* missing-value imputation is fitted on training data only
* categorical encoding is fitted on training data only
* numeric scaling with `StandardScaler` is fitted on training data only
* the fitted transformations are applied to the test data in transform-only mode

This prevents test-set information from leaking into model training.

## Models

Three classifiers are trained on the identical train/test split:

* Logistic Regression
* Decision Tree
* Random Forest

The Decision Tree is visualized using `plot_tree` with feature names and class names.

## Baseline Results

| Model               | Accuracy | Precision | Recall |     F1 | ROC-AUC |
| ------------------- | -------: | --------: | -----: | -----: | ------: |
| Logistic Regression |   0.8090 |    0.7833 | 0.6912 | 0.7344 |  0.8610 |
| Decision Tree       |   0.7640 |    0.7600 | 0.5588 | 0.6441 |  0.8374 |
| Random Forest       |   0.8090 |    0.7656 | 0.7206 | 0.7424 |  0.8196 |

Random Forest gives the highest baseline F1 score and recall among the three classifiers, while Logistic Regression gives the highest ROC-AUC.

Confusion matrices and ROC curves are generated for all three classifiers.

## Class Imbalance

The project compares three Logistic Regression variants:

| Variant                   | Precision | Recall |     F1 |
| ------------------------- | --------: | -----: | -----: |
| Baseline                  |    0.7833 | 0.6912 | 0.7344 |
| `class_weight="balanced"` |    0.7183 | 0.7500 | 0.7338 |
| SMOTE                     |    0.7353 | 0.7353 | 0.7353 |

SMOTE is applied **only to the training data**, and the test set remains unchanged for evaluation.

SMOTE produced the highest F1 score in this comparison, while `class_weight="balanced"` produced the highest recall. This shows the trade-off between improving minority-class recall and maintaining precision.

---

# 4. Random Forest Tuning

`GridSearchCV` is used to search over:

```text
n_estimators
max_depth
max_features
```

The parameter grid is:

```text
n_estimators = [100, 200]
max_depth    = [None, 5, 10]
max_features = ['sqrt', 'log2']
```

The best parameter combination found by GridSearchCV is:

```text
n_estimators = 200
max_depth    = 5
max_features = sqrt
```

The best cross-validation F1 score is:

```text
Best CV F1 = 0.7408
```

A final Random Forest is fitted with:

```python
RandomForestClassifier(
    oob_score=True,
    ...
)
```

The resulting OOB score is:

```text
OOB score = 0.8214
```

The tuned Random Forest achieves the following held-out test metrics:

```text
Accuracy  = 0.8315
Precision = 0.8654
Recall    = 0.6618
F1        = 0.7500
ROC-AUC   = 0.8389
```

The corresponding confusion-matrix counts are:

```text
TN = 103
FP = 7
FN = 23
TP = 45
```

---

# 5. Fare Regression

A separate regression task predicts `fare` as a continuous target using multivariate Linear Regression.

The regression model is evaluated using:

```text
MAE         = 21.0986
RMSE        = 41.7021
R²          = 0.3482
Adjusted R² = 0.3091
```

A residual plot is generated and the residual spread is examined across the predicted-fare range.

The residual spread is not constant across prediction values, so the residual plot suggests **possible heteroscedasticity**.

Classification and regression metrics are reported separately because they evaluate different types of prediction problems.

## Final Comparison Table

| Model               | Accuracy | Precision | Recall |     F1 | ROC-AUC |     MAE |    RMSE |     R² | Adjusted R² |
| ------------------- | -------: | --------: | -----: | -----: | ------: | ------: | ------: | -----: | ----------: |
| Logistic Regression |   0.8090 |    0.7833 | 0.6912 | 0.7344 |  0.8610 |       — |       — |      — |           — |
| Decision Tree       |   0.7640 |    0.7600 | 0.5588 | 0.6441 |  0.8374 |       — |       — |      — |           — |
| Random Forest       |   0.8090 |    0.7656 | 0.7206 | 0.7424 |  0.8196 |       — |       — |      — |           — |
| Linear Regression   |        — |         — |      — |      — |       — | 21.0986 | 41.7021 | 0.3482 |      0.3091 |

Classification and regression metrics are intentionally presented as separate metric groups because they evaluate different prediction problems and are not directly comparable as raw numbers.

## Final Model Recommendation

I would deploy **Random Forest** for the classification task because it provides the strongest baseline F1 score and recall among the three classifiers. Its baseline F1 score is **0.7424** and recall is **0.7206**, while Logistic Regression has the highest ROC-AUC at **0.8610**. After hyperparameter tuning, Random Forest achieves an improved test accuracy of **0.8315** and F1 score of **0.7500**, with precision of **0.8654** and ROC-AUC of **0.8389**. Therefore, Random Forest provides the best overall practical balance for the classification objective used in this project.

---

# 6. Saved Model

The complete preprocessing and Random Forest prediction pipeline is saved as:

```text
analytics/titanic_rf_pipeline.joblib
```

The saved artifact contains both preprocessing and the classifier.

The complete fitted pipeline is stored as a single `joblib` object so that raw new input can be passed directly to the saved artifact without manually repeating preprocessing steps.

The pipeline is reloaded with `joblib.load(...)` and tested with a raw passenger record to confirm that it can make predictions on raw input.

---

# 7. Support Assistant

The support assistant is a local retrieval-based customer-support system built around a small policy knowledge base.

## Knowledge Base

Exactly eight documents are stored under:

```text
support_assistant/docs/
```

Topics:

* delivery
* returns
* refunds
* membership
* order tracking
* cancellation
* gift cards
* support hours

Each file focuses on one support topic so that semantic retrieval can identify relevant policy information.

## Embedding Model

The assistant uses:

```text
all-MiniLM-L6-v2
```

Each document is represented using a **384-dimensional embedding**.

The embedding model runs locally using `sentence-transformers` and does not require an external embedding API.

## ChromaDB

Embeddings are persisted in:

```text
support_assistant/chroma_db/
```

Collection name:

```text
zepto_support
```

All eight support documents are indexed.

A query such as:

```text
How long does a refund take?
```

returns the refund policy as the highest-ranked result.

## Prompt Structure

The RAG prompt follows the required:

```text
ROLE
CONTEXT
TASK
FORMAT
LENGTH
```

structure.

It also includes:

* an explicit negative constraint against unsupported claims
* a few-shot example
* retrieved context
* the user's question

The assistant is instructed to remain grounded in the retrieved support documents.

---

# 8. LangGraph Workflow

The support assistant is implemented using a LangGraph `StateGraph`.

The three required nodes are:

```text
classify_intent
retrieve_and_answer
direct_answer
```

The workflow is:

```text
START
  |
  v
classify_intent
  |
  +---- policy_question ----> retrieve_and_answer ----+
  |                                                  |
  +---- general_question --> direct_answer ----------+
                                                     |
                                                    END
```

The mock classifier uses the required policy keywords:

* delivery
* return
* refund
* membership
* tracking
* cancel
* gift card
* support hours

In the default mock mode, if the lowercased query contains any of these keywords, it is labeled:

```text
policy_question
```

and routed to:

```text
retrieve_and_answer
```

Otherwise it is labeled:

```text
general_question
```

and routed to:

```text
direct_answer
```

The routing logic is deterministic in the required mock baseline.

---

# 9. Mock LLM Mode

The default configuration is:

```text
MOCK_LLM=1
```

This provides a deterministic local baseline and does not require an external LLM provider.

## Policy Questions

For a `policy_question`:

1. The query is embedded locally.
2. ChromaDB retrieves the top three similar chunks.
3. The top retrieved chunk is used for the deterministic mock answer.
4. The answer begins with:

```text
Based on the retrieved context:
```

5. Retrieved document identifiers are returned in `sources`.

## General Questions

For a `general_question`, the mock branch returns a fixed safe response:

```text
I can only answer questions about Zepto policies right now.
```

The general-question route does not use policy retrieval.

## MOCK_LLM Branching

In the graded baseline, the mock configuration avoids external LLM calls.

The embedding and ChromaDB retrieval stages remain local. The optional real-LLM path changes the classification/generation behavior when `MOCK_LLM=0`.

---

# 10. Structured Output

Pydantic validation is used for the assistant response.

The response contains:

```text
answer
sources
confidence
```

The confidence value is constrained to:

```text
0.0 <= confidence <= 1.0
```

In mock mode, the response is populated deterministically in Python.

For policy questions, `sources` contains the retrieved document/chunk identifiers.

For general questions, `sources` is an empty list.

---

# 11. FastAPI

The assistant exposes:

```text
POST /ask
```

## Example 1 — Policy Retrieval

Request:

```json
{
  "query": "How long does a refund take?"
}
```

Example response:

```json
{
  "answer": "Based on the retrieved context: ...",
  "sources": [
    "doc_006",
    "doc_007",
    "doc_001"
  ],
  "confidence": 0.6478
}
```

This query contains the `refund` keyword, so it is classified as `policy_question` and routed to `retrieve_and_answer`.

## Example 2 — General Question

Request:

```json
{
  "query": "Tell me something unrelated to Zepto policies."
}
```

Example response:

```json
{
  "answer": "I can only answer questions about Zepto policies right now.",
  "sources": [],
  "confidence": 1.0
}
```

This query does not contain any of the required policy keywords, so it is classified as `general_question` and routed to `direct_answer`.

## Support Hours Routing Check

Request:

```json
{
  "query": "What are the customer support hours?"
}
```

Because the required mock keyword heuristic includes `support hours`, this query is classified as `policy_question` and routed to `retrieve_and_answer`.

The relevant support-hours policy document is `doc_008`.

## Start the API

```bash
uvicorn support_assistant.api:app --reload
```

Swagger documentation:

```text
http://127.0.0.1:8000/docs
```

The `/ask` endpoint has been tested successfully with an HTTP 200 response.

---

# 12. RAG Architecture — Ingestion → Embedding → Retrieval → Generation

The support assistant follows four explicit RAG stages.

### 1. Ingestion

`support_assistant/ingest.py` reads the eight files in `support_assistant/docs/`.

Because each policy document is short and focused on one topic, each document is kept as a simple chunk.

The chunks and metadata are stored in the persistent ChromaDB collection:

```text
zepto_support
```

### 2. Embedding

`support_assistant/ingest.py` uses the local `all-MiniLM-L6-v2` model to convert each chunk into a 384-dimensional vector.

No external embedding API key is required.

### 3. Retrieval

`support_assistant/rag.py` embeds the incoming query and retrieves the top three most similar chunks from the `zepto_support` collection.

The LangGraph node `retrieve_and_answer` performs this retrieval.

### 4. Generation

For a `policy_question`, `retrieve_and_answer` uses the retrieved context and the structured prompt to produce the answer.

In the graded `MOCK_LLM=1` mode, the answer is deterministic and grounded in the top retrieved chunk.

The overall flow is:

```text
Policy documents
      |
      v
Ingestion / chunking
      |
      v
all-MiniLM-L6-v2 embeddings
      |
      v
ChromaDB: zepto_support
      |
      v
User query
      |
      v
classify_intent
      |
      +-------- general_question --------> direct_answer
      |
      +-------- policy_question --------> retrieve_and_answer
                                              |
                                              v
                                        Top-3 retrieval
                                              |
                                              v
                                           Answer
                                              |
                                              v
                                      Pydantic response
```

The embedding and retrieval stages remain local in both modes. The optional real-LLM path changes the classification/generation behavior when `MOCK_LLM=0`.

---

# 13. Docker

A Dockerfile is provided at the project root.

Build:

```bash
docker build -t zepto-data-ai .
```

Run:

```bash
docker run --rm -p 8000:8000 zepto-data-ai
```

Swagger:

```text
http://127.0.0.1:8000/docs
```

The Docker configuration keeps deterministic mock mode enabled by default.

The container serves the FastAPI `/ask` endpoint locally without requiring a paid service.

---

# 14. Useful Commands

### Create support documents

```bash
python support_assistant\create_docs.py
```

### Build the vector store

```bash
python support_assistant\ingest.py
```

### Test retrieval

```bash
python support_assistant\rag.py
```

### Test LangGraph

```bash
python -m support_assistant.graph
```

### Start FastAPI

```bash
uvicorn support_assistant.api:app --reload
```

### Run the data pipeline

```bash
python data_pipeline\scrape_and_load.py
python data_pipeline\queries.py
```

### Run analytics

```bash
python analytics\eda.py
python analytics\visual_analysis.py
python analytics\standardization.py
python analytics\modeling.py
python analytics\regression.py
python analytics\comparison.py
python analytics\persistence.py
```

---

# 15. Reproducibility Notes

The project is designed to run locally using the dependencies in the consolidated `requirements.txt`.

Important fixed settings include:

```text
GBP → INR conversion = 105.50
classification test size = 20%
random_state = 42
embedding model = all-MiniLM-L6-v2
mock assistant mode = enabled by default
```

No paid LLM service is required for the graded support-assistant workflow.

The analytics module includes `analytics/titanic.csv` as the committed offline fallback for grading.

The modeling workflow continues from the same cleaned Titanic dataset and applies predictive preprocessing after the train/test split.

---

# 16. Git Workflow

Development changes were made using a feature branch and incorporated into `main`.

The repository history contains multiple commits documenting the development and integration process, including the required feature-branch workflow and merge back into `main`.

---

# 17. Project Summary

The completed platform connects the major components as follows:

```text
Web Scraping
     |
     v
Cleaning + SQLite
     |
     v
SQL Analysis
     |
     v
EDA + Visualization
     |
     v
Machine Learning
     |
     v
Model Persistence
     |
     v
Support Documents
     |
     v
Embeddings + ChromaDB
     |
     v
LangGraph Routing
     |
     v
Pydantic Validation
     |
     v
FastAPI
     |
     v
Docker
```

The project brings these components together as one reproducible local platform rather than treating data collection, analytics, machine learning, and support automation as separate exercises.

---

# Final Submission Checklist

* All three modules are included in this single repository.
* `/data_pipeline`, `/analytics`, and `/support_assistant` are present at the repository root.
* One consolidated `requirements.txt` is provided.
* Root README documents setup, execution, and design decisions.
* Module 1 contains scraping, cleaning, fixed-rate conversion, normalized SQLite storage, SQL queries, outputs, `pd.read_sql`, and `pd.merge`.
* Module 2 contains the committed `titanic.csv` fallback, missing-value analysis, EDA, required correlation analysis, interpreted charts, stratified modeling, training-only preprocessing, three classifiers, imbalance comparison, Random Forest tuning with GridSearchCV and OOB evaluation, regression, comparison metrics, and the complete saved pipeline.
* Module 3 contains all eight policy documents, local embeddings, ChromaDB, the structured prompt, LangGraph routing, mock-mode behavior, Pydantic validation, FastAPI, Docker, example requests, and the RAG architecture description.
* Required outputs and reproducibility scripts are committed.
* Git feature-branch workflow is preserved in repository history.
* No paid service is required for the graded baseline.
* The single public GitHub repository is the submission artifact.

