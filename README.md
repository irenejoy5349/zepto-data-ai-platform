# Zepto Data & AI Platform

This project combines three practical components into one local AI and data platform:

1. A web data pipeline that collects and stores book information.
2. An analytics and machine learning workflow built around the Titanic dataset.
3. A local customer-support assistant using semantic retrieval and LangGraph.

The graded workflow is designed to run locally without requiring a paid AI API.

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

* price → floating-point value
* rating → integer from 1 to 5
* availability → Boolean `in_stock`

The pipeline handles parsing failures without stopping the complete run.

## Currency Conversion

A fixed conversion rate is used:

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
    category_name

books
    book_id (Primary Key)
    title
    price_gbp
    price_inr
    rating
    in_stock
    category_id (Foreign Key)
```

Separating categories into their own table avoids storing the same category text repeatedly.

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

SQL results are also loaded through pandas, and the category-book relationship is reproduced using `pd.merge()` for comparison.

## Running the Pipeline

```bash
python data_pipeline\scrape_and_load.py
python data_pipeline\queries.py
```

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

The cleaned dataset contains **889 rows**.

## Data Cleaning

The cleaning decisions are based on the measured missing-value percentages:

| Column        | Missing | Decision           |
| ------------- | ------: | ------------------ |
| `deck`        |  77.22% | Drop column        |
| `age`         |  19.87% | Median imputation  |
| `embarked`    |   0.22% | Drop affected rows |
| `embark_town` |   0.22% | Drop affected rows |

After cleaning, the dataset contains no remaining missing values.

## Exploratory Analysis

Histograms and boxplots are used to examine Age and Fare.

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

The large difference between the Fare mean and median, together with the positive skewness, indicates a strongly right-skewed Fare distribution.

## Survival Analysis

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

Combining sex and passenger class shows an even stronger relationship with survival. Female passengers in first and second class had the highest observed survival rates, while male passengers in second and third class had substantially lower rates.

## Correlation Analysis

The required correlation matrix contains exactly:

```text
survived
pclass
age
sibsp
parch
fare
```

The two strongest absolute correlations are:

```text
pclass ↔ fare = -0.5482
sibsp  ↔ parch = 0.4145
```

A correlation heatmap is also generated.

## Visual EDA

The project produces multiple charts covering:

* survival rate by sex
* survival rate by passenger class
* survival rate by sex and class
* Fare distribution by survival
* Age distribution by survival

Each chart has an accompanying written interpretation.

## Standardization

Age and Fare are standardized using `StandardScaler`.

After standardization, both features are centered around zero with a standard deviation close to one.

This exploratory experiment is kept separate from the final predictive workflow. The actual model pipeline performs preprocessing after the train/test split.

## Classification

The classification target is `survived`.

An 80/20 stratified train-test split is used:

```text
Training rows = 711
Testing rows  = 178
```

The class proportions remain almost unchanged between the full dataset, training set, and test set.

### Models

* Logistic Regression
* Decision Tree
* Random Forest

### Baseline Results

| Model               | Accuracy | Precision | Recall |     F1 | ROC-AUC |
| ------------------- | -------: | --------: | -----: | -----: | ------: |
| Logistic Regression |   0.8090 |    0.7833 | 0.6912 | 0.7344 |  0.8610 |
| Decision Tree       |   0.7640 |    0.7600 | 0.5588 | 0.6441 |  0.8374 |
| Random Forest       |   0.8090 |    0.7656 | 0.7206 | 0.7424 |  0.8196 |

Random Forest gives the highest F1 score among these three baseline models, while Logistic Regression gives the highest ROC-AUC.

### Class Imbalance

The project compares:

* baseline Logistic Regression
* Logistic Regression with `class_weight="balanced"`
* Logistic Regression with SMOTE

SMOTE is applied only to the training data. The test set remains unchanged for evaluation.

## Random Forest Tuning

`GridSearchCV` is used to search over:

```text
n_estimators
max_depth
max_features
```

The tuned model is evaluated on the held-out test set.

A final Random Forest is also fitted with:

```text
oob_score=True
```

The resulting OOB score is saved in the Task 12 report.

---

# 3. Fare Regression

A separate regression task predicts `fare` as a continuous target.

Linear Regression is evaluated using:

```text
MAE         = 21.0986
RMSE        = 41.7021
R²          = 0.3482
Adjusted R² = 0.3091
```

A residual plot is generated and residual variance is examined across prediction ranges.

Classification and regression metrics are reported separately because they evaluate different types of prediction problems.

---

# 4. Saved Model

The complete preprocessing and Random Forest prediction pipeline is saved as:

```text
analytics/titanic_rf_pipeline.joblib
```

The saved artifact contains both preprocessing and the classifier.

The pipeline is reloaded and tested with a raw passenger record, demonstrating that new input can be passed directly to the saved model without manually repeating preprocessing.

---

# 5. Support Assistant

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

The RAG prompt contains:

```text
ROLE
CONTEXT
TASK
FORMAT
LENGTH
```

It also includes:

* a negative constraint against unsupported claims
* a few-shot example
* retrieved context
* the user's question

The assistant is instructed to remain grounded in the retrieved support documents.

---

# 6. LangGraph Workflow

The support assistant is implemented using a LangGraph `StateGraph`.

The three main nodes are:

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
  +---- retrieve ----> retrieve_and_answer ----+
  |                                            |
  +---- direct ------> direct_answer ----------+
                                               |
                                              END
```

Recognized support intents include:

* delivery
* return
* refund
* membership
* tracking
* cancellation
* gift card
* support hours

Unknown queries receive a safe fallback response.

---

# 7. Mock LLM Mode

The default configuration is:

```text
MOCK_LLM=1
```

This provides a deterministic local baseline and does not require an external LLM provider.

Retrieval-based responses begin with:

```text
Based on the retrieved context:
```

The response also includes the retrieved source identifiers and a confidence value.

---

# 8. Structured Output

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

This prevents invalid structured output from being returned by the API.

---

# 9. FastAPI

The assistant exposes:

```text
POST /ask
```

### Example 1

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

### Example 2

Request:

```json
{
  "query": "What are the customer support hours?"
}
```

This request follows the direct-answer route in the local LangGraph workflow.

### Start the API

```bash
uvicorn support_assistant.api:app --reload
```

Swagger documentation:

```text
http://127.0.0.1:8000/docs
```

The `/ask` endpoint has been tested successfully with an HTTP 200 response.

---

# 10. Docker

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

---

# 11. Useful Commands

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

# 12. Reproducibility Notes

The project is designed to run locally using the dependencies in `requirements.txt`.

Important fixed settings include:

```text
GBP → INR conversion = 105.50
classification test size = 20%
random_state = 42
embedding model = all-MiniLM-L6-v2
mock assistant mode = enabled by default
```

No paid LLM service is required for the graded support-assistant workflow.

---

# 13. Git Workflow

Development changes were made using the `feature/capstone-final` branch and incorporated into `main`.

The repository contains multiple commits documenting the development process.

---

# 14. Project Summary

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
