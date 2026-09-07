# Zepto Data & AI Platform

This project brings together three practical parts of an AI and data platform:

1. A web data pipeline that collects and stores book information.
2. An analytics and machine learning workflow built using the Titanic dataset.
3. A local customer-support assistant that uses semantic retrieval and a LangGraph workflow.

The project is designed to run locally without requiring a paid AI API.

---

## Project Layout

```text
zepto-data-ai-platform/
│
├── data_pipeline/
│
├── analytics/
│
├── support_assistant/
│
├── Dockerfile
├── requirements.txt
└── README.md
```

---

# 1. Data Pipeline

The data pipeline collects book information from the Books to Scrape website using Python.

The scraper uses `requests` to retrieve pages and `BeautifulSoup` to read the HTML content.

### Categories collected

The implementation covers these three categories:

* Travel
* Mystery
* Historical Fiction

The final cleaned dataset contains **69 records** belonging to **3 categories**.

### Fields captured

Each book record contains:

* title
* price in GBP
* star rating
* availability
* category

During cleaning, the source values are converted into useful data types:

* price → floating-point number
* rating → integer from 1 to 5
* availability → Boolean `in_stock`

Invalid required values are handled without stopping the complete pipeline.

### Currency conversion

The project uses a fixed conversion rate:

```text
1 GBP = 105.50 INR
```

The INR value is calculated locally using:

```text
price_inr = price_gbp * 105.50
```

No currency API is needed.

### SQLite design

The cleaned data is stored in:

```text
data_pipeline/zepto_books.db
```

The database contains two related tables:

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

Using a separate category table avoids repeatedly storing the same category name for every book.

### SQL work

The SQL section demonstrates:

* filtering with `WHERE`
* sorting with `ORDER BY`
* limiting results with `LIMIT`
* unique values using `DISTINCT`
* filtering with `IN`
* range filtering using `BETWEEN`
* combining tables using `JOIN`

Query outputs are generated and saved by the project scripts.

The project also checks that SQL results can be loaded through pandas and that the category-book relationship can be reproduced in memory using `pd.merge()`.

### Running the pipeline

```bash
python data_pipeline\scrape_and_load.py
python data_pipeline\queries.py
```

---

# 2. Analytics and Machine Learning

The analytics module uses the Titanic dataset for exploratory analysis and predictive modeling.

The source dataset is loaded only once and cached locally as:

```text
analytics/titanic.csv
```

The cleaned version is stored as:

```text
analytics/titanic_cleaned.csv
```

The cleaned dataset contains **889 rows**.

---

## Data cleaning

The missing-value decisions are based on the measured percentages from the original dataset.

| Column        | Missing | Decision          |
| ------------- | ------: | ----------------- |
| `deck`        |  77.22% | Removed           |
| `age`         |  19.87% | Median imputation |
| `embarked`    |   0.22% | Rows removed      |
| `embark_town` |   0.22% | Rows removed      |

After cleaning, no missing values remain.

---

## Exploratory analysis

The project examines both individual variables and relationships between variables.

### Age and Fare

Histograms and boxplots are generated for `age` and `fare`.

Using the IQR rule:

* Age outliers: **65**
* Fare outliers: **114**

For Fare:

```text
Mean     = 32.0967
Median   = 14.4542
Mode     = 8.0500
Skewness = 4.8014
```

The large difference between the Fare mean and median, together with the positive skewness value, indicates a strongly right-skewed distribution.

---

## Survival analysis

The analysis shows a large difference in survival rates across passenger groups.

### Survival by sex

```text
Female = 74.04%
Male   = 18.89%
```

### Survival by passenger class

```text
1st class = 62.62%
2nd class = 47.28%
3rd class = 24.24%
```

Combining sex and class reveals an even clearer pattern. Female passengers in first and second class had the highest observed survival rates, while male passengers in second and third class had much lower rates.

---

## Correlation analysis

The required correlation matrix uses exactly these six variables:

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

A heatmap is also generated to make these relationships easier to inspect visually.

---

## Visual EDA

The project produces multiple charts, including:

* survival rate by sex
* survival rate by passenger class
* survival rate by sex and passenger class
* fare distribution by survival
* age distribution by survival

Every chart has a corresponding written interpretation.

---

## Standardization

Age and Fare are also standardized using `StandardScaler`.

Before scaling, their means and spreads are very different.

After scaling, both variables are centered around zero with a standard deviation close to one.

This experiment is kept separate from the final model pipeline so that the actual predictive workflow can perform preprocessing only after the train/test split.

---

## Classification

The target for classification is `survived`.

An 80/20 stratified split is used:

```text
Training rows = 711
Testing rows  = 178
```

The survival proportions remain nearly unchanged after splitting.

### Models tested

* Logistic Regression
* Decision Tree
* Random Forest

### Baseline results

| Model               | Accuracy | Precision | Recall |     F1 | ROC-AUC |
| ------------------- | -------: | --------: | -----: | -----: | ------: |
| Logistic Regression |   0.8090 |    0.7833 | 0.6912 | 0.7344 |  0.8610 |
| Decision Tree       |   0.7640 |    0.7600 | 0.5588 | 0.6441 |  0.8374 |
| Random Forest       |   0.8090 |    0.7656 | 0.7206 | 0.7424 |  0.8196 |

Random Forest has the best F1 score among the three baseline classifiers, while Logistic Regression has the highest ROC-AUC.

### Imbalanced-class experiment

The project compares:

* normal Logistic Regression
* Logistic Regression with `class_weight="balanced"`
* Logistic Regression trained after applying SMOTE

SMOTE is applied only to the training side. The test set remains untouched so that evaluation continues to represent unseen data.

---

## Random Forest tuning

`GridSearchCV` is used to explore:

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

The resulting OOB score is recorded in the Task 12 report.

---

# 3. Fare Regression

A separate regression problem is created by treating `fare` as the continuous target.

A Linear Regression model is evaluated with:

```text
MAE          = 21.0986
RMSE         = 41.7021
R²           = 0.3482
Adjusted R²  = 0.3091
```

A residual plot is produced to inspect prediction errors.

Residual variance is also examined across prediction ranges to check whether the error spread changes substantially.

Classification and regression metrics are reported separately because they measure different types of prediction problems.

---

# 4. Saved Model

The complete preprocessing and Random Forest prediction workflow is stored as a single Joblib artifact:

```text
analytics/titanic_rf_pipeline.joblib
```

The saved object contains the preprocessing steps as well as the classifier.

After saving, the pipeline is loaded again and tested with a raw passenger record. This demonstrates that a new record can be passed directly to the saved pipeline without manually repeating preprocessing steps.

---

# 5. Support Assistant

The support assistant is a local retrieval-based customer support system.

It is designed around a small policy knowledge base rather than an external web search.

---

## Knowledge base

Exactly eight support documents are included under:

```text
support_assistant/docs/
```

They cover:

* delivery
* returns
* refunds
* membership
* order tracking
* cancellation
* gift cards
* support hours

The documents are intentionally separated by topic so that retrieval can return relevant policy information.

---

## Embedding model

The assistant uses:

```text
all-MiniLM-L6-v2
```

Each document is converted into a **384-dimensional embedding**.

---

## ChromaDB

The embeddings are stored in a persistent ChromaDB collection:

```text
Collection: zepto_support
Database: support_assistant/chroma_db/
```

All eight documents are indexed.

For a query such as:

```text
How long does a refund take?
```

the refund policy document is returned as the highest-ranked result.

---

## Prompt structure

The RAG prompt is organized into clear sections:

```text
ROLE
CONTEXT
TASK
FORMAT
LENGTH
```

It also contains:

* a negative constraint against unsupported claims
* a few-shot example
* retrieved context
* the user's actual question

The assistant is instructed to stay grounded in the retrieved documents.

---

# 6. LangGraph Workflow

The support assistant uses a `StateGraph`.

The graph contains three main nodes:

```text
classify_intent
retrieve_and_answer
direct_answer
```

The overall flow is:

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

The intent classifier recognizes support topics such as:

* delivery
* return
* refund
* membership
* tracking
* cancellation
* gift card
* support hours

Unknown questions receive a safe fallback response.

---

# 7. Mock LLM Mode

The project defaults to:

```text
MOCK_LLM=1
```

This is intentional.

The mock path provides a deterministic local baseline and does not need an external LLM provider or API key.

For retrieval-based responses, the answer begins with:

```text
Based on the retrieved context:
```

The returned source identifiers and confidence value are included in the structured response.

---

# 8. Structured Output

Pydantic validation is used for the assistant response.

The response contains:

```text
answer
sources
confidence
```

The confidence value is restricted to:

```text
0.0 <= confidence <= 1.0
```

This prevents invalid output from being returned through the API.

---

# 9. FastAPI

The assistant is exposed through:

```text
POST /ask
```

Example request:

```json
{
  "query": "How long does a refund take?"
}
```

Example response structure:

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

The API can be started locally with:

```bash
uvicorn support_assistant.api:app --reload
```

Swagger documentation is available at:

```text
http://127.0.0.1:8000/docs
```

The `/ask` endpoint has been tested successfully with an HTTP 200 response.

---

# 10. Docker

A Dockerfile is included at the project root.

Build the container:

```bash
docker build -t zepto-support-assistant .
```

Run it:

```bash
docker run --rm -p 8000:8000 zepto-support-assistant
```

The API can then be accessed through:

```text
http://127.0.0.1:8000/docs
```

The container also keeps the deterministic mock mode enabled by default.

---

# 11. Useful Commands

### Create the support documents

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

### Test the LangGraph workflow

```bash
python -m support_assistant.graph
```

### Start the API

```bash
uvicorn support_assistant.api:app --reload
```

### Run the analytics workflow

```bash
python analytics\eda.py
python analytics\visual_analysis.py
python analytics\standardization.py
python analytics\modeling.py
python analytics\regression.py
python analytics\comparison.py
python analytics\persistence.py
```

### Run the data pipeline

```bash
python data_pipeline\scrape_and_load.py
python data_pipeline\queries.py
```

---

# 12. Reproducibility Notes

The project is intended to run locally with the dependencies listed in `requirements.txt`.

Important fixed settings include:

```text
GBP → INR conversion = 105.50
classification test size = 20%
random_state = 42
mock assistant mode = enabled by default
embedding model = all-MiniLM-L6-v2
```

The project does not depend on a paid LLM service for the graded support-assistant workflow.

---

# 13. Summary

The final project combines:

```text
Web Scraping
      ↓
Cleaning + SQLite
      ↓
SQL Analysis
      ↓
EDA + Visualization
      ↓
Machine Learning
      ↓
Model Persistence
      ↓
Document Embeddings
      ↓
ChromaDB Retrieval
      ↓
LangGraph Routing
      ↓
Pydantic Validation
      ↓
FastAPI
      ↓
Docker
```

The main goal was to build the individual components as one reproducible local platform rather than treating scraping, analytics, machine learning, and the support assistant as unrelated exercises.

## Project Status

The three capstone modules have been implemented locally: data pipeline, analytics, and support assistant.
