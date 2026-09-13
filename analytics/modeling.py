import os
import warnings

import matplotlib.pyplot as plt
import pandas as pd
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    RocCurveDisplay,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, plot_tree

warnings.filterwarnings("ignore")

DATA_DIR = "analytics"
CLEANED_DATA_PATH = os.path.join(DATA_DIR, "titanic_cleaned.csv")
METRICS_PATH = os.path.join(DATA_DIR, "task10_classification_metrics.csv")
ROC_PATH = os.path.join(DATA_DIR, "task10_roc_curves.png")
IMBALANCE_METRICS_PATH = os.path.join(DATA_DIR, "task11_imbalance_metrics.csv")
IMBALANCE_REPORT_PATH = os.path.join(DATA_DIR, "task11_imbalance_report.txt")
DT_PATH = os.path.join(DATA_DIR, "task9_decision_tree.png")
TUNING_PATH = os.path.join(DATA_DIR, "task12_random_forest_tuning_report.txt")
PARAMS_PATH = os.path.join(DATA_DIR, "task12_best_parameters.txt")


def make_preprocessor():
    numeric_features = ["pclass", "age", "sibsp", "parch", "fare"]
    categorical_features = ["sex", "embarked"]
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features),
    ])


def prepare_xy(df):
    target = "survived"
    columns_to_drop = ["alive", "adult_male", "alone", "class", "who", "embark_town"]
    X = df.drop(columns=[target] + columns_to_drop)
    y = df[target].astype(int)
    return X, y


def classifier_metrics(name, model, X_test, y_test):
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    cm = confusion_matrix(y_test, pred)
    return {
        "model": name,
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred, zero_division=0),
        "f1_score": f1_score(y_test, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, proba),
        "tn": int(cm[0, 0]), "fp": int(cm[0, 1]), "fn": int(cm[1, 0]), "tp": int(cm[1, 1]),
    }


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    df = pd.read_csv(CLEANED_DATA_PATH)
    X, y = prepare_xy(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # Task 8/9/10: fit preprocessing only on training data inside each pipeline.
    models = {
        "Logistic Regression": Pipeline([
            ("preprocessor", make_preprocessor()),
            ("model", LogisticRegression(max_iter=1000, random_state=42)),
        ]),
        "Decision Tree": Pipeline([
            ("preprocessor", make_preprocessor()),
            ("model", DecisionTreeClassifier(max_depth=5, random_state=42)),
        ]),
        "Random Forest": Pipeline([
            ("preprocessor", make_preprocessor()),
            ("model", RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)),
        ]),
    }

    metrics = []
    fitted_models = {}
    for name, pipeline in models.items():
        pipeline.fit(X_train, y_train)
        fitted_models[name] = pipeline
        row = classifier_metrics(name, pipeline, X_test, y_test)
        metrics.append(row)

        ConfusionMatrixDisplay.from_predictions(y_test, pipeline.predict(X_test))
        plt.title(f"Confusion Matrix - {name}")
        plt.tight_layout()
        safe_name = name.lower().replace(" ", "_")
        plt.savefig(os.path.join(DATA_DIR, f"task10_confusion_matrix_{safe_name}.png"), dpi=150)
        plt.close()

    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(METRICS_PATH, index=False)

    with open(os.path.join(DATA_DIR, "task9_10_classification_report.txt"), "w", encoding="utf-8") as f:
        f.write("TASK 9/10 - CLASSIFICATION MODELS AND METRICS\n\n")
        f.write(metrics_df.to_string(index=False))
        f.write("\n\nEach classifier has a saved confusion matrix and is evaluated with accuracy, precision, recall, F1 and ROC-AUC.\n")

    # ROC curves for all three classifiers.
    fig, ax = plt.subplots(figsize=(9, 6))
    for name, model in fitted_models.items():
        RocCurveDisplay.from_estimator(model, X_test, y_test, name=name, ax=ax)
    ax.set_title("ROC Curves - Three Classifiers")
    fig.tight_layout()
    fig.savefig(ROC_PATH, dpi=150)
    plt.close(fig)

    # Decision tree plot with feature/class labels.
    tree_pipeline = fitted_models["Decision Tree"]
    tree_model = tree_pipeline.named_steps["model"]
    preprocessor = tree_pipeline.named_steps["preprocessor"]
    feature_names = preprocessor.get_feature_names_out()
    plt.figure(figsize=(20, 12))
    plot_tree(tree_model, feature_names=feature_names, class_names=["Did not survive", "Survived"], filled=False, max_depth=3)
    plt.title("Decision Tree Classifier")
    plt.tight_layout()
    plt.savefig(DT_PATH, dpi=150)
    plt.close()

    # Task 11: baseline vs class_weight balanced vs SMOTE. Sampling occurs only after the split.
    imbalance_models = {
        "Baseline Logistic Regression": Pipeline([
            ("preprocessor", make_preprocessor()),
            ("model", LogisticRegression(max_iter=1000, random_state=42)),
        ]),
        "Balanced Logistic Regression": Pipeline([
            ("preprocessor", make_preprocessor()),
            ("model", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)),
        ]),
        "SMOTE Logistic Regression": ImbPipeline([
            ("preprocessor", make_preprocessor()),
            ("smote", SMOTE(random_state=42)),
            ("model", LogisticRegression(max_iter=1000, random_state=42)),
        ]),
    }
    imbalance_rows = []
    for name, model in imbalance_models.items():
        model.fit(X_train, y_train)
        row = classifier_metrics(name, model, X_test, y_test)
        imbalance_rows.append({k: row[k] for k in ["model", "precision", "recall", "f1_score"]})

    imbalance_df = pd.DataFrame(imbalance_rows)
    imbalance_df.to_csv(IMBALANCE_METRICS_PATH, index=False)
    best_row = imbalance_df.loc[imbalance_df["f1_score"].idxmax()]
    with open(IMBALANCE_REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("TASK 11 - CLASS IMBALANCE\n\n")
        f.write(f"Positive-class share (survived=1): {y_train.mean():.4f}\n")
        f.write("\nCompared models:\n")
        f.write(imbalance_df.to_string(index=False))
        f.write("\n\nConclusion: The model with the highest test F1 in this comparison was "
                f"{best_row['model']} ({best_row['f1_score']:.4f}). Precision and recall should be considered together because class imbalance changes the cost of false positives and false negatives.\n")

    # Visual comparison for Task 11.
    ax = imbalance_df.set_index("model")[["precision", "recall", "f1_score"]].plot(kind="bar", figsize=(10, 6))
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.set_title("Class Imbalance Handling Comparison")
    plt.tight_layout()
    plt.savefig(os.path.join(DATA_DIR, "task11_imbalance_comparison.png"), dpi=150)
    plt.close()

    # Task 12: grid search over required RF hyperparameters; estimator uses OOB scoring.
    rf_for_grid = Pipeline([
        ("preprocessor", make_preprocessor()),
        ("model", RandomForestClassifier(random_state=42, n_jobs=-1, oob_score=True, bootstrap=True)),
    ])
    param_grid = {
        "model__n_estimators": [100, 200],
        "model__max_depth": [None, 5, 10],
        "model__max_features": ["sqrt", "log2"],
    }
    grid = GridSearchCV(rf_for_grid, param_grid=param_grid, cv=5, scoring="f1", n_jobs=-1, return_train_score=True)
    grid.fit(X_train, y_train)
    best_pipeline = grid.best_estimator_
    best_rf = best_pipeline.named_steps["model"]
    tuned_row = classifier_metrics("Tuned Random Forest", best_pipeline, X_test, y_test)

    with open(PARAMS_PATH, "w", encoding="utf-8") as f:
        f.write(f"Best parameters: {grid.best_params_}\n")
        f.write(f"Best CV F1: {grid.best_score_:.4f}\n")
        f.write(f"OOB score of best estimator: {best_rf.oob_score_:.4f}\n")
        f.write(f"Test F1: {tuned_row['f1_score']:.4f}\n")

    with open(TUNING_PATH, "w", encoding="utf-8") as f:
        f.write("TASK 12 - RANDOM FOREST GRID SEARCH AND OOB\n\n")
        f.write(f"Parameter grid: {param_grid}\n")
        f.write(f"Best parameters: {grid.best_params_}\n")
        f.write(f"Best CV F1: {grid.best_score_:.4f}\n")
        f.write(f"OOB score: {best_rf.oob_score_:.4f}\n")
        f.write(f"Tuned test metrics: {tuned_row}\n")

    print("Task 9/10/11/12 modeling completed successfully.")
    print(metrics_df[["model", "accuracy", "precision", "recall", "f1_score", "roc_auc"]].to_string(index=False))


if __name__ == "__main__":
    main()
