import os
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.ensemble import RandomForestClassifier


# =========================================================
# PATHS
# =========================================================

DATA_DIR = "analytics"

CLEANED_DATA_PATH = os.path.join(
    DATA_DIR,
    "titanic_cleaned.csv"
)

REPORT_PATH = os.path.join(
    DATA_DIR,
    "task12_random_forest_tuning_report.txt"
)

PARAMS_PATH = os.path.join(
    DATA_DIR,
    "task12_best_parameters.txt"
)


# =========================================================
# LOAD DATA
# =========================================================

df = pd.read_csv(
    CLEANED_DATA_PATH
)

print("=" * 70)
print("TASK 12: RANDOM FOREST TUNING")
print("=" * 70)

print(
    f"Rows loaded: {len(df)}"
)


# =========================================================
# FEATURES / TARGET
# =========================================================

target = "survived"

X = df.drop(
    columns=[target]
)

y = df[target]


# =========================================================
# DROP REDUNDANT / DERIVED COLUMNS
# =========================================================

columns_to_drop = [
    "alive",
    "adult_male",
    "alone",
    "class",
    "who",
    "embark_town"
]

X = X.drop(
    columns=columns_to_drop
)


# =========================================================
# STRATIFIED TRAIN / TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("\n" + "-" * 70)
print("TRAIN / TEST SPLIT")
print("-" * 70)

print(
    f"Training rows: {len(X_train)}"
)

print(
    f"Testing rows : {len(X_test)}"
)


# =========================================================
# FEATURE GROUPS
# =========================================================

numeric_features = [
    "pclass",
    "age",
    "sibsp",
    "parch",
    "fare"
]

categorical_features = [
    "sex",
    "embarked"
]


# =========================================================
# PREPROCESSING
# =========================================================

numeric_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        ),
        (
            "scaler",
            StandardScaler()
        )
    ]
)


categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            )
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )
        )
    ]
)


preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            numeric_pipeline,
            numeric_features
        ),
        (
            "categorical",
            categorical_pipeline,
            categorical_features
        )
    ]
)


# =========================================================
# RANDOM FOREST PIPELINE
# =========================================================

rf_pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "model",
            RandomForestClassifier(
                random_state=42,
                n_jobs=-1
            )
        )
    ]
)


# =========================================================
# GRID SEARCH PARAMETERS
# =========================================================

param_grid = {
    "model__n_estimators": [
        100,
        200,
        300
    ],

    "model__max_depth": [
        None,
        5,
        10,
        15
    ],

    "model__max_features": [
        "sqrt",
        "log2",
        None
    ]
}


print("\n" + "-" * 70)
print("GRID SEARCH")
print("-" * 70)

print(
    "Searching over:"
)

print(param_grid)


# =========================================================
# GRID SEARCH CV
# =========================================================

grid_search = GridSearchCV(
    estimator=rf_pipeline,
    param_grid=param_grid,
    cv=5,
    scoring="f1",
    n_jobs=-1,
    return_train_score=True
)


grid_search.fit(
    X_train,
    y_train
)


# =========================================================
# BEST PARAMETERS
# =========================================================

best_params = grid_search.best_params_

best_cv_score = grid_search.best_score_

best_pipeline = grid_search.best_estimator_


print("\n" + "-" * 70)
print("BEST PARAMETERS")
print("-" * 70)

for parameter, value in best_params.items():
    print(
        f"{parameter}: {value}"
    )


print(
    f"\nBest CV F1 Score: "
    f"{best_cv_score:.4f}"
)


# =========================================================
# TEST PERFORMANCE OF TUNED MODEL
# =========================================================

tuned_predictions = best_pipeline.predict(
    X_test
)


from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


tuned_accuracy = accuracy_score(
    y_test,
    tuned_predictions
)

tuned_precision = precision_score(
    y_test,
    tuned_predictions,
    zero_division=0
)

tuned_recall = recall_score(
    y_test,
    tuned_predictions,
    zero_division=0
)

tuned_f1 = f1_score(
    y_test,
    tuned_predictions,
    zero_division=0
)


print("\n" + "-" * 70)
print("TUNED RANDOM FOREST TEST RESULTS")
print("-" * 70)

print(
    f"Accuracy  : {tuned_accuracy:.4f}"
)

print(
    f"Precision : {tuned_precision:.4f}"
)

print(
    f"Recall    : {tuned_recall:.4f}"
)

print(
    f"F1 Score  : {tuned_f1:.4f}"
)


# =========================================================
# FINAL RANDOM FOREST WITH OOB SCORE
# =========================================================

final_preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            numeric_pipeline,
            numeric_features
        ),
        (
            "categorical",
            categorical_pipeline,
            categorical_features
        )
    ]
)


final_rf = Pipeline(
    steps=[
        (
            "preprocessor",
            final_preprocessor
        ),
        (
            "model",
            RandomForestClassifier(
                n_estimators=best_params[
                    "model__n_estimators"
                ],
                max_depth=best_params[
                    "model__max_depth"
                ],
                max_features=best_params[
                    "model__max_features"
                ],
                random_state=42,
                oob_score=True,
                bootstrap=True,
                n_jobs=-1
            )
        )
    ]
)


final_rf.fit(
    X_train,
    y_train
)


# =========================================================
# OOB SCORE
# =========================================================

oob_score = (
    final_rf
    .named_steps["model"]
    .oob_score_
)


print("\n" + "-" * 70)
print("OUT-OF-BAG SCORE")
print("-" * 70)

print(
    f"OOB Score: {oob_score:.4f}"
)


# =========================================================
# SAVE BEST PARAMETERS
# =========================================================

with open(
    PARAMS_PATH,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "TASK 12 - BEST RANDOM FOREST PARAMETERS\n"
    )

    file.write(
        "=" * 60 + "\n\n"
    )

    for parameter, value in best_params.items():
        file.write(
            f"{parameter}: {value}\n"
        )

    file.write(
        f"\nBest CV F1 Score: "
        f"{best_cv_score:.6f}\n"
    )

    file.write(
        f"Tuned Test Accuracy: "
        f"{tuned_accuracy:.6f}\n"
    )

    file.write(
        f"Tuned Test Precision: "
        f"{tuned_precision:.6f}\n"
    )

    file.write(
        f"Tuned Test Recall: "
        f"{tuned_recall:.6f}\n"
    )

    file.write(
        f"Tuned Test F1: "
        f"{tuned_f1:.6f}\n"
    )

    file.write(
        f"OOB Score: "
        f"{oob_score:.6f}\n"
    )


# =========================================================
# SAVE REPORT
# =========================================================

with open(
    REPORT_PATH,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "TASK 12 - RANDOM FOREST TUNING\n"
    )

    file.write(
        "=" * 60 + "\n\n"
    )

    file.write(
        "GridSearchCV configuration\n"
    )

    file.write(
        "-" * 60 + "\n"
    )

    file.write(
        "Cross-validation folds: 5\n"
        "Scoring metric: F1\n"
        "Random state: 42\n\n"
    )

    file.write(
        "Parameter grid\n"
    )

    file.write(
        "-" * 60 + "\n"
    )

    for parameter, values in param_grid.items():
        file.write(
            f"{parameter}: {values}\n"
        )

    file.write("\n")

    file.write(
        "Best parameters\n"
    )

    file.write(
        "-" * 60 + "\n"
    )

    for parameter, value in best_params.items():
        file.write(
            f"{parameter}: {value}\n"
        )

    file.write(
        f"\nBest CV F1 Score: "
        f"{best_cv_score:.6f}\n\n"
    )

    file.write(
        "Tuned model test metrics\n"
    )

    file.write(
        "-" * 60 + "\n"
    )

    file.write(
        f"Accuracy: {tuned_accuracy:.6f}\n"
        f"Precision: {tuned_precision:.6f}\n"
        f"Recall: {tuned_recall:.6f}\n"
        f"F1 Score: {tuned_f1:.6f}\n\n"
    )

    file.write(
        "Final Random Forest OOB score\n"
    )

    file.write(
        "-" * 60 + "\n"
    )

    file.write(
        f"OOB Score: {oob_score:.6f}\n\n"
    )

    file.write(
        "Methodology\n"
    )

    file.write(
        "-" * 60 + "\n"
    )

    file.write(
        "GridSearchCV was performed only on the training data. "
        "The preprocessing pipeline is part of the estimator, "
        "so transformations are fitted within each training "
        "fold. The final tuned Random Forest uses the best "
        "hyperparameters and reports an out-of-bag score using "
        "bootstrap samples from the training set.\n"
    )


print("\nBest parameters saved to:")
print(PARAMS_PATH)

print("\nTask 12 report saved to:")
print(REPORT_PATH)


print("\n" + "=" * 70)
print("TASK 12 COMPLETE")
print("=" * 70)