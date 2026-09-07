import os
import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


# =========================================================
# PATHS
# =========================================================

DATA_DIR = "analytics"

CLEANED_DATA_PATH = os.path.join(
    DATA_DIR,
    "titanic_cleaned.csv"
)

MODEL_PATH = os.path.join(
    DATA_DIR,
    "titanic_rf_pipeline.joblib"
)

REPORT_PATH = os.path.join(
    DATA_DIR,
    "task15_model_persistence_report.txt"
)


# =========================================================
# LOAD CLEANED DATASET
# =========================================================

df = pd.read_csv(
    CLEANED_DATA_PATH
)

print("=" * 70)
print("TASK 15: MODEL PERSISTENCE")
print("=" * 70)

print(f"Rows loaded: {len(df)}")


# =========================================================
# FEATURES AND TARGET
# =========================================================

target = "survived"

X = df.drop(
    columns=[target]
)

y = df[target]


# Remove redundant / derived columns
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
# SAME STRATIFIED SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
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
# COMPLETE FITTED PIPELINE
# =========================================================

pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "model",
            RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                max_features="sqrt",
                random_state=42,
                n_jobs=-1
            )
        )
    ]
)


# =========================================================
# FIT PIPELINE
# =========================================================

print("\n" + "-" * 70)
print("FITTING COMPLETE PIPELINE")
print("-" * 70)

pipeline.fit(
    X_train,
    y_train
)

print("Pipeline fitted successfully.")


# =========================================================
# TEST BEFORE SAVING
# =========================================================

test_predictions = pipeline.predict(
    X_test
)

test_accuracy = (
    test_predictions == y_test
).mean()

print(
    f"Test accuracy before saving: "
    f"{test_accuracy:.4f}"
)


# =========================================================
# SAVE COMPLETE PIPELINE
# =========================================================

joblib.dump(
    pipeline,
    MODEL_PATH
)

print("\nModel pipeline saved to:")
print(MODEL_PATH)


# =========================================================
# RELOAD PIPELINE
# =========================================================

loaded_pipeline = joblib.load(
    MODEL_PATH
)

print("\nPipeline reloaded successfully.")


# =========================================================
# RAW INPUT PREDICTION
# =========================================================
# This represents a new passenger record.
# No manual preprocessing is performed here.
# The saved pipeline handles preprocessing internally.

raw_passenger = pd.DataFrame(
    [
        {
            "pclass": 1,
            "sex": "female",
            "age": 30,
            "sibsp": 0,
            "parch": 0,
            "fare": 80.0,
            "embarked": "S"
        }
    ]
)


prediction = loaded_pipeline.predict(
    raw_passenger
)

prediction_probability = (
    loaded_pipeline
    .predict_proba(raw_passenger)
    [0, 1]
)


prediction_label = (
    "Survived"
    if prediction[0] == 1
    else "Did Not Survive"
)


print("\n" + "-" * 70)
print("RAW INPUT PREDICTION")
print("-" * 70)

print("Input:")
print(
    raw_passenger.to_string(
        index=False
    )
)

print(
    f"\nPrediction: {prediction_label}"
)

print(
    f"Survival probability: "
    f"{prediction_probability:.4f}"
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
        "TASK 15 - MODEL PERSISTENCE\n"
    )

    file.write(
        "=" * 60 + "\n\n"
    )

    file.write(
        "Saved artifact\n"
    )

    file.write(
        "-" * 60 + "\n"
    )

    file.write(
        f"{MODEL_PATH}\n\n"
    )

    file.write(
        "Pipeline components\n"
    )

    file.write(
        "-" * 60 + "\n"
    )

    file.write(
        "1. Numeric preprocessing: median imputation + StandardScaler\n"
        "2. Categorical preprocessing: most-frequent imputation + OneHotEncoder\n"
        "3. Random Forest classifier\n\n"
    )

    file.write(
        "Test accuracy before persistence\n"
    )

    file.write(
        "-" * 60 + "\n"
    )

    file.write(
        f"{test_accuracy:.6f}\n\n"
    )

    file.write(
        "Reload validation\n"
    )

    file.write(
        "-" * 60 + "\n"
    )

    file.write(
        "The complete fitted pipeline was saved with joblib and "
        "successfully reloaded. A raw passenger record was passed "
        "directly to the reloaded pipeline, demonstrating that "
        "preprocessing and prediction are packaged together.\n\n"
    )

    file.write(
        "Raw input prediction\n"
    )

    file.write(
        "-" * 60 + "\n"
    )

    file.write(
        raw_passenger.to_string(
            index=False
        )
    )

    file.write("\n\n")

    file.write(
        f"Prediction: {prediction_label}\n"
    )

    file.write(
        f"Survival probability: "
        f"{prediction_probability:.6f}\n"
    )


print("\nTask 15 report saved to:")
print(REPORT_PATH)


print("\n" + "=" * 70)
print("TASK 15 COMPLETE")
print("=" * 70)