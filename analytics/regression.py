import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


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
    "task13_regression_report.txt"
)

METRICS_PATH = os.path.join(
    DATA_DIR,
    "task13_regression_metrics.csv"
)

RESIDUAL_PLOT_PATH = os.path.join(
    DATA_DIR,
    "task13_residual_plot.png"
)


# =========================================================
# LOAD DATA
# =========================================================

df = pd.read_csv(
    CLEANED_DATA_PATH
)

print("=" * 70)
print("TASK 13: REGRESSION - FARE PREDICTION")
print("=" * 70)

print(
    f"Rows loaded: {len(df)}"
)


# =========================================================
# TARGET = FARE
# =========================================================

target = "fare"

y = df[target].copy()

X = df.drop(
    columns=[target]
)


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
# TRAIN / TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
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
    "survived",
    "pclass",
    "age",
    "sibsp",
    "parch"
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
# LINEAR REGRESSION PIPELINE
# =========================================================

regression_pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "model",
            LinearRegression()
        )
    ]
)


# =========================================================
# TRAIN
# =========================================================

print("\n" + "-" * 70)
print("TRAINING LINEAR REGRESSION")
print("-" * 70)

regression_pipeline.fit(
    X_train,
    y_train
)


# =========================================================
# PREDICTIONS
# =========================================================

y_pred = regression_pipeline.predict(
    X_test
)


# =========================================================
# METRICS
# =========================================================

mae = mean_absolute_error(
    y_test,
    y_pred
)

mse = mean_squared_error(
    y_test,
    y_pred
)

rmse = np.sqrt(
    mse
)

r2 = r2_score(
    y_test,
    y_pred
)


# Number of observations
n = len(y_test)

# Number of model predictors after preprocessing
processed_feature_count = (
    regression_pipeline
    .named_steps["preprocessor"]
    .get_feature_names_out()
    .shape[0]
)

p = processed_feature_count


# Adjusted R²
if n > p + 1:
    adjusted_r2 = (
        1
        - (
            (1 - r2)
            * (n - 1)
            / (n - p - 1)
        )
    )
else:
    adjusted_r2 = np.nan


# =========================================================
# PRINT METRICS
# =========================================================

print("\n" + "=" * 70)
print("REGRESSION METRICS")
print("=" * 70)

print(
    f"MAE           : {mae:.4f}"
)

print(
    f"RMSE          : {rmse:.4f}"
)

print(
    f"R²            : {r2:.4f}"
)

print(
    f"Adjusted R²   : {adjusted_r2:.4f}"
)

print(
    f"Predictors (p): {p}"
)


# =========================================================
# RESIDUALS
# =========================================================

residuals = y_test - y_pred


# =========================================================
# RESIDUAL PLOT
# =========================================================

plt.figure(
    figsize=(9, 6)
)

plt.scatter(
    y_pred,
    residuals,
    alpha=0.7
)

plt.axhline(
    y=0,
    linestyle="--"
)

plt.xlabel(
    "Predicted Fare"
)

plt.ylabel(
    "Residual (Actual - Predicted)"
)

plt.title(
    "Residual Plot for Fare Prediction"
)

plt.tight_layout()

plt.savefig(
    RESIDUAL_PLOT_PATH,
    dpi=150
)

plt.close()


print("\nResidual plot saved to:")
print(RESIDUAL_PLOT_PATH)


# =========================================================
# SIMPLE HETEROSCEDASTICITY CHECK
# =========================================================

# Divide predictions into four equally sized groups
prediction_bins = pd.qcut(
    y_pred,
    q=4,
    duplicates="drop"
)

residual_variance = (
    pd.DataFrame(
        {
            "predicted_fare": y_pred,
            "residual": residuals
        }
    )
    .assign(
        prediction_group=prediction_bins
    )
    .groupby(
        "prediction_group",
        observed=True
    )["residual"]
    .var()
)


variance_ratio = (
    residual_variance.max()
    / residual_variance.min()
)


if variance_ratio > 4:
    heteroscedasticity_conclusion = (
        "Residual variance changes substantially across "
        "prediction ranges, suggesting possible heteroscedasticity."
    )
else:
    heteroscedasticity_conclusion = (
        "Residual variance is relatively similar across "
        "prediction ranges, so there is no strong evidence "
        "of heteroscedasticity from this check."
    )


print("\n" + "-" * 70)
print("HETEROSCEDASTICITY CHECK")
print("-" * 70)

print(
    residual_variance
)

print(
    f"\nMaximum/minimum residual variance ratio: "
    f"{variance_ratio:.4f}"
)

print(
    heteroscedasticity_conclusion
)


# =========================================================
# SAVE METRICS
# =========================================================

metrics_df = pd.DataFrame(
    [
        {
            "model": "Linear Regression",
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
            "adjusted_r2": adjusted_r2
        }
    ]
)

metrics_df.to_csv(
    METRICS_PATH,
    index=False
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
        "TASK 13 - REGRESSION: FARE PREDICTION\n"
    )

    file.write(
        "=" * 60 + "\n\n"
    )

    file.write(
        "Target variable: fare\n"
    )

    file.write(
        f"Training rows: {len(X_train)}\n"
    )

    file.write(
        f"Testing rows: {len(X_test)}\n\n"
    )

    file.write(
        "Evaluation metrics\n"
    )

    file.write(
        "-" * 60 + "\n"
    )

    file.write(
        f"MAE: {mae:.6f}\n"
        f"RMSE: {rmse:.6f}\n"
        f"R2: {r2:.6f}\n"
        f"Adjusted R2: {adjusted_r2:.6f}\n"
    )

    file.write(
        f"Number of predictors after preprocessing: {p}\n\n"
    )

    file.write(
        "Residual variance by prediction group\n"
    )

    file.write(
        "-" * 60 + "\n"
    )

    file.write(
        residual_variance.to_string()
    )

    file.write("\n\n")

    file.write(
        "Heteroscedasticity assessment\n"
    )

    file.write(
        "-" * 60 + "\n"
    )

    file.write(
        f"Maximum/minimum variance ratio: "
        f"{variance_ratio:.6f}\n\n"
    )

    file.write(
        heteroscedasticity_conclusion
    )

    file.write("\n\n")

    file.write(
        "Methodology\n"
    )

    file.write(
        "-" * 60 + "\n"
    )

    file.write(
        "Fare was modeled as a continuous target using Linear "
        "Regression. The train/test split was performed before "
        "preprocessing. Numeric variables use median imputation "
        "and standardization, while categorical variables use "
        "most-frequent imputation and one-hot encoding. The "
        "preprocessing pipeline was fitted only on the training "
        "data.\n"
    )


print("\nMetrics saved to:")
print(METRICS_PATH)

print("\nRegression report saved to:")
print(REPORT_PATH)


print("\n" + "=" * 70)
print("TASK 13 COMPLETE")
print("=" * 70)