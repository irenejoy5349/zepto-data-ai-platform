import os
import pandas as pd


# =========================================================
# PATHS
# =========================================================

DATA_DIR = "analytics"

CLASSIFICATION_PATH = os.path.join(
    DATA_DIR,
    "task10_classification_metrics.csv"
)

REGRESSION_PATH = os.path.join(
    DATA_DIR,
    "task13_regression_metrics.csv"
)

COMPARISON_PATH = os.path.join(
    DATA_DIR,
    "task14_model_comparison.csv"
)

REPORT_PATH = os.path.join(
    DATA_DIR,
    "task14_comparison_report.md"
)


# =========================================================
# LOAD SAVED RESULTS
# =========================================================

classification_df = pd.read_csv(
    CLASSIFICATION_PATH
)

regression_df = pd.read_csv(
    REGRESSION_PATH
)


print("=" * 70)
print("TASK 14: CLASSIFICATION VS REGRESSION COMPARISON")
print("=" * 70)


# =========================================================
# CLASSIFICATION METRICS
# =========================================================

classification_comparison = classification_df[
    [
        "model",
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "roc_auc"
    ]
].copy()


print("\n" + "-" * 70)
print("CLASSIFICATION MODELS")
print("-" * 70)

print(
    classification_comparison.to_string(
        index=False
    )
)


# =========================================================
# REGRESSION METRICS
# =========================================================

regression_comparison = regression_df[
    [
        "model",
        "mae",
        "rmse",
        "r2",
        "adjusted_r2"
    ]
].copy()


print("\n" + "-" * 70)
print("REGRESSION MODELS")
print("-" * 70)

print(
    regression_comparison.to_string(
        index=False
    )
)


# =========================================================
# BEST CLASSIFICATION MODEL
# Based on F1 score
# =========================================================

best_classification = classification_df.loc[
    classification_df["f1_score"].idxmax()
]


# =========================================================
# BEST REGRESSION MODEL
# Based on R²
# =========================================================

best_regression = regression_df.loc[
    regression_df["r2"].idxmax()
]


print("\n" + "-" * 70)
print("BEST CLASSIFICATION MODEL")
print("-" * 70)

print(
    f"Model     : {best_classification['model']}"
)

print(
    f"Accuracy  : {best_classification['accuracy']:.4f}"
)

print(
    f"Precision : {best_classification['precision']:.4f}"
)

print(
    f"Recall    : {best_classification['recall']:.4f}"
)

print(
    f"F1 Score  : {best_classification['f1_score']:.4f}"
)

print(
    f"ROC-AUC   : {best_classification['roc_auc']:.4f}"
)


print("\n" + "-" * 70)
print("BEST REGRESSION MODEL")
print("-" * 70)

print(
    f"Model       : {best_regression['model']}"
)

print(
    f"MAE         : {best_regression['mae']:.4f}"
)

print(
    f"RMSE        : {best_regression['rmse']:.4f}"
)

print(
    f"R²          : {best_regression['r2']:.4f}"
)

print(
    f"Adjusted R² : {best_regression['adjusted_r2']:.4f}"
)


# =========================================================
# FINAL RECOMMENDATION
# =========================================================

classification_model = best_classification["model"]

classification_f1 = best_classification["f1_score"]
classification_auc = best_classification["roc_auc"]
classification_recall = best_classification["recall"]

regression_model = best_regression["model"]

regression_r2 = best_regression["r2"]
regression_rmse = best_regression["rmse"]
regression_mae = best_regression["mae"]


recommendation = (
    f"The classification task is best represented by "
    f"{classification_model}, which achieved an F1 score of "
    f"{classification_f1:.4f} and a ROC-AUC of "
    f"{classification_auc:.4f}. Its recall of "
    f"{classification_recall:.4f} indicates how effectively "
    f"the model identifies passengers who survived. "
    f"For the fare regression task, {regression_model} achieved "
    f"an R² of {regression_r2:.4f}, with an RMSE of "
    f"{regression_rmse:.4f} and MAE of {regression_mae:.4f}. "
    f"Therefore, the classification and regression tasks should "
    f"be evaluated using their respective metrics rather than "
    f"comparing raw score values across the two task types."
)


# =========================================================
# CREATE LONG-FORM COMPARISON TABLE
# =========================================================

comparison_rows = []

for _, row in classification_df.iterrows():

    comparison_rows.append(
        {
            "task": "Classification",
            "model": row["model"],
            "metric_group": "Classification Metrics",
            "accuracy": row["accuracy"],
            "precision": row["precision"],
            "recall": row["recall"],
            "f1_score": row["f1_score"],
            "roc_auc": row["roc_auc"],
            "mae": None,
            "rmse": None,
            "r2": None,
            "adjusted_r2": None
        }
    )


for _, row in regression_df.iterrows():

    comparison_rows.append(
        {
            "task": "Regression",
            "model": row["model"],
            "metric_group": "Regression Metrics",
            "accuracy": None,
            "precision": None,
            "recall": None,
            "f1_score": None,
            "roc_auc": None,
            "mae": row["mae"],
            "rmse": row["rmse"],
            "r2": row["r2"],
            "adjusted_r2": row["adjusted_r2"]
        }
    )


comparison_df = pd.DataFrame(
    comparison_rows
)


# =========================================================
# SAVE COMPARISON CSV
# =========================================================

comparison_df.to_csv(
    COMPARISON_PATH,
    index=False
)


# =========================================================
# SAVE MARKDOWN REPORT
# =========================================================

with open(
    REPORT_PATH,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "# Task 14 – Classification vs Regression\n\n"
    )

    file.write(
        "## Classification Metrics\n\n"
    )

    file.write(
        classification_comparison.to_markdown(
            index=False,
            floatfmt=".4f"
        )
    )

    file.write(
        "\n\n"
    )

    file.write(
        "Classification metrics measure the performance of a "
        "binary prediction problem. Accuracy measures overall "
        "correct predictions, while precision, recall and F1 "
        "focus on the positive class. ROC-AUC measures the "
        "model's ability to rank positive examples above "
        "negative examples across classification thresholds.\n\n"
    )

    file.write(
        "## Regression Metrics\n\n"
    )

    file.write(
        regression_comparison.to_markdown(
            index=False,
            floatfmt=".4f"
        )
    )

    file.write(
        "\n\n"
    )

    file.write(
        "Regression metrics evaluate prediction error for a "
        "continuous target. MAE gives the average absolute "
        "prediction error, RMSE penalizes larger errors more "
        "strongly, and R² measures the proportion of target "
        "variance explained by the model. Adjusted R² also "
        "accounts for the number of predictors in the model.\n\n"
    )

    file.write(
        "## Final Recommendation\n\n"
    )

    file.write(
        recommendation
    )

    file.write(
        "\n\n"
    )

    file.write(
        "### Best Classification Model\n\n"
    )

    file.write(
        f"- Model: {classification_model}\n"
        f"- F1 Score: {classification_f1:.4f}\n"
        f"- ROC-AUC: {classification_auc:.4f}\n"
        f"- Recall: {classification_recall:.4f}\n\n"
    )

    file.write(
        "### Best Regression Model\n\n"
    )

    file.write(
        f"- Model: {regression_model}\n"
        f"- R²: {regression_r2:.4f}\n"
        f"- RMSE: {regression_rmse:.4f}\n"
        f"- MAE: {regression_mae:.4f}\n"
    )


# =========================================================
# FINAL OUTPUT
# =========================================================

print("\n" + "=" * 70)
print("FINAL RECOMMENDATION")
print("=" * 70)

print(recommendation)

print("\nComparison table saved to:")
print(COMPARISON_PATH)

print("\nTask 14 report saved to:")
print(REPORT_PATH)

print("\n" + "=" * 70)
print("TASK 14 COMPLETE")
print("=" * 70)