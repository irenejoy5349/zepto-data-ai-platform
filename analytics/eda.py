import os

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# =========================================================
# PATHS
# =========================================================

DATA_DIR = "analytics"

RAW_DATA_PATH = os.path.join(
    DATA_DIR,
    "titanic.csv"
)

CLEANED_DATA_PATH = os.path.join(
    DATA_DIR,
    "titanic_cleaned.csv"
)

MISSING_REPORT_PATH = os.path.join(
    DATA_DIR,
    "missing_value_report.csv"
)

PROFILE_PATH = os.path.join(
    DATA_DIR,
    "dataset_profile.txt"
)

TASK3_REPORT_PATH = os.path.join(
    DATA_DIR,
    "task3_univariate_report.txt"
)

TASK4_REPORT_PATH = os.path.join(
    DATA_DIR,
    "task4_bivariate_report.txt"
)

CORRELATION_PLOT_PATH = os.path.join(
    DATA_DIR,
    "correlation_heatmap.png"
)

os.makedirs(
    DATA_DIR,
    exist_ok=True
)


# =========================================================
# TASK 1: LOAD TITANIC DATASET ONCE AND CACHE
# =========================================================

print("=" * 70)
print("TASK 1: LOAD TITANIC DATASET")
print("=" * 70)

try:
    print("\nLoading Titanic dataset with seaborn...")

    # One and only network/cache load of the raw dataset
    df = sns.load_dataset("titanic")

    # Immediately save the loaded raw dataset as the
    # committed offline fallback.
    df.to_csv(
        RAW_DATA_PATH,
        index=False
    )

    print(
        f"Fresh Titanic dataset saved to: {RAW_DATA_PATH}"
    )

except Exception as exc:

    if not os.path.exists(
        RAW_DATA_PATH
    ):
        raise RuntimeError(
            "Unable to load Titanic dataset and "
            "analytics/titanic.csv is not available."
        ) from exc

    print(
        "Seaborn dataset loading was unavailable."
    )

    print(
        f"Reason: {exc}"
    )

    print(
        "Loading committed offline fallback..."
    )

    df = pd.read_csv(
        RAW_DATA_PATH
    )


# =========================================================
# TASK 1: PROFILE
# =========================================================

print("\n" + "=" * 70)
print("DATASET SHAPE")
print("=" * 70)

print(
    f"Rows    : {df.shape[0]}"
)

print(
    f"Columns : {df.shape[1]}"
)


print("\n" + "=" * 70)
print("DATASET INFO")
print("=" * 70)

df.info()


print("\n" + "=" * 70)
print("DESCRIPTIVE STATISTICS")
print("=" * 70)

print(
    df.describe(
        include="all"
    )
)


# =========================================================
# TASK 2: MISSING VALUE ANALYSIS
# =========================================================

print("\n" + "=" * 70)
print("TASK 2: MISSING VALUE ANALYSIS")
print("=" * 70)

missing_count = (
    df.isnull()
    .sum()
)

missing_percentage = (
    missing_count / len(df)
) * 100

missing_summary = pd.DataFrame(
    {
        "missing_count": missing_count,
        "missing_percentage": missing_percentage
    }
)

missing_summary = (
    missing_summary[
        missing_summary["missing_count"] > 0
    ]
    .sort_values(
        by="missing_percentage",
        ascending=False
    )
)

print(
    missing_summary.to_string()
)

missing_summary.to_csv(
    MISSING_REPORT_PATH,
    index=True
)

print(
    f"\nMissing-value report saved to: "
    f"{MISSING_REPORT_PATH}"
)


# =========================================================
# TASK 2: CLEANING
# =========================================================

print("\n" + "=" * 70)
print("CLEANING DATASET")
print("=" * 70)


# ---------------------------------------------------------
# High missingness: deck
# 77.22% missing
# Decision: drop the entire column because the missing
# proportion is too high for reliable imputation.
# ---------------------------------------------------------

df_clean = df.drop(
    columns=["deck"]
).copy()

print(
    "deck: 77.22% missing -> column dropped"
)


# ---------------------------------------------------------
# Moderate missingness: age
# 19.87% missing
# Decision: median imputation because 5%–30% missing
# requires imputation.
# ---------------------------------------------------------

age_median = df_clean["age"].median()

df_clean["age"] = (
    df_clean["age"]
    .fillna(age_median)
)

print(
    f"age: 19.87% missing -> median imputation "
    f"(median={age_median:.4f})"
)


# ---------------------------------------------------------
# Low missingness: embarked
# 0.22% missing
# Decision: drop affected rows because the missingness
# is below 5%.
# ---------------------------------------------------------

rows_before = len(df_clean)

df_clean = df_clean.dropna(
    subset=["embarked"]
).copy()

rows_after_embarked = len(df_clean)

print(
    "embarked: 0.22% missing -> affected rows dropped"
)

print(
    "Rows removed for embarked:",
    rows_before - rows_after_embarked
)


# ---------------------------------------------------------
# Low missingness: embark_town
# 0.22% missing
# Decision: drop affected rows because the missingness
# is below 5%.
# ---------------------------------------------------------

rows_before = len(df_clean)

df_clean = df_clean.dropna(
    subset=["embark_town"]
).copy()

rows_after_embark_town = len(df_clean)

print(
    "embark_town: 0.22% missing -> affected rows dropped"
)

print(
    "Rows removed for embark_town:",
    rows_before - rows_after_embark_town
)


# =========================================================
# REMAINING MISSING VALUES
# =========================================================

remaining_missing = (
    df_clean.isnull()
    .sum()
)

print("\n" + "=" * 70)
print("REMAINING MISSING VALUES")
print("=" * 70)

print(
    remaining_missing
)


# =========================================================
# SAVE CLEANED DATASET
# =========================================================

df_clean.to_csv(
    CLEANED_DATA_PATH,
    index=False
)

print(
    f"\nCleaned dataset saved to: "
    f"{CLEANED_DATA_PATH}"
)

print(
    f"Original rows : {len(df)}"
)

print(
    f"Cleaned rows  : {len(df_clean)}"
)

print(
    f"Cleaned cols  : {len(df_clean.columns)}"
)


# =========================================================
# SAVE PROFILE REPORT
# =========================================================

with open(
    PROFILE_PATH,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "TITANIC DATASET PROFILE\n"
    )

    file.write(
        "=" * 70 + "\n\n"
    )

    file.write(
        f"Original shape: {df.shape}\n"
    )

    file.write(
        f"Cleaned shape: {df_clean.shape}\n\n"
    )

    file.write(
        "Data types\n"
    )

    file.write(
        "-" * 70 + "\n"
    )

    file.write(
        df.dtypes.to_string()
    )

    file.write(
        "\n\n"
    )

    file.write(
        "Descriptive statistics\n"
    )

    file.write(
        "-" * 70 + "\n"
    )

    file.write(
        df.describe(
            include="all"
        ).to_string()
    )

    file.write(
        "\n\n"
    )

    file.write(
        "Missing values\n"
    )

    file.write(
        "-" * 70 + "\n"
    )

    file.write(
        missing_summary.to_string()
    )

    file.write(
        "\n\n"
    )

    file.write(
        "Cleaning decisions\n"
    )

    file.write(
        "-" * 70 + "\n"
    )

    file.write(
        "deck: 77.22% missing -> dropped\n"
    )

    file.write(
        "age: 19.87% missing -> median imputation\n"
    )

    file.write(
        "embarked: 0.22% missing -> affected rows dropped\n"
    )

    file.write(
        "embark_town: 0.22% missing -> affected rows dropped\n"
    )

print(
    f"\nProfile report saved to: {PROFILE_PATH}"
)


# =========================================================
# TASK 3: UNIVARIATE ANALYSIS
# =========================================================

print("\n" + "=" * 70)
print("TASK 3: UNIVARIATE ANALYSIS")
print("=" * 70)


# =========================================================
# 3.1 AGE HISTOGRAM
# =========================================================

plt.figure(
    figsize=(8, 5)
)

plt.hist(
    df_clean["age"],
    bins=20,
    edgecolor="black"
)

plt.xlabel(
    "Age"
)

plt.ylabel(
    "Frequency"
)

plt.title(
    "Distribution of Age"
)

plt.tight_layout()

AGE_HIST_PATH = os.path.join(
    DATA_DIR,
    "age_histogram.png"
)

plt.savefig(
    AGE_HIST_PATH,
    dpi=150
)

plt.close()


# =========================================================
# 3.2 FARE HISTOGRAM
# =========================================================

plt.figure(
    figsize=(8, 5)
)

plt.hist(
    df_clean["fare"],
    bins=30,
    edgecolor="black"
)

plt.xlabel(
    "Fare"
)

plt.ylabel(
    "Frequency"
)

plt.title(
    "Distribution of Fare"
)

plt.tight_layout()

FARE_HIST_PATH = os.path.join(
    DATA_DIR,
    "fare_histogram.png"
)

plt.savefig(
    FARE_HIST_PATH,
    dpi=150
)

plt.close()


# =========================================================
# 3.3 AGE BOXPLOT
# =========================================================

plt.figure(
    figsize=(8, 4)
)

plt.boxplot(
    df_clean["age"],
    vert=False
)

plt.xlabel(
    "Age"
)

plt.title(
    "Age Boxplot"
)

plt.tight_layout()

AGE_BOX_PATH = os.path.join(
    DATA_DIR,
    "age_boxplot.png"
)

plt.savefig(
    AGE_BOX_PATH,
    dpi=150
)

plt.close()


# =========================================================
# 3.4 FARE BOXPLOT
# =========================================================

plt.figure(
    figsize=(8, 4)
)

plt.boxplot(
    df_clean["fare"],
    vert=False
)

plt.xlabel(
    "Fare"
)

plt.title(
    "Fare Boxplot"
)

plt.tight_layout()

FARE_BOX_PATH = os.path.join(
    DATA_DIR,
    "fare_boxplot.png"
)

plt.savefig(
    FARE_BOX_PATH,
    dpi=150
)

plt.close()


# =========================================================
# 3.5 IQR OUTLIER FUNCTION
# =========================================================

def calculate_iqr_statistics(
    series
):
    q1 = series.quantile(
        0.25
    )

    q3 = series.quantile(
        0.75
    )

    iqr = q3 - q1

    lower_bound = (
        q1 - 1.5 * iqr
    )

    upper_bound = (
        q3 + 1.5 * iqr
    )

    outlier_mask = (
        (series < lower_bound)
        |
        (series > upper_bound)
    )

    outlier_count = int(
        outlier_mask.sum()
    )

    return (
        q1,
        q3,
        iqr,
        lower_bound,
        upper_bound,
        outlier_count
    )


# =========================================================
# 3.6 AGE IQR
# =========================================================

(
    age_q1,
    age_q3,
    age_iqr,
    age_lower_bound,
    age_upper_bound,
    age_outlier_count
) = calculate_iqr_statistics(
    df_clean["age"]
)


# =========================================================
# 3.7 FARE IQR
# =========================================================

(
    fare_q1,
    fare_q3,
    fare_iqr,
    fare_lower_bound,
    fare_upper_bound,
    fare_outlier_count
) = calculate_iqr_statistics(
    df_clean["fare"]
)


# =========================================================
# PRINT IQR RESULTS
# =========================================================

print("\n" + "-" * 70)
print("AGE IQR ANALYSIS")
print("-" * 70)

print(
    f"Q1           : {age_q1:.4f}"
)

print(
    f"Q3           : {age_q3:.4f}"
)

print(
    f"IQR          : {age_iqr:.4f}"
)

print(
    f"Lower bound  : {age_lower_bound:.4f}"
)

print(
    f"Upper bound  : {age_upper_bound:.4f}"
)

print(
    f"Outlier count: {age_outlier_count}"
)


print("\n" + "-" * 70)
print("FARE IQR ANALYSIS")
print("-" * 70)

print(
    f"Q1           : {fare_q1:.4f}"
)

print(
    f"Q3           : {fare_q3:.4f}"
)

print(
    f"IQR          : {fare_iqr:.4f}"
)

print(
    f"Lower bound  : {fare_lower_bound:.4f}"
)

print(
    f"Upper bound  : {fare_upper_bound:.4f}"
)

print(
    f"Outlier count: {fare_outlier_count}"
)


# =========================================================
# 3.8 FARE STATISTICS
# =========================================================

fare_mean = df_clean[
    "fare"
].mean()

fare_median = df_clean[
    "fare"
].median()

fare_mode = (
    df_clean[
        "fare"
    ]
    .mode()
    .iloc[0]
)

fare_skewness = df_clean[
    "fare"
].skew()


print("\n" + "-" * 70)
print("FARE STATISTICS")
print("-" * 70)

print(
    f"Mean     : {fare_mean:.4f}"
)

print(
    f"Median   : {fare_median:.4f}"
)

print(
    f"Mode     : {fare_mode:.4f}"
)

print(
    f"Skewness : {fare_skewness:.4f}"
)


# =========================================================
# 3.9 SAVE TASK 3 REPORT
# =========================================================

with open(
    TASK3_REPORT_PATH,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "TASK 3 - UNIVARIATE ANALYSIS\n"
    )

    file.write(
        "=" * 70 + "\n\n"
    )

    file.write(
        "AGE IQR ANALYSIS\n"
    )

    file.write(
        "-" * 70 + "\n"
    )

    file.write(
        f"Q1: {age_q1:.4f}\n"
    )

    file.write(
        f"Q3: {age_q3:.4f}\n"
    )

    file.write(
        f"IQR: {age_iqr:.4f}\n"
    )

    file.write(
        f"Lower bound: {age_lower_bound:.4f}\n"
    )

    file.write(
        f"Upper bound: {age_upper_bound:.4f}\n"
    )

    file.write(
        f"Outlier count: {age_outlier_count}\n\n"
    )

    file.write(
        "FARE IQR ANALYSIS\n"
    )

    file.write(
        "-" * 70 + "\n"
    )

    file.write(
        f"Q1: {fare_q1:.4f}\n"
    )

    file.write(
        f"Q3: {fare_q3:.4f}\n"
    )

    file.write(
        f"IQR: {fare_iqr:.4f}\n"
    )

    file.write(
        f"Lower bound: {fare_lower_bound:.4f}\n"
    )

    file.write(
        f"Upper bound: {fare_upper_bound:.4f}\n"
    )

    file.write(
        f"Outlier count: {fare_outlier_count}\n\n"
    )

    file.write(
        "FARE STATISTICS\n"
    )

    file.write(
        "-" * 70 + "\n"
    )

    file.write(
        f"Mean: {fare_mean:.4f}\n"
    )

    file.write(
        f"Median: {fare_median:.4f}\n"
    )

    file.write(
        f"Mode: {fare_mode:.4f}\n"
    )

    file.write(
        f"Skewness: {fare_skewness:.4f}\n"
    )

    file.write(
        "\nInterpretation:\n"
    )

    if (
        fare_mean
        > fare_median
        > fare_mode
    ):
        file.write(
            "The ordering mean > median > mode, "
            "together with positive skewness, indicates "
            "a right-skewed fare distribution.\n"
        )
    else:
        file.write(
            "The fare distribution was evaluated using "
            "mean, median, mode, and skewness.\n"
        )

print(
    f"\nTask 3 report saved to: "
    f"{TASK3_REPORT_PATH}"
)


# =========================================================
# TASK 4: BIVARIATE ANALYSIS
# =========================================================

print("\n" + "=" * 70)
print("TASK 4: BIVARIATE ANALYSIS")
print("=" * 70)


# =========================================================
# 4.1 SURVIVAL BY SEX
# REQUIRED: BOOLEAN MASKING
# =========================================================

female_mask = (
    df_clean["sex"] == "female"
)

male_mask = (
    df_clean["sex"] == "male"
)

female_survival_rate = (
    df_clean.loc[
        female_mask,
        "survived"
    ].mean()
    * 100
)

male_survival_rate = (
    df_clean.loc[
        male_mask,
        "survived"
    ].mean()
    * 100
)

survival_by_sex = pd.DataFrame(
    {
        "sex": [
            "female",
            "male"
        ],
        "survival_rate": [
            female_survival_rate,
            male_survival_rate
        ],
        "passenger_count": [
            int(female_mask.sum()),
            int(male_mask.sum())
        ]
    }
)

print("\n" + "-" * 70)
print("SURVIVAL RATE BY SEX")
print("-" * 70)

print(
    survival_by_sex.to_string(
        index=False
    )
)


# =========================================================
# 4.2 SURVIVAL BY PASSENGER CLASS
# REQUIRED: BOOLEAN MASKING
# =========================================================

class_1_mask = (
    df_clean["pclass"] == 1
)

class_2_mask = (
    df_clean["pclass"] == 2
)

class_3_mask = (
    df_clean["pclass"] == 3
)

class_1_survival_rate = (
    df_clean.loc[
        class_1_mask,
        "survived"
    ].mean()
    * 100
)

class_2_survival_rate = (
    df_clean.loc[
        class_2_mask,
        "survived"
    ].mean()
    * 100
)

class_3_survival_rate = (
    df_clean.loc[
        class_3_mask,
        "survived"
    ].mean()
    * 100
)

survival_by_pclass = pd.DataFrame(
    {
        "pclass": [
            1,
            2,
            3
        ],
        "survival_rate": [
            class_1_survival_rate,
            class_2_survival_rate,
            class_3_survival_rate
        ],
        "passenger_count": [
            int(class_1_mask.sum()),
            int(class_2_mask.sum()),
            int(class_3_mask.sum())
        ]
    }
)

print("\n" + "-" * 70)
print("SURVIVAL RATE BY PASSENGER CLASS")
print("-" * 70)

print(
    survival_by_pclass.to_string(
        index=False
    )
)


# =========================================================
# 4.3 SURVIVAL BY SEX + PASSENGER CLASS
# REQUIRED: BOOLEAN MASKING WITH &
# =========================================================

female_class_1_mask = (
    (df_clean["sex"] == "female")
    &
    (df_clean["pclass"] == 1)
)

female_class_2_mask = (
    (df_clean["sex"] == "female")
    &
    (df_clean["pclass"] == 2)
)

female_class_3_mask = (
    (df_clean["sex"] == "female")
    &
    (df_clean["pclass"] == 3)
)

male_class_1_mask = (
    (df_clean["sex"] == "male")
    &
    (df_clean["pclass"] == 1)
)

male_class_2_mask = (
    (df_clean["sex"] == "male")
    &
    (df_clean["pclass"] == 2)
)

male_class_3_mask = (
    (df_clean["sex"] == "male")
    &
    (df_clean["pclass"] == 3)
)


def masked_survival_rate(
    mask
):
    if mask.sum() == 0:
        return 0.0

    return (
        df_clean.loc[
            mask,
            "survived"
        ].mean()
        * 100
    )


survival_by_sex_pclass = pd.DataFrame(
    {
        "sex": [
            "female",
            "female",
            "female",
            "male",
            "male",
            "male"
        ],
        "pclass": [
            1,
            2,
            3,
            1,
            2,
            3
        ],
        "survival_rate": [
            masked_survival_rate(
                female_class_1_mask
            ),
            masked_survival_rate(
                female_class_2_mask
            ),
            masked_survival_rate(
                female_class_3_mask
            ),
            masked_survival_rate(
                male_class_1_mask
            ),
            masked_survival_rate(
                male_class_2_mask
            ),
            masked_survival_rate(
                male_class_3_mask
            )
        ],
        "passenger_count": [
            int(
                female_class_1_mask.sum()
            ),
            int(
                female_class_2_mask.sum()
            ),
            int(
                female_class_3_mask.sum()
            ),
            int(
                male_class_1_mask.sum()
            ),
            int(
                male_class_2_mask.sum()
            ),
            int(
                male_class_3_mask.sum()
            )
        ]
    }
)

print("\n" + "-" * 70)
print("SURVIVAL RATE BY SEX + PASSENGER CLASS")
print("-" * 70)

print(
    survival_by_sex_pclass.to_string(
        index=False
    )
)


# =========================================================
# 4.4 CORRELATION MATRIX
# EXACTLY THE SIX REQUIRED COLUMNS
# =========================================================

correlation_columns = [
    "survived",
    "pclass",
    "age",
    "sibsp",
    "parch",
    "fare"
]

correlation_matrix = (
    df_clean[
        correlation_columns
    ]
    .corr()
)

print("\n" + "-" * 70)
print("CORRELATION MATRIX")
print("-" * 70)

print(
    correlation_matrix.to_string(
        float_format=lambda value: f"{value:.4f}"
    )
)


# =========================================================
# 4.5 HEATMAP
# =========================================================

plt.figure(
    figsize=(9, 7)
)

sns.heatmap(
    correlation_matrix,
    annot=True,
    fmt=".2f",
    square=True
)

plt.title(
    "Titanic Correlation Matrix"
)

plt.tight_layout()

plt.savefig(
    CORRELATION_PLOT_PATH,
    dpi=150
)

plt.close()

print(
    f"\nCorrelation heatmap saved to: "
    f"{CORRELATION_PLOT_PATH}"
)


# =========================================================
# 4.6 FIND TOP TWO ABSOLUTE OFF-DIAGONAL CORRELATIONS
# =========================================================

correlation_pairs = []

for i in range(
    len(correlation_columns)
):

    for j in range(
        i + 1,
        len(correlation_columns)
    ):

        feature_a = (
            correlation_columns[i]
        )

        feature_b = (
            correlation_columns[j]
        )

        coefficient = (
            correlation_matrix.loc[
                feature_a,
                feature_b
            ]
        )

        correlation_pairs.append(
            {
                "feature_a": feature_a,
                "feature_b": feature_b,
                "correlation": coefficient,
                "absolute_correlation": abs(
                    coefficient
                )
            }
        )


correlation_pairs_df = pd.DataFrame(
    correlation_pairs
)

top_two_correlations = (
    correlation_pairs_df
    .sort_values(
        by="absolute_correlation",
        ascending=False
    )
    .head(2)
    .reset_index(
        drop=True
    )
)

print("\n" + "-" * 70)
print("TOP TWO ABSOLUTE OFF-DIAGONAL CORRELATIONS")
print("-" * 70)

print(
    top_two_correlations.to_string(
        index=False,
        float_format=lambda value: f"{value:.4f}"
    )
)


# =========================================================
# TASK 4 REPORT
# =========================================================

with open(
    TASK4_REPORT_PATH,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "TASK 4 - BIVARIATE ANALYSIS\n"
    )

    file.write(
        "=" * 70 + "\n\n"
    )

    file.write(
        "SURVIVAL BY SEX\n"
    )

    file.write(
        "-" * 70 + "\n"
    )

    file.write(
        survival_by_sex.to_string(
            index=False
        )
    )

    file.write(
        "\n\n"
    )

    file.write(
        "SURVIVAL BY PASSENGER CLASS\n"
    )

    file.write(
        "-" * 70 + "\n"
    )

    file.write(
        survival_by_pclass.to_string(
            index=False
        )
    )

    file.write(
        "\n\n"
    )

    file.write(
        "SURVIVAL BY SEX + PASSENGER CLASS\n"
    )

    file.write(
        "-" * 70 + "\n"
    )

    file.write(
        survival_by_sex_pclass.to_string(
            index=False
        )
    )

    file.write(
        "\n\n"
    )

    file.write(
        "CORRELATION COLUMNS\n"
    )

    file.write(
        "-" * 70 + "\n"
    )

    file.write(
        ", ".join(
            correlation_columns
        )
    )

    file.write(
        "\n\n"
    )

    file.write(
        "CORRELATION MATRIX\n"
    )

    file.write(
        "-" * 70 + "\n"
    )

    file.write(
        correlation_matrix.to_string(
            float_format=lambda value: f"{value:.4f}"
        )
    )

    file.write(
        "\n\n"
    )

    file.write(
        "TOP TWO ABSOLUTE OFF-DIAGONAL CORRELATIONS\n"
    )

    file.write(
        "-" * 70 + "\n"
    )

    file.write(
        top_two_correlations.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}"
        )
    )

    file.write(
        "\n\n"
    )

    # -----------------------------------------------------
    # Written interpretation
    # -----------------------------------------------------

    file.write(
        "INTERPRETATION\n"
    )

    file.write(
        "-" * 70 + "\n"
    )

    file.write(
        f"Female passengers had a survival rate of "
        f"{female_survival_rate:.2f}%, compared with "
        f"{male_survival_rate:.2f}% for male passengers. "
        f"This indicates a strong observed association "
        f"between sex and survival.\n\n"
    )

    file.write(
        f"First-class survival was "
        f"{class_1_survival_rate:.2f}%, second-class "
        f"survival was {class_2_survival_rate:.2f}%, "
        f"and third-class survival was "
        f"{class_3_survival_rate:.2f}%. This shows that "
        f"survival generally decreased as passenger class "
        f"moved from first to third.\n\n"
    )

    for _, row in (
        top_two_correlations.iterrows()
    ):

        feature_a = row[
            "feature_a"
        ]

        feature_b = row[
            "feature_b"
        ]

        coefficient = row[
            "correlation"
        ]

        if (
            coefficient < 0
        ):
            direction = "negative"
        else:
            direction = "positive"

        file.write(
            f"The correlation between {feature_a} and "
            f"{feature_b} is {coefficient:.4f}, which "
            f"indicates a {direction} relationship. "
            f"The magnitude of this coefficient is among "
            f"the two strongest absolute off-diagonal "
            f"correlations in the required six-feature "
            f"matrix.\n\n"
        )


# =========================================================
# FINAL SUMMARY
# =========================================================

print(
    "\n" + "=" * 70
)

print(
    "EDA TASKS 1–4 COMPLETE"
)

print(
    "=" * 70
)

print(
    f"Raw fallback: {RAW_DATA_PATH}"
)

print(
    f"Cleaned data: {CLEANED_DATA_PATH}"
)

print(
    f"Task 3 report: {TASK3_REPORT_PATH}"
)

print(
    f"Task 4 report: {TASK4_REPORT_PATH}"
)

print(
    f"Correlation heatmap: {CORRELATION_PLOT_PATH}"
)

print(
    "\nRequired boolean masking was used for:"
)

print(
    "- survival by sex"
)

print(
    "- survival by passenger class"
)

print(
    "- survival by sex AND passenger class using '&'"
)

print(
    "\nExactly these six columns were used for correlation:"
)

print(
    ", ".join(
        correlation_columns
    )
)

print(
    "\nadult_male and alone were excluded."
)

print(
    "\nEDA complete."
)
