import os
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------
DATA_DIR = "analytics"

RAW_DATA_PATH = os.path.join(DATA_DIR, "titanic.csv")
CLEANED_DATA_PATH = os.path.join(DATA_DIR, "titanic_cleaned.csv")

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

os.makedirs(DATA_DIR, exist_ok=True)


# =========================================================
# TASK 1: LOAD TITANIC DATASET ONCE AND CACHE
# =========================================================

if os.path.exists(RAW_DATA_PATH):
    print("Loading cached Titanic dataset...")
    df = pd.read_csv(RAW_DATA_PATH)
else:
    print("Downloading Titanic dataset using seaborn...")
    df = sns.load_dataset("titanic")

    df.to_csv(
        RAW_DATA_PATH,
        index=False
    )

    print(f"Dataset cached to: {RAW_DATA_PATH}")


# =========================================================
# TASK 1: BASIC PROFILE
# =========================================================

print("\n" + "=" * 70)
print("DATASET SHAPE")
print("=" * 70)

print(f"Rows    : {df.shape[0]}")
print(f"Columns : {df.shape[1]}")


print("\n" + "=" * 70)
print("COLUMN INFORMATION")
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
print("MISSING VALUE ANALYSIS")
print("=" * 70)

missing_count = df.isnull().sum()

missing_percentage = (
    missing_count / len(df)
) * 100

missing_summary = pd.DataFrame(
    {
        "missing_count": missing_count,
        "missing_percentage": missing_percentage
    }
)

missing_summary = missing_summary.sort_values(
    by="missing_percentage",
    ascending=False
)

print(missing_summary)


missing_summary.to_csv(
    MISSING_REPORT_PATH
)

print("\nMissing-value report saved to:")
print(MISSING_REPORT_PATH)


# =========================================================
# TASK 2: CLEAN DATASET
# =========================================================

print("\n" + "=" * 70)
print("CLEANING DATASET")
print("=" * 70)


# Drop deck: 77.22% missing
df_clean = df.drop(
    columns=["deck"]
).copy()

print("Dropped column: deck")


# Drop rows with missing embarked / embark_town
rows_before = len(df_clean)

df_clean = df_clean.dropna(
    subset=[
        "embarked",
        "embark_town"
    ]
).copy()

rows_after_row_drop = len(df_clean)

print(
    "Dropped rows because "
    "embarked/embark_town were missing:",
    rows_before - rows_after_row_drop
)


# Median imputation for age
age_median = df_clean["age"].median()

df_clean["age"] = df_clean["age"].fillna(
    age_median
)

print(
    f"Age median used for imputation: "
    f"{age_median:.2f}"
)


# Verify remaining missing values
remaining_missing = df_clean.isnull().sum()

print("\n" + "=" * 70)
print("REMAINING MISSING VALUES")
print("=" * 70)

print(remaining_missing)


# Save cleaned dataset
df_clean.to_csv(
    CLEANED_DATA_PATH,
    index=False
)

print("\nCleaned dataset saved to:")
print(CLEANED_DATA_PATH)

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
# SAVE PROFILE
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
        "=" * 60 + "\n\n"
    )

    file.write(
        f"Original rows: {df.shape[0]}\n"
    )

    file.write(
        f"Original columns: {df.shape[1]}\n\n"
    )

    file.write(
        "Column Data Types\n"
    )

    file.write(
        "-" * 60 + "\n"
    )

    file.write(
        df.dtypes.to_string()
    )

    file.write("\n\n")

    file.write(
        "Descriptive Statistics\n"
    )

    file.write(
        "-" * 60 + "\n"
    )

    file.write(
        df.describe(
            include="all"
        ).to_string()
    )

    file.write("\n\n")

    file.write(
        "Missing Value Summary\n"
    )

    file.write(
        "-" * 60 + "\n"
    )

    file.write(
        missing_summary.to_string()
    )

    file.write("\n\n")

    file.write(
        "Cleaning Decisions\n"
    )

    file.write(
        "-" * 60 + "\n"
    )

    file.write(
        "deck: 77.22% missing -> dropped column\n"
        "age: 19.87% missing -> median imputation\n"
        "embarked: 0.22% missing -> affected rows dropped\n"
        "embark_town: 0.22% missing -> affected rows dropped\n"
    )

print(
    "\nDataset profile saved to:"
)

print(PROFILE_PATH)


# =========================================================
# TASK 3: UNIVARIATE ANALYSIS
# =========================================================

print("\n" + "=" * 70)
print("TASK 3: UNIVARIATE ANALYSIS")
print("=" * 70)


# ---------------------------------------------------------
# Age Histogram
# ---------------------------------------------------------

plt.figure(
    figsize=(8, 5)
)

plt.hist(
    df_clean["age"],
    bins=20,
    edgecolor="black"
)

plt.xlabel("Age")
plt.ylabel("Frequency")
plt.title("Distribution of Age")

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


# ---------------------------------------------------------
# Fare Histogram
# ---------------------------------------------------------

plt.figure(
    figsize=(8, 5)
)

plt.hist(
    df_clean["fare"],
    bins=30,
    edgecolor="black"
)

plt.xlabel("Fare")
plt.ylabel("Frequency")
plt.title("Distribution of Fare")

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


# ---------------------------------------------------------
# Age Boxplot
# ---------------------------------------------------------

plt.figure(
    figsize=(7, 4)
)

plt.boxplot(
    df_clean["age"],
    orientation="horizontal"
)

plt.xlabel("Age")
plt.title("Age Boxplot")

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


# ---------------------------------------------------------
# Fare Boxplot
# ---------------------------------------------------------

plt.figure(
    figsize=(7, 4)
)

plt.boxplot(
    df_clean["fare"],
    orientation="horizontal"
)

plt.xlabel("Fare")
plt.title("Fare Boxplot")

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


# ---------------------------------------------------------
# IQR Function
# ---------------------------------------------------------

def calculate_iqr_outliers(series):

    q1 = series.quantile(0.25)

    q3 = series.quantile(0.75)

    iqr = q3 - q1

    lower_bound = q1 - (
        1.5 * iqr
    )

    upper_bound = q3 + (
        1.5 * iqr
    )

    outliers = series[
        (series < lower_bound)
        |
        (series > upper_bound)
    ]

    return (
        q1,
        q3,
        iqr,
        lower_bound,
        upper_bound,
        len(outliers)
    )


# ---------------------------------------------------------
# Age IQR
# ---------------------------------------------------------

(
    age_q1,
    age_q3,
    age_iqr,
    age_lower,
    age_upper,
    age_outliers
) = calculate_iqr_outliers(
    df_clean["age"]
)


# ---------------------------------------------------------
# Fare IQR
# ---------------------------------------------------------

(
    fare_q1,
    fare_q3,
    fare_iqr,
    fare_lower,
    fare_upper,
    fare_outliers
) = calculate_iqr_outliers(
    df_clean["fare"]
)


print("\n" + "-" * 70)
print("AGE IQR ANALYSIS")
print("-" * 70)

print(
    f"Q1               : {age_q1:.2f}"
)

print(
    f"Q3               : {age_q3:.2f}"
)

print(
    f"IQR              : {age_iqr:.2f}"
)

print(
    f"Lower Bound      : {age_lower:.2f}"
)

print(
    f"Upper Bound      : {age_upper:.2f}"
)

print(
    f"Outlier Count    : {age_outliers}"
)


print("\n" + "-" * 70)
print("FARE IQR ANALYSIS")
print("-" * 70)

print(
    f"Q1               : {fare_q1:.2f}"
)

print(
    f"Q3               : {fare_q3:.2f}"
)

print(
    f"IQR              : {fare_iqr:.2f}"
)

print(
    f"Lower Bound      : {fare_lower:.2f}"
)

print(
    f"Upper Bound      : {fare_upper:.2f}"
)

print(
    f"Outlier Count    : {fare_outliers}"
)


# ---------------------------------------------------------
# Fare Statistics
# ---------------------------------------------------------

fare_mean = df_clean["fare"].mean()

fare_median = df_clean["fare"].median()

fare_mode = (
    df_clean["fare"]
    .mode()
    .iloc[0]
)

fare_skewness = (
    df_clean["fare"].skew()
)


print("\n" + "-" * 70)
print("FARE STATISTICS")
print("-" * 70)

print(
    f"Mean       : {fare_mean:.4f}"
)

print(
    f"Median     : {fare_median:.4f}"
)

print(
    f"Mode       : {fare_mode:.4f}"
)

print(
    f"Skewness   : {fare_skewness:.4f}"
)


# ---------------------------------------------------------
# Save Task 3 report
# ---------------------------------------------------------

with open(
    TASK3_REPORT_PATH,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "TASK 3 - UNIVARIATE ANALYSIS\n"
    )

    file.write(
        "=" * 60 + "\n\n"
    )

    file.write(
        "AGE IQR ANALYSIS\n"
    )

    file.write(
        "-" * 60 + "\n"
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
        f"Lower Bound: {age_lower:.4f}\n"
    )

    file.write(
        f"Upper Bound: {age_upper:.4f}\n"
    )

    file.write(
        f"Outlier Count: {age_outliers}\n\n"
    )

    file.write(
        "FARE IQR ANALYSIS\n"
    )

    file.write(
        "-" * 60 + "\n"
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
        f"Lower Bound: {fare_lower:.4f}\n"
    )

    file.write(
        f"Upper Bound: {fare_upper:.4f}\n"
    )

    file.write(
        f"Outlier Count: {fare_outliers}\n\n"
    )

    file.write(
        "FARE STATISTICS\n"
    )

    file.write(
        "-" * 60 + "\n"
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

print(
    "\nTask 3 report saved to:"
)

print(TASK3_REPORT_PATH)


# =========================================================
# TASK 4: BIVARIATE ANALYSIS
# =========================================================

print("\n" + "=" * 70)
print("TASK 4: BIVARIATE ANALYSIS")
print("=" * 70)


# ---------------------------------------------------------
# 4.1 Survival by Sex
# ---------------------------------------------------------

survival_by_sex = (
    df_clean
    .groupby("sex", observed=True)["survived"]
    .agg(
        survival_rate="mean",
        passenger_count="count"
    )
    .reset_index()
)

survival_by_sex["survival_rate"] *= 100

print("\n" + "-" * 70)
print("SURVIVAL RATE BY SEX")
print("-" * 70)

print(
    survival_by_sex.to_string(
        index=False
    )
)


# ---------------------------------------------------------
# 4.2 Survival by Passenger Class
# ---------------------------------------------------------

survival_by_pclass = (
    df_clean
    .groupby("pclass", observed=True)["survived"]
    .agg(
        survival_rate="mean",
        passenger_count="count"
    )
    .reset_index()
)

survival_by_pclass["survival_rate"] *= 100

print("\n" + "-" * 70)
print("SURVIVAL RATE BY PCLASS")
print("-" * 70)

print(
    survival_by_pclass.to_string(
        index=False
    )
)


# ---------------------------------------------------------
# 4.3 Survival by Sex + Passenger Class
# ---------------------------------------------------------

survival_by_sex_pclass = (
    df_clean
    .groupby(
        ["sex", "pclass"],
        observed=True
    )["survived"]
    .agg(
        survival_rate="mean",
        passenger_count="count"
    )
    .reset_index()
)

survival_by_sex_pclass["survival_rate"] *= 100

print("\n" + "-" * 70)
print("SURVIVAL RATE BY SEX + PCLASS")
print("-" * 70)

print(
    survival_by_sex_pclass.to_string(
        index=False
    )
)


# ---------------------------------------------------------
# 4.4 Correlation Matrix
# Exactly six required columns
# ---------------------------------------------------------

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
    correlation_matrix.round(4)
)


# ---------------------------------------------------------
# 4.5 Heatmap
# ---------------------------------------------------------

plt.figure(
    figsize=(8, 6)
)

sns.heatmap(
    correlation_matrix,
    annot=True,
    fmt=".2f",
    linewidths=0.5,
    square=True
)

plt.title(
    "Titanic Correlation Matrix"
)

plt.tight_layout()

CORRELATION_HEATMAP_PATH = os.path.join(
    DATA_DIR,
    "correlation_heatmap.png"
)

plt.savefig(
    CORRELATION_HEATMAP_PATH,
    dpi=150
)

plt.close()

print(
    "\nCorrelation heatmap saved to:"
)

print(
    CORRELATION_HEATMAP_PATH
)


# ---------------------------------------------------------
# 4.6 Find two strongest absolute
# off-diagonal correlations
# ---------------------------------------------------------

correlation_pairs = []

for i in range(
    len(correlation_columns)
):

    for j in range(
        i + 1,
        len(correlation_columns)
    ):

        col_1 = correlation_columns[i]
        col_2 = correlation_columns[j]

        value = correlation_matrix.loc[
            col_1,
            col_2
        ]

        correlation_pairs.append(
            {
                "variable_1": col_1,
                "variable_2": col_2,
                "correlation": value,
                "absolute_correlation": abs(value)
            }
        )


correlation_pairs_df = pd.DataFrame(
    correlation_pairs
)

strongest_two = (
    correlation_pairs_df
    .sort_values(
        by="absolute_correlation",
        ascending=False
    )
    .head(2)
)


print("\n" + "-" * 70)
print("TWO STRONGEST ABSOLUTE CORRELATIONS")
print("-" * 70)

print(
    strongest_two.to_string(
        index=False
    )
)


# ---------------------------------------------------------
# Save Task 4 numerical outputs
# ---------------------------------------------------------

SURVIVAL_SEX_PATH = os.path.join(
    DATA_DIR,
    "survival_by_sex.csv"
)

SURVIVAL_PCLASS_PATH = os.path.join(
    DATA_DIR,
    "survival_by_pclass.csv"
)

SURVIVAL_SEX_PCLASS_PATH = os.path.join(
    DATA_DIR,
    "survival_by_sex_pclass.csv"
)

CORRELATION_MATRIX_PATH = os.path.join(
    DATA_DIR,
    "correlation_matrix.csv"
)

STRONGEST_CORRELATIONS_PATH = os.path.join(
    DATA_DIR,
    "strongest_correlations.csv"
)


survival_by_sex.to_csv(
    SURVIVAL_SEX_PATH,
    index=False
)

survival_by_pclass.to_csv(
    SURVIVAL_PCLASS_PATH,
    index=False
)

survival_by_sex_pclass.to_csv(
    SURVIVAL_SEX_PCLASS_PATH,
    index=False
)

correlation_matrix.to_csv(
    CORRELATION_MATRIX_PATH
)

strongest_two.to_csv(
    STRONGEST_CORRELATIONS_PATH,
    index=False
)


# ---------------------------------------------------------
# Save Task 4 report
# ---------------------------------------------------------

with open(
    TASK4_REPORT_PATH,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "TASK 4 - BIVARIATE ANALYSIS\n"
    )

    file.write(
        "=" * 60 + "\n\n"
    )

    file.write(
        "SURVIVAL BY SEX\n"
    )

    file.write(
        "-" * 60 + "\n"
    )

    file.write(
        survival_by_sex.to_string(
            index=False
        )
    )

    file.write("\n\n")

    file.write(
        "SURVIVAL BY PCLASS\n"
    )

    file.write(
        "-" * 60 + "\n"
    )

    file.write(
        survival_by_pclass.to_string(
            index=False
        )
    )

    file.write("\n\n")

    file.write(
        "SURVIVAL BY SEX + PCLASS\n"
    )

    file.write(
        "-" * 60 + "\n"
    )

    file.write(
        survival_by_sex_pclass.to_string(
            index=False
        )
    )

    file.write("\n\n")

    file.write(
        "CORRELATION MATRIX\n"
    )

    file.write(
        "-" * 60 + "\n"
    )

    file.write(
        correlation_matrix.round(4).to_string()
    )

    file.write("\n\n")

    file.write(
        "TWO STRONGEST ABSOLUTE CORRELATIONS\n"
    )

    file.write(
        "-" * 60 + "\n"
    )

    file.write(
        strongest_two.to_string(
            index=False
        )
    )


print(
    "\nTask 4 report saved to:"
)

print(TASK4_REPORT_PATH)


# =========================================================
# FINAL PREVIEW
# =========================================================

print("\n" + "=" * 70)
print("CLEANED DATASET - FIRST 5 ROWS")
print("=" * 70)

print(
    df_clean.head()
)
