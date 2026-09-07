import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------
DATA_DIR = "analytics"

CLEANED_DATA_PATH = os.path.join(
    DATA_DIR,
    "titanic_cleaned.csv"
)

REPORT_PATH = os.path.join(
    DATA_DIR,
    "task6_standardization_report.txt"
)

CHART_PATH = os.path.join(
    DATA_DIR,
    "task6_standardization_comparison.png"
)


# ---------------------------------------------------------
# Load the same cleaned dataset
# ---------------------------------------------------------
df = pd.read_csv(CLEANED_DATA_PATH)

print("=" * 70)
print("TASK 6: EXPLORATORY STANDARDIZATION")
print("=" * 70)

print(f"Rows loaded : {len(df)}")


# ---------------------------------------------------------
# Select numeric columns
# ---------------------------------------------------------
features = ["age", "fare"]

original_data = df[features].copy()


# ---------------------------------------------------------
# Statistics BEFORE standardization
# ---------------------------------------------------------
before_mean = original_data.mean()
before_std = original_data.std()


print("\n" + "-" * 70)
print("BEFORE STANDARDIZATION")
print("-" * 70)

for column in features:
    print(
        f"{column:5s} -> "
        f"mean: {before_mean[column]:.4f}, "
        f"std: {before_std[column]:.4f}"
    )


# ---------------------------------------------------------
# Standardization
# ---------------------------------------------------------
scaler = StandardScaler()

standardized_values = scaler.fit_transform(
    original_data
)

standardized_data = pd.DataFrame(
    standardized_values,
    columns=features
)


# ---------------------------------------------------------
# Statistics AFTER standardization
# ---------------------------------------------------------
after_mean = standardized_data.mean()
after_std = standardized_data.std()


print("\n" + "-" * 70)
print("AFTER STANDARDIZATION")
print("-" * 70)

for column in features:
    print(
        f"{column:5s} -> "
        f"mean: {after_mean[column]:.4f}, "
        f"std: {after_std[column]:.4f}"
    )


# ---------------------------------------------------------
# Comparison chart
# ---------------------------------------------------------
fig, axes = plt.subplots(
    2,
    2,
    figsize=(12, 8)
)

# Age before
axes[0, 0].hist(
    original_data["age"],
    bins=20,
    edgecolor="black"
)

axes[0, 0].set_title(
    "Age Before Standardization"
)

axes[0, 0].set_xlabel("Age")
axes[0, 0].set_ylabel("Frequency")


# Age after
axes[0, 1].hist(
    standardized_data["age"],
    bins=20,
    edgecolor="black"
)

axes[0, 1].set_title(
    "Age After Standardization"
)

axes[0, 1].set_xlabel("Standardized Age")
axes[0, 1].set_ylabel("Frequency")


# Fare before
axes[1, 0].hist(
    original_data["fare"],
    bins=30,
    edgecolor="black"
)

axes[1, 0].set_title(
    "Fare Before Standardization"
)

axes[1, 0].set_xlabel("Fare")
axes[1, 0].set_ylabel("Frequency")


# Fare after
axes[1, 1].hist(
    standardized_data["fare"],
    bins=30,
    edgecolor="black"
)

axes[1, 1].set_title(
    "Fare After Standardization"
)

axes[1, 1].set_xlabel("Standardized Fare")
axes[1, 1].set_ylabel("Frequency")


plt.tight_layout()

plt.savefig(
    CHART_PATH,
    dpi=150
)

plt.close()


print("\nComparison chart saved to:")
print(CHART_PATH)


# ---------------------------------------------------------
# Save standardized data
# ---------------------------------------------------------
STANDARDIZED_DATA_PATH = os.path.join(
    DATA_DIR,
    "titanic_age_fare_standardized.csv"
)

standardized_data.to_csv(
    STANDARDIZED_DATA_PATH,
    index=False
)

print("\nStandardized values saved to:")
print(STANDARDIZED_DATA_PATH)


# ---------------------------------------------------------
# Save report
# ---------------------------------------------------------
with open(
    REPORT_PATH,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "TASK 6 - EXPLORATORY STANDARDIZATION\n"
    )

    file.write(
        "=" * 60 + "\n\n"
    )

    file.write(
        "Features standardized: age, fare\n\n"
    )

    file.write(
        "BEFORE STANDARDIZATION\n"
    )

    file.write(
        "-" * 60 + "\n"
    )

    for column in features:
        file.write(
            f"{column}: "
            f"mean={before_mean[column]:.6f}, "
            f"std={before_std[column]:.6f}\n"
        )

    file.write("\n")

    file.write(
        "AFTER STANDARDIZATION\n"
    )

    file.write(
        "-" * 60 + "\n"
    )

    for column in features:
        file.write(
            f"{column}: "
            f"mean={after_mean[column]:.6f}, "
            f"std={after_std[column]:.6f}\n"
        )

    file.write("\n")

    file.write(
        "INTERPRETATION\n"
    )

    file.write(
        "-" * 60 + "\n"
    )

    file.write(
        "Standardization transforms each selected numeric "
        "feature so that its mean is approximately 0 and its "
        "standard deviation is approximately 1. The original "
        "distribution shape is retained while the feature scale "
        "is changed. This makes Age and Fare directly comparable "
        "in models that are sensitive to feature scale.\n"
    )


print("\nTask 6 report saved to:")
print(REPORT_PATH)


# ---------------------------------------------------------
# Preview
# ---------------------------------------------------------
print("\n" + "-" * 70)
print("STANDARDIZED DATA PREVIEW")
print("-" * 70)

print(
    standardized_data.head()
)