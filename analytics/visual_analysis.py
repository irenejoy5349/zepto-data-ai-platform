import os
import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------
DATA_DIR = "analytics"
CLEANED_DATA_PATH = os.path.join(
    DATA_DIR,
    "titanic_cleaned.csv"
)

os.makedirs(DATA_DIR, exist_ok=True)


# ---------------------------------------------------------
# Load the SAME cleaned dataset
# ---------------------------------------------------------
df = pd.read_csv(CLEANED_DATA_PATH)

print("=" * 70)
print("TASK 5: VISUAL EXPLORATORY ANALYSIS")
print("=" * 70)

print(f"Rows loaded : {len(df)}")
print(f"Columns     : {len(df.columns)}")


# =========================================================
# CHART 1: Survival Count by Sex
# =========================================================

survival_sex = (
    df.groupby(
        "sex",
        observed=True
    )["survived"]
    .mean()
    .mul(100)
)

plt.figure(figsize=(8, 5))

survival_sex.plot(
    kind="bar"
)

plt.xlabel("Sex")
plt.ylabel("Survival Rate (%)")
plt.title("Survival Rate by Sex")
plt.xticks(rotation=0)
plt.tight_layout()

chart1_path = os.path.join(
    DATA_DIR,
    "chart1_survival_by_sex.png"
)

plt.savefig(
    chart1_path,
    dpi=150
)

plt.close()

print(f"\nChart 1 saved: {chart1_path}")


# =========================================================
# CHART 2: Survival Rate by Passenger Class
# =========================================================

survival_class = (
    df.groupby(
        "pclass",
        observed=True
    )["survived"]
    .mean()
    .mul(100)
)

plt.figure(figsize=(8, 5))

survival_class.plot(
    kind="bar"
)

plt.xlabel("Passenger Class")
plt.ylabel("Survival Rate (%)")
plt.title("Survival Rate by Passenger Class")
plt.xticks(rotation=0)
plt.tight_layout()

chart2_path = os.path.join(
    DATA_DIR,
    "chart2_survival_by_pclass.png"
)

plt.savefig(
    chart2_path,
    dpi=150
)

plt.close()

print(f"Chart 2 saved: {chart2_path}")


# =========================================================
# CHART 3: Survival Rate by Sex and Passenger Class
# =========================================================

survival_sex_class = (
    df.groupby(
        ["sex", "pclass"],
        observed=True
    )["survived"]
    .mean()
    .mul(100)
    .unstack()
)

plt.figure(figsize=(9, 5))

survival_sex_class.plot(
    kind="bar",
    ax=plt.gca()
)

plt.xlabel("Sex")
plt.ylabel("Survival Rate (%)")
plt.title("Survival Rate by Sex and Passenger Class")
plt.xticks(rotation=0)
plt.legend(title="Passenger Class")
plt.tight_layout()

chart3_path = os.path.join(
    DATA_DIR,
    "chart3_survival_sex_pclass.png"
)

plt.savefig(
    chart3_path,
    dpi=150
)

plt.close()

print(f"Chart 3 saved: {chart3_path}")


# =========================================================
# CHART 4: Fare Distribution by Survival
# =========================================================

plt.figure(figsize=(8, 5))

df.boxplot(
    column="fare",
    by="survived"
)

plt.xlabel("Survival Status")
plt.ylabel("Fare")
plt.title("Fare Distribution by Survival Status")
plt.suptitle("")

plt.xticks(
    [1, 2],
    ["Did Not Survive", "Survived"]
)

plt.tight_layout()

chart4_path = os.path.join(
    DATA_DIR,
    "chart4_fare_by_survival.png"
)

plt.savefig(
    chart4_path,
    dpi=150
)

plt.close()

print(f"Chart 4 saved: {chart4_path}")


# =========================================================
# CHART 5: Age Distribution by Survival
# Extra chart for stronger EDA
# =========================================================

plt.figure(figsize=(8, 5))

df.boxplot(
    column="age",
    by="survived"
)

plt.xlabel("Survival Status")
plt.ylabel("Age")
plt.title("Age Distribution by Survival Status")
plt.suptitle("")

plt.xticks(
    [1, 2],
    ["Did Not Survive", "Survived"]
)

plt.tight_layout()

chart5_path = os.path.join(
    DATA_DIR,
    "chart5_age_by_survival.png"
)

plt.savefig(
    chart5_path,
    dpi=150
)

plt.close()

print(f"Chart 5 saved: {chart5_path}")


# =========================================================
# Save interpretation report
# =========================================================

REPORT_PATH = os.path.join(
    DATA_DIR,
    "task5_visual_interpretations.md"
)

female_rate = survival_sex["female"]
male_rate = survival_sex["male"]

class_1_rate = survival_class[1]
class_2_rate = survival_class[2]
class_3_rate = survival_class[3]

fare_survivor_median = (
    df.loc[
        df["survived"] == 1,
        "fare"
    ].median()
)

fare_non_survivor_median = (
    df.loc[
        df["survived"] == 0,
        "fare"
    ].median()
)

age_survivor_median = (
    df.loc[
        df["survived"] == 1,
        "age"
    ].median()
)

age_non_survivor_median = (
    df.loc[
        df["survived"] == 0,
        "age"
    ].median()
)


with open(
    REPORT_PATH,
    "w",
    encoding="utf-8"
) as file:

    file.write("# Task 5 – Visual Exploratory Analysis\n\n")

    file.write("## Chart 1 – Survival Rate by Sex\n\n")
    file.write(
        f"The survival rate for female passengers was "
        f"{female_rate:.2f}%, compared with {male_rate:.2f}% "
        f"for male passengers. This indicates a strong difference "
        f"in survival outcomes between the two groups. Sex appears "
        f"to be an important variable associated with survival.\n\n"
    )

    file.write("## Chart 2 – Survival Rate by Passenger Class\n\n")
    file.write(
        f"Passengers in first class had a survival rate of "
        f"{class_1_rate:.2f}%, while second-class passengers had "
        f"{class_2_rate:.2f}% and third-class passengers had "
        f"{class_3_rate:.2f}%. Survival decreased as passenger "
        f"class moved from first to third class, suggesting that "
        f"passenger class was strongly associated with survival.\n\n"
    )

    file.write(
        "## Chart 3 – Survival Rate by Sex and Passenger Class\n\n"
    )
    file.write(
        "Combining sex and passenger class reveals a stronger "
        "survival pattern than either variable alone. Female "
        "passengers generally had higher survival rates within "
        "each class, while third-class passengers had lower "
        "survival rates. This suggests that the interaction "
        "between sex and class is important for understanding "
        "survival outcomes.\n\n"
    )

    file.write(
        "## Chart 4 – Fare Distribution by Survival Status\n\n"
    )
    file.write(
        f"The median fare for survivors was {fare_survivor_median:.2f}, "
        f"compared with {fare_non_survivor_median:.2f} for passengers "
        f"who did not survive. The higher median fare among survivors "
        f"suggests that fare level was associated with survival. "
        f"The distribution also contains high-fare outliers.\n\n"
    )

    file.write(
        "## Chart 5 – Age Distribution by Survival Status\n\n"
    )
    file.write(
        f"The median age among survivors was {age_survivor_median:.2f}, "
        f"while the median age among non-survivors was "
        f"{age_non_survivor_median:.2f}. The age distributions "
        f"overlap substantially, so age alone does not provide as "
        f"clear a separation as sex or passenger class. Age may "
        f"still contribute useful information when combined with "
        f"other variables in a predictive model.\n"
    )


print("\n" + "=" * 70)
print("TASK 5 COMPLETE")
print("=" * 70)

print("Interpretation report saved to:")
print(REPORT_PATH)