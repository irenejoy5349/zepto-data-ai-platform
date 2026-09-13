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

os.makedirs(
    DATA_DIR,
    exist_ok=True
)


# ---------------------------------------------------------
# Load the SAME cleaned dataset
# ---------------------------------------------------------
df = pd.read_csv(
    CLEANED_DATA_PATH
)

print("=" * 70)
print("TASK 5: VISUAL EXPLORATORY ANALYSIS")
print("=" * 70)

print(
    f"Rows loaded : {len(df)}"
)

print(
    f"Columns     : {len(df.columns)}"
)


# =========================================================
# CHART 1: Survival Rate by Sex
# =========================================================

survival_sex = (
    df.groupby(
        "sex",
        observed=True
    )["survived"]
    .mean()
    .mul(100)
)

plt.figure(
    figsize=(8, 5)
)

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

print(
    f"\nChart 1 saved: {chart1_path}"
)


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

plt.figure(
    figsize=(8, 5)
)

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

print(
    f"Chart 2 saved: {chart2_path}"
)


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

plt.figure(
    figsize=(9, 5)
)

survival_sex_class.plot(
    kind="bar",
    ax=plt.gca()
)

plt.xlabel("Sex")
plt.ylabel("Survival Rate (%)")
plt.title(
    "Survival Rate by Sex and Passenger Class"
)

plt.xticks(
    rotation=0
)

plt.legend(
    title="Passenger Class"
)

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

print(
    f"Chart 3 saved: {chart3_path}"
)


# =========================================================
# CHART 4: Fare Distribution by Survival
# =========================================================

plt.figure(
    figsize=(8, 5)
)

df.boxplot(
    column="fare",
    by="survived"
)

plt.xlabel("Survival Status")
plt.ylabel("Fare")
plt.title(
    "Fare Distribution by Survival Status"
)

plt.suptitle("")

plt.xticks(
    [1, 2],
    [
        "Did Not Survive",
        "Survived"
    ]
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

print(
    f"Chart 4 saved: {chart4_path}"
)


# =========================================================
# CHART 5: Age Distribution by Survival
# =========================================================

plt.figure(
    figsize=(8, 5)
)

df.boxplot(
    column="age",
    by="survived"
)

plt.xlabel("Survival Status")
plt.ylabel("Age")
plt.title(
    "Age Distribution by Survival Status"
)

plt.suptitle("")

plt.xticks(
    [1, 2],
    [
        "Did Not Survive",
        "Survived"
    ]
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

print(
    f"Chart 5 saved: {chart5_path}"
)


# =========================================================
# CHART 6: Age vs Fare by Survival Status
# Genuinely multivariate:
# age + fare + survival
# =========================================================

plt.figure(
    figsize=(9, 6)
)

survivors = df[
    df["survived"] == 1
]

non_survivors = df[
    df["survived"] == 0
]

plt.scatter(
    non_survivors["age"],
    non_survivors["fare"],
    alpha=0.5,
    marker="o",
    label="Did Not Survive"
)

plt.scatter(
    survivors["age"],
    survivors["fare"],
    alpha=0.5,
    marker="x",
    label="Survived"
)

plt.xlabel("Age")
plt.ylabel("Fare")
plt.title(
    "Age vs Fare by Survival Status"
)

plt.legend()

plt.tight_layout()

chart6_path = os.path.join(
    DATA_DIR,
    "chart6_age_fare_survival.png"
)

plt.savefig(
    chart6_path,
    dpi=150
)

plt.close()

print(
    f"Chart 6 saved: {chart6_path}"
)


# =========================================================
# CHART 7: Fare by Passenger Class and Survival Status
# Genuinely multivariate:
# passenger class + fare + survival
# =========================================================

fare_class_survival = {}

for survival_value, label in [
    (0, "Did Not Survive"),
    (1, "Survived")
]:

    fare_class_survival[label] = []

    for class_value in sorted(
        df["pclass"].dropna().unique()
    ):

        values = df.loc[
            (
                df["pclass"] == class_value
            )
            & (
                df["survived"] == survival_value
            ),
            "fare"
        ].dropna()

        fare_class_survival[label].append(
            values
        )


positions_non_survivors = [
    1,
    2,
    3
]

positions_survivors = [
    1.25,
    2.25,
    3.25
]

plt.figure(
    figsize=(10, 6)
)

plt.boxplot(
    fare_class_survival["Did Not Survive"],
    positions=positions_non_survivors,
    widths=0.2,
    patch_artist=False
)

plt.boxplot(
    fare_class_survival["Survived"],
    positions=positions_survivors,
    widths=0.2,
    patch_artist=False
)

plt.xticks(
    [
        1.125,
        2.125,
        3.125
    ],
    [
        "1st Class",
        "2nd Class",
        "3rd Class"
    ]
)

plt.xlabel("Passenger Class")
plt.ylabel("Fare")
plt.title(
    "Fare by Passenger Class and Survival Status"
)

plt.legend(
    [
        plt.Line2D(
            [],
            [],
            color="black",
            linewidth=1.5
        ),
        plt.Line2D(
            [],
            [],
            color="black",
            linewidth=1.5,
            linestyle="--"
        )
    ],
    [
        "Did Not Survive",
        "Survived"
    ]
)

plt.tight_layout()

chart7_path = os.path.join(
    DATA_DIR,
    "chart7_fare_class_survival.png"
)

plt.savefig(
    chart7_path,
    dpi=150
)

plt.close()

print(
    f"Chart 7 saved: {chart7_path}"
)


# =========================================================
# CHART 8: Age by Passenger Class and Survival Status
# Genuinely multivariate:
# passenger class + age + survival
# =========================================================

age_class_survival = {}

for survival_value, label in [
    (0, "Did Not Survive"),
    (1, "Survived")
]:

    age_class_survival[label] = []

    for class_value in sorted(
        df["pclass"].dropna().unique()
    ):

        values = df.loc[
            (
                df["pclass"] == class_value
            )
            & (
                df["survived"] == survival_value
            ),
            "age"
        ].dropna()

        age_class_survival[label].append(
            values
        )


plt.figure(
    figsize=(10, 6)
)

plt.boxplot(
    age_class_survival["Did Not Survive"],
    positions=positions_non_survivors,
    widths=0.2,
    patch_artist=False
)

plt.boxplot(
    age_class_survival["Survived"],
    positions=positions_survivors,
    widths=0.2,
    patch_artist=False
)

plt.xticks(
    [
        1.125,
        2.125,
        3.125
    ],
    [
        "1st Class",
        "2nd Class",
        "3rd Class"
    ]
)

plt.xlabel("Passenger Class")
plt.ylabel("Age")
plt.title(
    "Age by Passenger Class and Survival Status"
)

plt.legend(
    [
        plt.Line2D(
            [],
            [],
            color="black",
            linewidth=1.5
        ),
        plt.Line2D(
            [],
            [],
            color="black",
            linewidth=1.5,
            linestyle="--"
        )
    ],
    [
        "Did Not Survive",
        "Survived"
    ]
)

plt.tight_layout()

chart8_path = os.path.join(
    DATA_DIR,
    "chart8_age_class_survival.png"
)

plt.savefig(
    chart8_path,
    dpi=150
)

plt.close()

print(
    f"Chart 8 saved: {chart8_path}"
)


# =========================================================
# CHART 9: Sibling/Spouse vs Parent/Child by Survival
# Genuinely multivariate:
# sibsp + parch + survival
# =========================================================

plt.figure(
    figsize=(9, 6)
)

plt.scatter(
    non_survivors["sibsp"],
    non_survivors["parch"],
    alpha=0.5,
    marker="o",
    label="Did Not Survive"
)

plt.scatter(
    survivors["sibsp"],
    survivors["parch"],
    alpha=0.5,
    marker="x",
    label="Survived"
)

plt.xlabel(
    "Number of Siblings / Spouses (sibsp)"
)

plt.ylabel(
    "Number of Parents / Children (parch)"
)

plt.title(
    "Family Relationships and Survival Status"
)

plt.legend()

plt.tight_layout()

chart9_path = os.path.join(
    DATA_DIR,
    "chart9_family_survival.png"
)

plt.savefig(
    chart9_path,
    dpi=150
)

plt.close()

print(
    f"Chart 9 saved: {chart9_path}"
)


# =========================================================
# SAVE INTERPRETATION REPORT
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

    file.write(
        "# Task 5 – Visual Exploratory Analysis\n\n"
    )


    # -----------------------------------------------------
    # Chart 1 interpretation
    # -----------------------------------------------------

    file.write(
        "## Chart 1 – Survival Rate by Sex\n\n"
    )

    file.write(
        f"The survival rate for female passengers was "
        f"{female_rate:.2f}%, compared with "
        f"{male_rate:.2f}% for male passengers. "
        f"This indicates a strong difference in survival "
        f"outcomes between the two groups. Sex appears "
        f"to be an important variable associated with "
        f"survival.\n\n"
    )


    # -----------------------------------------------------
    # Chart 2 interpretation
    # -----------------------------------------------------

    file.write(
        "## Chart 2 – Survival Rate by Passenger Class\n\n"
    )

    file.write(
        f"Passengers in first class had a survival rate of "
        f"{class_1_rate:.2f}%, while second-class passengers "
        f"had {class_2_rate:.2f}% and third-class passengers "
        f"had {class_3_rate:.2f}%. Survival decreased as "
        f"passenger class moved from first to third class, "
        f"suggesting that passenger class was strongly "
        f"associated with survival.\n\n"
    )


    # -----------------------------------------------------
    # Chart 3 interpretation
    # -----------------------------------------------------

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


    # -----------------------------------------------------
    # Chart 4 interpretation
    # -----------------------------------------------------

    file.write(
        "## Chart 4 – Fare Distribution by Survival Status\n\n"
    )

    file.write(
        f"The median fare for survivors was "
        f"{fare_survivor_median:.2f}, compared with "
        f"{fare_non_survivor_median:.2f} for passengers "
        f"who did not survive. The higher median fare among "
        f"survivors suggests that fare level was associated "
        f"with survival. The distribution also contains "
        f"high-fare outliers.\n\n"
    )


    # -----------------------------------------------------
    # Chart 5 interpretation
    # -----------------------------------------------------

    file.write(
        "## Chart 5 – Age Distribution by Survival Status\n\n"
    )

    file.write(
        f"The median age among survivors was "
        f"{age_survivor_median:.2f}, while the median age "
        f"among non-survivors was "
        f"{age_non_survivor_median:.2f}. The age "
        f"distributions overlap substantially, so age alone "
        f"does not provide as clear a separation as sex or "
        f"passenger class. Age may still contribute useful "
        f"information when combined with other variables in "
        f"a predictive model.\n\n"
    )


    # -----------------------------------------------------
    # Chart 6 interpretation
    # -----------------------------------------------------

    file.write(
        "## Chart 6 – Age vs Fare by Survival Status\n\n"
    )

    file.write(
        "This chart combines age and fare while separating "
        "passengers by survival status. Survivors are more "
        "frequently observed among passengers with higher "
        "fares, while age shows substantial overlap between "
        "the two outcome groups. The combined view shows why "
        "multiple features can provide more information than "
        "examining age or fare separately.\n\n"
    )


    # -----------------------------------------------------
    # Chart 7 interpretation
    # -----------------------------------------------------

    file.write(
        "## Chart 7 – Fare by Passenger Class and Survival Status\n\n"
    )

    file.write(
        "Fare distributions differ substantially across "
        "passenger classes, with first-class passengers "
        "generally paying higher fares than second- and "
        "third-class passengers. Comparing survival status "
        "within each class provides a more detailed view than "
        "looking at fare or class alone. This suggests that "
        "fare and passenger class contain related but useful "
        "information for understanding survival outcomes.\n\n"
    )


    # -----------------------------------------------------
    # Chart 8 interpretation
    # -----------------------------------------------------

    file.write(
        "## Chart 8 – Age by Passenger Class and Survival Status\n\n"
    )

    file.write(
        "The age distributions vary across passenger classes "
        "and also differ between survivors and non-survivors. "
        "The class-wise comparison shows that survival patterns "
        "cannot be interpreted from age alone because passenger "
        "class changes the distribution. Combining age, class, "
        "and survival provides a more complete exploratory view.\n\n"
    )


    # -----------------------------------------------------
    # Chart 9 interpretation
    # -----------------------------------------------------

    file.write(
        "## Chart 9 – Family Relationships and Survival Status\n\n"
    )

    file.write(
        "The chart compares the number of siblings or spouses "
        "with the number of parents or children while separating "
        "survivors from non-survivors. Most passengers are "
        "concentrated at low family-count values, while larger "
        "family structures are less common. The overlap between "
        "survival groups indicates that these family variables "
        "alone are not sufficient to explain survival, but they "
        "can contribute useful information when combined with "
        "other features.\n\n"
    )


print(
    "\n" + "=" * 70
)

print(
    "TASK 5 COMPLETE"
)

print(
    "=" * 70
)

print(
    "Interpretation report saved to:"
)

print(
    REPORT_PATH
)
