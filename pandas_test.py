import pandas as pd

import matplotlib.pyplot as plt
# ============================================================
# 1. LOAD DATASET
# ============================================================

students = pd.read_csv("students.csv")

print("\n===== ORIGINAL DATASET =====")
print(students)


# ============================================================
# 2. BASIC DATASET INFORMATION
# ============================================================

print("\n===== DATASET INFORMATION =====")

# Display first 5 rows
print("\nFirst 5 rows:")
print(students.head())

# Display number of rows and columns
print("\nDataset shape:")
print(students.shape)


# ============================================================
# 3. CHECK FOR DUPLICATES
# ============================================================

print("\n===== DUPLICATE CHECK =====")

duplicate_count = students.duplicated().sum()

print("Number of duplicate rows:", duplicate_count)

if duplicate_count > 0:
    print("\nDuplicate rows:")
    print(students[students.duplicated()])

    # Remove duplicate rows
    students = students.drop_duplicates()

    print("\nDuplicates removed successfully.")


# ============================================================
# 4. CONVERT MARKS TO NUMERIC
# ============================================================

print("\n===== DATA TYPE CLEANING =====")

# Convert marks to numbers.
# Invalid values such as "abc" become NaN.
students["marks"] = pd.to_numeric(
    students["marks"],
    errors="coerce"
)

print("\nData types after conversion:")
print(students.dtypes)


# ============================================================
# 5. CHECK FOR MISSING VALUES
# ============================================================

print("\n===== MISSING VALUE CHECK =====")

print(students.isnull().sum())


# ============================================================
# 6. CALCULATE MEAN AND MEDIAN
# ============================================================

print("\n===== STATISTICS BEFORE IMPUTATION =====")

mean_marks = students["marks"].mean()
median_marks = students["marks"].median()

print("Mean:", mean_marks)
print("Median:", median_marks)


# ============================================================
# 7. HANDLE MISSING VALUES
# ============================================================

print("\n===== HANDLING MISSING VALUES =====")

# Fill missing marks using the median
students["marks"] = students["marks"].fillna(median_marks)

print("Missing values after imputation:")
print(students.isnull().sum())


# ============================================================
# 8. CREATE GRADES
# ============================================================

def get_grade(marks):
    """
    Convert a student's marks into a grade.
    """

    if marks >= 90:
        return "A"

    elif marks >= 80:
        return "B"

    elif marks >= 70:
        return "C"

    elif marks >= 60:
        return "D"

    else:
        return "F"


students["grade"] = students["marks"].apply(get_grade)


# ============================================================
# 9. FINAL CLEAN DATASET
# ============================================================

print("\n===== CLEAN DATASET =====")
print(students)


# ============================================================
# 10. BASIC STATISTICS
# ============================================================

print("\n===== BASIC STATISTICS =====")

print("Average Marks:", students["marks"].mean())
print("Highest Marks:", students["marks"].max())
print("Lowest Marks:", students["marks"].min())


# ============================================================
# 11. FILTER STUDENTS
# ============================================================

print("\n===== STUDENTS WITH MARKS >= 70 =====")

high_scorers = students[students["marks"] >= 70]

print(high_scorers)


# ============================================================
# 12. SORT STUDENTS
# ============================================================

print("\n===== STUDENTS RANKED BY MARKS =====")

ranked_students = students.sort_values(
    "marks",
    ascending=False
)

print(ranked_students)


# ============================================================
# 13. STATISTICAL SUMMARY
# ============================================================

print("\n===== STATISTICAL SUMMARY =====")

print(students.describe())


# ============================================================
# 14. OUTLIER DETECTION USING IQR
# ============================================================

print("\n===== OUTLIER DETECTION =====")

Q1 = students["marks"].quantile(0.25)
Q3 = students["marks"].quantile(0.75)

IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

print("Q1:", Q1)
print("Q3:", Q3)
print("IQR:", IQR)
print("Lower Bound:", lower_bound)
print("Upper Bound:", upper_bound)


# ============================================================
# 15. FIND POTENTIAL OUTLIERS
# ============================================================

potential_outliers = students[
    (students["marks"] < lower_bound) |
    (students["marks"] > upper_bound)
]

print("\nPotential outliers:")

if potential_outliers.empty:
    print("No potential outliers found.")
else:
    print(potential_outliers)

print("\n===== ILOC EXAMPLE =====")

print("First student:")
print(students.iloc[0])

print("\nThird student:")
print(students.iloc[2])

print("\nfirst 5 student:")
print(students.iloc[0:5])

print(students.iloc[0,0])

print(students.iloc[0,1])

print(students.iloc[0:5,0:2])

print("\n===== LOC EXAMPLE =====")

print(students.loc[2])

print(students.loc[:,"name"])

print(students.loc[0:4,["name","marks"]])

print("\n===== MARKS BETWEEN 70 AND 90 =====")

students_between_70_and_90 = students[
    (students["marks"] >= 70) & (students["marks"] <= 90)
]
print(students_between_70_and_90)

print("\n===== MARKS BELOW 50 AND ABOVE 90 =====")
students_below_50_or_above_90 = students[
    (students["marks"] < 50) | (students["marks"] > 90)
]
print(students_below_50_or_above_90)

print("\n===GRADE DISTRIBUTION===")

print(students["grade"].value_counts())

print("\n===GRADE PERCENTAGE===")

grade_percentage=students["grade"].value_counts(normalize=True)*100

print(grade_percentage)

print("/n===Pass/Fail Analysis===")

passed=students[students["marks"]>=60]
failed=students[students["marks"]<60]

print("passed:",len(passed))
print("failed:",len(failed))

pass_percentage=len(passed)/len(students)*100

print("Pass Percentage:",pass_percentage)

print("\n===Above Average Analysis===")

above_average=students[students["marks"]>students["marks"].mean()]
print("Above Average:",len(above_average))

print("\n===Top performaing students===")
top_performing_students=students[students["marks"]==students["marks"].max()]
print(top_performing_students)

print("\n===Top performing student using idxmax===")
top_index=students["marks"].idxmax()
top_student=students.loc[top_index]

print(top_student)

print("\n===Bottom performing students===")
bottom_index=students["marks"].idxmin()
bottom_student=students.loc[bottom_index]
print(bottom_student)

print("\n=== Average Marks by Grade===")

average=students.groupby("grade")["marks"].mean()

print(average)

print("\n=== Count of Students by Grade===")
count=students.groupby("grade")["marks"].count()
print(count)

print("\n === Grade Statistics ===")

grade_Stats = students.groupby("grade")["marks"].agg(
    ["count","mean","min","max"]
)

print(grade_Stats)

# ============================================================
# 16. GRADE DISTRIBUTION VISUALIZATION
# ============================================================

grade_counts=students["grade"].value_counts()

plt.bar(grade_counts.index,grade_counts.values)

plt.title("Number of Students by Grade")
plt.xlabel("Grades")
plt.ylabel("Number of Students")

plt.show()

# ============================================================
# 17. MARKS DISTRIBUTION
# ============================================================

plt.hist(students["marks"],bins=10)

plt.title("Distribution of Marks")
plt.xlabel("Marks")
plt.ylabel("Number of Students")

plt.show()

# ============================================================
# 18. STUDENT MARKS VISUALIZATION
# ============================================================

plt.bar(students["name"],students["marks"])

plt.title("Students marks")

plt.xlabel("Students")

plt.ylabel("Marks")

plt.xticks(rotation=45)

plt.tight_layout()

plt.show()