import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 1. LOAD DATASET
# ============================================================

employees = pd.read_csv("employee_data.csv")

print(employees)

# ============================================================
# 2. BASIC DATASET INSPECTION
# ============================================================

print("\n=== FIRST 5 ROWS ===")
print(employees.head())

print("\n=== SHAPE ===")
print(employees.shape)

print("\n=== COLUMNS ===")
print(employees.columns)

print("\n=== DATA TYPES ===")
print(employees.dtypes)

print("\n=== DATASET INFO ===")
print(employees.info())

print("\n=== STATISTICS ===")
print(employees.describe())

# ============================================================
# 3. MISSING VALUE ANALYSIS
# ============================================================

print("\n=== MISSING VALUES ===")
print(employees.isnull().sum())


# ============================================================
# 4. DEPARTMENT ANALYSIS
# ============================================================

print("\n=== EMPLOYEES BY DEPARTMENT ===")
print(employees["department"].value_counts())


# ============================================================
# 5. AVERAGE SALARY BY DEPARTMENT
# ============================================================

print("\n=== AVERAGE SALARY BY DEPARTMENT ===")
print(
    employees.groupby("department")["salary"].mean()
)

# ============================================================
# 6. HIGHEST AND LOWEST SALARY
# ============================================================
highest_paid=employees.loc[employees["salary"].idxmax()]
lowest_paid=employees.loc[employees["salary"].idxmin()]

print("\n=== HIGHEST PAID EMPLOYEE ===")
print(highest_paid)

print("\n=== LOWEST PAID EMPLOYEE ===")
print(lowest_paid)  

# ============================================================
# 7. EXPERIENCE VS SALARY CORRELATION
# ============================================================
correlation=employees["experience"].corr(employees["salary"])
print("\n=== EXPERIENCE VS SALARY CORRELATION ===")
print("Correlation:",correlation)

# ============================================================
# 8. EXPERIENCE VS SALARY SCATTER PLOT
# ============================================================
plt.scatter(
    employees["experience"],
    employees["salary"]
)

plt.title("Experience vs Salary")
plt.xlabel("Experience (years)") 
plt.ylabel("Salary")

plt.tight_layout()
plt.show()

# ============================================================
# 9. SALARY OUTLIER DETECTION
# ============================================================

q1=employees["salary"].quantile(0.25)
q3=employees["salary"].quantile(0.75)   
IQR=q3-q1
lower_bound=q1-1.5*IQR
upper_bound=q3+1.5*IQR

print("\n=== SALARY OUTLIER DETECTION ===")
print("Q1:",q1)
print("Q3:",q3)
print("IQR:",IQR)
print("Lower Bound:",lower_bound)
print("Upper Bound:",upper_bound)

outliers = employees[
    (employees["salary"] < lower_bound) |
    (employees["salary"] > upper_bound)
]
print("\n=== POTENTIAL OUTLIERS ===")
print(outliers)

# ============================================================
# 10. SALARY BY DEPARTMENT
# ============================================================

department_salary = employees.groupby("department")["salary"].mean()

print("\n=== AVERAGE SALARY BY DEPARTMENT ===")
print(department_salary)

plt.bar(
    department_salary.index,
    department_salary.values
)

plt.title("Average Salary by Department")
plt.xlabel("Department")
plt.ylabel("Average Salary")

plt.tight_layout()
plt.show()

print("highest age:",employees[employees["age"] == employees["age"].max()].iloc[0])
print("more than 5 years of experience:",employees[employees["experience"]>5])
print("average age of each department:",employees.groupby("department")["age"].mean())
print("highest salary of each department:",employees.groupby("department")["salary"].max())
print("correlation between age and salary:",employees["age"].corr(employees["salary"]))

plt.scatter(
    employees["age"],
    employees["salary"]
)

plt.title("age vs Salary")
plt.xlabel("Age (years)") 
plt.ylabel("Salary")

plt.tight_layout()
plt.show()
