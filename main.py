import pandas as pd
from src.analyzer import (
    average_salary_by_department,
    highest_paid_employee,
    check_missing_values,
    calculate_correlation,
    detect_salary_outliers
)
def main():
    employees=pd.read_csv("employee_data.csv")

    result=average_salary_by_department(employees)
    highest_paid=highest_paid_employee(employees)
    missing_values=check_missing_values(employees)
    correlation = calculate_correlation(
        employees,
        "experience",
        "salary"
    )
    outliers=detect_salary_outliers(employees)

    print("\n ===Average Salary of Each Department===")
    print(result)

    print("\n ===Highest Paid Employee===")
    print(highest_paid)

    print("\n ===Missing Values===")
    print(missing_values)

    print("\n=== EXPERIENCE VS SALARY CORRELATION ===")
    print(correlation)

    print("\n=== SALARY OUTLIERS ===")
    print(outliers)

if __name__ == "__main__":
    main()