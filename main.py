import pandas as pd
from src.analyzer import (
    average_salary_by_department,
    highest_paid_employee,
    check_missing_values,
    calculate_correlation,
    detect_salary_outliers,
    department_statistics
)
def main():

    employees = pd.read_csv("employee_data.csv")
    
    print("\n=== AVERAGE SALARY BY DEPARTMENT ===")
    print(average_salary_by_department(employees))

    print("\n=== HIGHEST PAID EMPLOYEE ===")
    print(highest_paid_employee(employees))

    print("\n=== MISSING VALUES ===")
    print(check_missing_values(employees))

    print("\n=== EXPERIENCE VS SALARY CORRELATION ===")
    print(calculate_correlation(
         employees,
        "experience",
        "salary"
    ))

    print("\n=== SALARY OUTLIERS ===")
    print(detect_salary_outliers(employees))

    print("\n=== DEPARTMENT STATISTICS ===")
    print(department_statistics(employees))

if __name__ == "__main__":
    main()