def average_salary_by_department(employees):
    return employees.groupby("department")["salary"].mean()

def highest_paid_employee(employees):
    index=employees["salary"].idxmax()
    return employees.loc[index]

def check_missing_values(employees):
    return employees.isnull().sum()

def calculate_correlation(employees,colomn1,colomn2):
    return employees[colomn1].corr(employees[colomn2])

def detect_salary_outliers(employees):
    q1=employees["salary"].quantile(0.25)
    q3=employees["salary"].quantile(0.75)

    IQR=q3-q1

    lower_bound=q1-1.5*IQR
    upper_bound=q3+1.5*IQR

    return employees[
        (employees["salary"]<lower_bound)|
        (employees["salary"]>upper_bound)
    ]