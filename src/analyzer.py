def average_salary_by_department(employees):
    return employees.groupby("department")["salary"].mean()

def highest_paid_employee(employees):
    index=employees["salary"].idxmax()
    return employees.loc[index]

def check_missing_values(employees):
    return employees.isnull().sum()

def calculate_correlation(employees, column1, column2):
    return employees[column1].corr(employees[column2])

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

def department_statistics(employees):
    return employees.groupby("department")["salary"].agg(
        ["count","mean","min","max"]
    )

def highest_average_salary_department(employees):
    return employees.groupby("department")["salary"].mean().idxmax()


def salary_range_by_department(employees):
    return (
        employees.groupby("department")["salary"].max()
        - employees.groupby("department")["salary"].min()
    )


def employees_above_average_salary(employees):
    return employees[employees["salary"] > employees["salary"].mean()]

def analyze_dataset(df):
    return {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "column_names": df.columns.tolist(),
        "data_types": df.dtypes.astype(str).to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "duplicate_rows": int(df.duplicated().sum())
    }

def statistical_summary(df):
    numeric_columns = df.select_dtypes(include="number")

    if numeric_columns.empty:
        return {}

    return numeric_columns.describe().to_dict()

def categorical_summary(df):
    categorical_columns = df.select_dtypes(include="object")

    summary = {}

    for column in categorical_columns.columns:
        value_counts = df[column].value_counts()

        if value_counts.max() > 1:
            summary[column] = value_counts.to_dict()

    return summary

def data_quality(df):
    missing_values = df.isnull().sum().to_dict()
    duplicate_rows = int(df.duplicated().sum())

    return {
        "missing_values": missing_values,
        "duplicate_rows": duplicate_rows,
        "outliers": detect_outliers(df)
    }
def detect_outliers(df):
    numeric_columns = df.select_dtypes(include="number")

    outliers = {}

    for column in numeric_columns.columns:
        q1 = numeric_columns[column].quantile(0.25)
        q3 = numeric_columns[column].quantile(0.75)

        iqr = q3 - q1

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        count = int(
            ((numeric_columns[column] < lower_bound) |
             (numeric_columns[column] > upper_bound)).sum()
        )

        outliers[column] = count

    return outliers

def run_analysis(df):
    return {
        "overview": analyze_dataset(df),
        "statistics": statistical_summary(df),
        "categorical": categorical_summary(df),
        "data_quality": data_quality(df)
    }

