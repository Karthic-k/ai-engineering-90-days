from src.loader import load_dataset
from src.analyzer import run_analysis
from src.ml import train_classification_model,detect_problem_type,train_regression_model

df = load_dataset("data/employee_data.csv")

result = run_analysis(df)

students = load_dataset("data/students_ml.csv")

result = train_classification_model(
    students,
    target_column="passed"
)

print("\n=== CLASSIFICATION MODEL ===")
print("Precision:", result["precision"])
print("Recall:", result["recall"])
print("F1 Score:", result["f1"])
print("Confusion Matrix:")
print(result["confusion_matrix"])
print("=== Other Features ===")
print(result)
problem_type = detect_problem_type(
    students,
    "passed"
)

print("\n=== PROBLEM TYPE ===")
print(problem_type)

salary_data = load_dataset("data/salary_ml.csv")

regression_result = train_regression_model(
    salary_data,
    target_column="salary"
)

print("\n=== REGRESSION MODEL ===")
print("Model:", type(regression_result["model"]).__name__)
print("MAE:", regression_result["mae"])
print("MSE:", regression_result["mse"])
print("RMSE:", regression_result["rmse"])
print("R2 Score:", regression_result["r2"])

