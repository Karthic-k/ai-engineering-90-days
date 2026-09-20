from src.loader import load_dataset
from src.analyzer import run_analysis
from src.ml import (
    detect_ml_features,
    detect_problem_type,
    train_classification_model,
    train_regression_model,
    compare_classification_models,
    compare_regression_models
)
import pandas as pd

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

comparison = compare_classification_models(
    students,
    target_column="passed"
)

print("\n=== MODEL COMPARISON ===")

for model, metrics in comparison["models"].items():
    print(
        f"{model}: "
        f"Mean Accuracy = {metrics['mean_accuracy']:.2%}, "
        f"Std = {metrics['std_accuracy']:.2%}"
    )

print("\nBest Model:", comparison["best_model"])

salary = pd.read_csv("data/salary_ml.csv")

comparison = compare_regression_models(
    salary,
    target_column="salary"
)

print("\n=== REGRESSION MODEL COMPARISON ===")

for model, metrics in comparison["models"].items():
    print(
        f"{model}: "
        f"MAE = {metrics['mean_mae']:.2f}, "
        f"RMSE = {metrics['mean_rmse']:.2f}, "
        f"R² = {metrics['mean_r2']:.2f}"
    )

print("\nBest Model:", comparison["best_model"])