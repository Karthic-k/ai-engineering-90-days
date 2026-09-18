from src.loader import load_dataset
from src.analyzer import run_analysis
from src.ml import train_classification_model

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