from src.loader import load_dataset
from src.analyzer import run_analysis

df = load_dataset("data/employee_data.csv")

result = run_analysis(df)

print(result)