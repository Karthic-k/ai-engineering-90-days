import pandas as pd

employees=pd.read_csv("employee_data.csv")

correlation=employees[["experience","age","salary"]].corr()

print(correlation)