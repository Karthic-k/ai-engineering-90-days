import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

employees=pd.read_csv("employee_data.csv")

X = employees[["experience"]]


y=employees["salary"]

model=LinearRegression()
scores = cross_val_score(
  model,
  X,
  y,
  cv=5,
  scoring="neg_mean_absolute_error"
)
mae_scores = -scores

print("MAE scores:",mae_scores)

print("Average MAE:", mae_scores.mean())

print("Stadard deviation:",mae_scores.std())