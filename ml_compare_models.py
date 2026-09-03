import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score


def evaluate_model(model, X, y):

    scores = cross_val_score(
        model,
        X,
        y,
        cv=5,
        scoring="neg_mean_absolute_error"
    )

    mae = -scores

    return mae.mean(), mae.std()

employees = pd.read_csv("employee_data.csv")

y = employees["salary"]


# Model 1
X1 = employees[["experience"]]

model1 = LinearRegression()


avg_mae1, std_mae1 = evaluate_model(model1, X1, y)

print("Model 1")
print("Average MAE:", avg_mae1)
print("Standard Deviation:", std_mae1)
# Model 2
X2 = pd.get_dummies(
    employees[["experience", "age", "department"]],
    columns=["department"],
    drop_first=True,
    dtype=int
)

model2 = LinearRegression()

avg_mae2, std_mae2 = evaluate_model(model2, X2, y)


print("\nModel 2")
print("Average MAE:", avg_mae2)
print("Standard Deviation:", std_mae2)


results=pd.DataFrame({
    "Model":["Model 1","Model 2"],
    "Features":[
        "Experience",
        "Experience, Age, Department"
    ],
    "Average MAE":[avg_mae1,avg_mae2],
    "Standard Deviation":[std_mae1,std_mae2]
})

print("\nComparison of Models:")
print(results.to_string(index=False))

best_model = results.loc[results["Average MAE"].idxmin()]

print("\nBest Model:", best_model["Model"])
print("Features:", best_model["Features"])
print("Average MAE:", best_model["Average MAE"])
print("Standard Deviation:", best_model["Standard Deviation"])