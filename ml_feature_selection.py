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


# Model 1: Experience
X1 = employees[["experience"]]

model1 = LinearRegression()

avg_mae1, std_mae1 = evaluate_model(model1, X1, y)


# Model 2: Experience + Age
X2 = employees[["experience", "age"]]

model2 = LinearRegression()

avg_mae2, std_mae2 = evaluate_model(model2, X2, y)


# Model 3: Experience + Department
X3 = pd.get_dummies(
    employees[["experience", "department"]],
    columns=["department"],
    drop_first=True,
    dtype=int
)

model3 = LinearRegression()

avg_mae3, std_mae3 = evaluate_model(model3, X3, y)


# Model 4: Experience + Age + Department
X4 = pd.get_dummies(
    employees[["experience", "age", "department"]],
    columns=["department"],
    drop_first=True,
    dtype=int
)

model4 = LinearRegression()

avg_mae4, std_mae4 = evaluate_model(model4, X4, y)


# Comparison
results = pd.DataFrame({
    "Model": ["Model 1", "Model 2", "Model 3", "Model 4"],
    "Features": [
        "Experience",
        "Experience + Age",
        "Experience + Department",
        "Experience + Age + Department"
    ],
    "Average MAE": [
        avg_mae1,
        avg_mae2,
        avg_mae3,
        avg_mae4
    ],
    "Standard Deviation": [
        std_mae1,
        std_mae2,
        std_mae3,
        std_mae4
    ]
})

results["Average MAE"] = results["Average MAE"].round(2)
results["Standard Deviation"] = results["Standard Deviation"].round(2)

print("\nFeature Comparison:")
print(results.to_string(index=False))