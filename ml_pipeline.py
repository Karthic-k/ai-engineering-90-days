import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

employees=pd.read_csv("employee_data.csv")
X=employees[["experience","age","department"]]
y=employees["salary"]

preprossesor=ColumnTransformer(
    transformers=[
        (
            "department",
            OneHotEncoder(drop="first",handle_unknown="ignore"),
            ["department"]
        )
    ],
    remainder="passthrough"
)

model=Pipeline(
    steps=[
        ("preprocessor",preprossesor),
        ("regressor",LinearRegression())
    ]
)
scores = cross_val_score(
    model,
    X,
    y,
    cv=5,
    scoring="neg_mean_absolute_error"
)

mae = -scores

print("MAE scores:", mae)
print("Average MAE:", mae.mean())
print("Standard Deviation:", mae.std())