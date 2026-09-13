from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
import joblib
data=load_breast_cancer()

X=data.data
y=data.target

model=RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42
    )
model.fit(X,y)

joblib.dump(model,"breast_cancer_model.pkl")

print("Model saved successfully")