from sklearn.datasets import load_breast_cancer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import joblib
from sklearn.model_selection import cross_val_score


data=load_breast_cancer()

X=data.data
y=data.target

pipeline=Pipeline([
    ("scalar",StandardScaler()),
    ("model",LogisticRegression(max_iter=10000))
])

pipeline.fit(X,y)

print("Pipeline Trained Successfully")

sample=X[100].reshape(1,-1)

prediction = pipeline.predict(sample)

print("prediction:",prediction)
print("Actual:",y[100])

joblib.dump(pipeline,"breast_cancer_pipeline.pkl")

print("Pipeline saved successfuly")

scores=cross_val_score(
    pipeline,
    X,
    y,
    cv=5,
    scoring="accuracy"
)

print("CV scores:",scores)
print("Mean Accuracy",scores.mean())
print("Std:",scores.std())