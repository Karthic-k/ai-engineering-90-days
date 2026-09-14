import joblib
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score

pipeline = joblib.load("Breast_cancer_pipeline.pkl")

data=load_breast_cancer()

sample=data.data[350].reshape(1,-1)

prediction=pipeline.predict(sample)

print("prediction:",prediction)

print("Actual:",data.target[350])

samples = data.data[350:355]

predictions = pipeline.predict(samples)

print("Predictions:", predictions)
print("Actual:", data.target[350:355])

accuracy = accuracy_score(data.target[350:355], predictions)

print("Accuracy:", accuracy)

