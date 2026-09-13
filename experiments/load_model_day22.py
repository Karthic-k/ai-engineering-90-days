import joblib
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score

model=joblib.load("breast_cancer_model.pkl")
data=load_breast_cancer()
sample_index=100
sample=data.data[sample_index].reshape(1,-1)

prediction = model.predict(sample)

print("Sample Index:",sample_index)
print("Prediction:",prediction)
print("Actual:", data.target[sample_index])


samples = data.data[350:355]

predictions = model.predict(samples)

print("Predictions:", predictions)
print("Actual:", data.target[350:355])