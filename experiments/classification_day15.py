import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)




data={
    "hours_studied":[1,2,3,4,5,6,7,8,9,10],
    "passed":[0,0,0,0,1,1,1,1,1,1]
}

students=pd.DataFrame(data)

print(students)

X = students[["hours_studied"]]
y = students["passed"]

print("X:")
print(X)

print("\ny:")
print(y)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nX_train:")
print(X_train)

print("\ny_train:")
print(y_train)

print("\nX_test:")
print(X_test)

print("\ny_test:")
print(y_test)

model = LogisticRegression()

model.fit(X_train, y_train)

predictions=model.predict(X_test)

print("\nPredictions:")
print(predictions)

print("\nActual values:")
print(y_test.values)

accuracy=accuracy_score(y_test,predictions)

print("\nAccuracy:", accuracy)

cm=confusion_matrix(y_test,predictions)

print("\nConfusion Matrix:")
print(cm)

new_students = pd.DataFrame({
    "hours_studied": [4, 7, 10]
})

new_predictions = model.predict(new_students)

print("New predictions:")
print(new_predictions)


precision = precision_score(y_test,predictions)
recall=recall_score(y_test,predictions)
f1=f1_score(y_test,predictions)

print("\nPrecision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print(classification_report(y_test,predictions))