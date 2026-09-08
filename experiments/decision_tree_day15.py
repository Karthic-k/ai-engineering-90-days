import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

data = {
    "hours_studied": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "passed":        [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
}

students = pd.DataFrame(data)

X = students[["hours_studied"]]
y = students["passed"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

model = DecisionTreeClassifier(
    max_depth=10,
    random_state=42
)


model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("Predictions:")
print(predictions)

print("Actual:")
print(y_test.values)

accuracy = accuracy_score(y_test, predictions)

print("Decision Tree Accuracy:", accuracy)

plt.figure(figsize=(8,5))

plot_tree(
    model,
    feature_names=["hours_studied"],
    class_names=["fail","pass"],
    filled=True
)

plt.show()

train_predictions = model.predict(X_train)
test_predictions = model.predict(X_test)

train_accuracy = accuracy_score(y_train, train_predictions)
test_accuracy = accuracy_score(y_test, test_predictions)

print("Training Accuracy:", train_accuracy)
print("Testing Accuracy:", test_accuracy)

random_forest_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
random_forest_model.fit(X_train,y_train)

random_forest_predictions = random_forest_model.predict(X_test)

print("Random Forest Predictions:")
print(random_forest_predictions)

print("Actual:")
print(y_test.values)

rf_accuracy = accuracy_score(y_test, random_forest_predictions)
rf_precision = precision_score(y_test, random_forest_predictions)
rf_recall = recall_score(y_test, random_forest_predictions)
rf_f1 = f1_score(y_test, random_forest_predictions)

print("\nRandom Forest:")
print("Accuracy:", rf_accuracy)
print("Precision:", rf_precision)
print("Recall:", rf_recall)
print("F1 Score:", rf_f1)