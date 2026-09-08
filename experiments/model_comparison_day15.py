import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score

# Dataset
data = {
    "hours_studied": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "passed":        [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
}

students = pd.DataFrame(data)

# Features and target
X = students[["hours_studied"]]
y = students["passed"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Models
logistic_model = LogisticRegression()
tree_model = DecisionTreeClassifier(random_state=42)

# Train
logistic_model.fit(X_train, y_train)
tree_model.fit(X_train, y_train)

# Predictions
logistic_predictions = logistic_model.predict(X_test)
tree_predictions = tree_model.predict(X_test)

# Compare
models = {
    "Logistic Regression": logistic_predictions,
    "Decision Tree": tree_predictions
}

for name, predictions in models.items():

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)

    print("\n", name)
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)

    logistic_scores = cross_val_score(
    logistic_model,
    X,
    y,
    cv=5,
    scoring="accuracy"
)

tree_scores = cross_val_score(
    tree_model,
    X,
    y,
    cv=5,
    scoring="accuracy"
)

print("\nLogistic Regression CV Accuracy:")
print(logistic_scores)
print("Average:", logistic_scores.mean())

print("\nDecision Tree CV Accuracy:")
print(tree_scores)
print("Average:", tree_scores.mean())