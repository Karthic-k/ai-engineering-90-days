from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
data=load_breast_cancer()

print("Features:",data.data.shape)
print("Labels:",data.target.shape)
print("Feature Names:",data.feature_names)

X=data.data
y=data.target

print("X shape:",X.shape)
print("y shape:",y.shape)

X_train,X_test,y_train,y_test=train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])


logistic_model=LogisticRegression(
    max_iter=10000,
    random_state=42
)

logistic_model.fit(X_train,y_train)

logistic_predictions=logistic_model.predict(X_test)

print("Predictions:",logistic_predictions)

accuracy=accuracy_score(y_test,logistic_predictions)
precision=precision_score(y_test,logistic_predictions)
recall=recall_score(y_test,logistic_predictions)
f1=f1_score(y_test,logistic_predictions)

print("Accuracy:",accuracy)
print("Precision:",precision)
print("Recall:",recall)
print("F1 Score:",f1)

tree_model=DecisionTreeClassifier(
    max_depth=5,
    random_state=42
)

tree_model.fit(X_train,y_train)

tree_predictions=tree_model.predict(X_test)

print("\nDecision Tree:")
print("Accuracy:", accuracy_score(y_test, tree_predictions))
print("Precision:", precision_score(y_test, tree_predictions))
print("Recall:", recall_score(y_test, tree_predictions))
print("F1 Score:", f1_score(y_test, tree_predictions))

random_forest_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

random_forest_model.fit(X_train, y_train)

random_forest_predictions = random_forest_model.predict(X_test)

print("\nRandom Forest:")
print("Accuracy:", accuracy_score(y_test, random_forest_predictions))
print("Precision:", precision_score(y_test, random_forest_predictions))
print("Recall:", recall_score(y_test, random_forest_predictions))
print("F1 Score:", f1_score(y_test, random_forest_predictions))


logistic_Regression_cv=cross_val_score(
    logistic_model,
    X,
    y,
    cv=5,
    scoring="accuracy"
)

tree_model_cv=cross_val_score(
    tree_model,
    X,
    y,
    cv=5,
    scoring="accuracy"
)

random_forest_model_cv=cross_val_score(
    random_forest_model,
    X,
    y,
    cv=5,
    scoring="accuracy"
)

print("Logistic Regression:", logistic_Regression_cv)
print("Average:", logistic_Regression_cv.mean())
print("Std:", logistic_Regression_cv.std())

print("\nDecision Tree:", tree_model_cv)
print("Average:", tree_model_cv.mean())
print("Std:", tree_model_cv.std())

print("\nRandom Forest:", random_forest_model_cv)
print("Average:", random_forest_model_cv.mean())
print("Std:", random_forest_model_cv.std())