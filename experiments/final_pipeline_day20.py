from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
from sklearn.metrics import confusion_matrix
# Load dataset

data = load_breast_cancer()

X = data.data
y = data.target

# Split before doing any feature selection
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("=== DATASET SPLIT ===")
print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])
print("Number of features:", X_train.shape[1])


importance_model=RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

importance_model.fit(X_train,y_train)

feature_importance=list(
    zip(data.feature_names,importance_model.feature_importances_)

)

feature_importance.sort(
    key=lambda x:x[1],
    reverse=True
)

print("\n=== TOP 20 FEATURES ===")

for feature, score in feature_importance[:20]:
    print(f"{feature}: {score:.4f}")

# Get the top 20 feature names
top_20_features = [
    feature
    for feature, score in feature_importance[:20]
]

# Get their column indexes
top_20_indices = [
    list(data.feature_names).index(feature)
    for feature in top_20_features
]

# Create training and testing data with only top 20 features
X_train_top20 = X_train[:, top_20_indices]
X_test_top20 = X_test[:, top_20_indices]

print("\n=== FEATURE SELECTION ===")
print("Original features:", X_train.shape[1])
print("Selected features:", X_train_top20.shape[1])

# Train model using selected features
final_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42
)

final_model.fit(X_train_top20, y_train)

# Predict on untouched test data
y_pred = final_model.predict(X_test_top20)

# Evaluate
print("\n=== FINAL MODEL TEST RESULTS ===")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)

print("\n=== CONFUSION MATRIX ===")
print(cm)