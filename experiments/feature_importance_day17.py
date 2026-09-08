from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score


data=load_breast_cancer()

X=data.data
y=data.target

model=RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X,y)

# Create feature-importance pairs
feature_importance = list(zip(data.feature_names, model.feature_importances_))

# Sort from highest to lowest importance
feature_importance.sort(key=lambda x: x[1], reverse=True)

# Display sorted results
print("=== FEATURE IMPORTANCE ===")

for feature, score in feature_importance:
    print(f"{feature}: {score:.4f}")

from sklearn.model_selection import cross_val_score

# Select top 10 features
top_features = [feature for feature, score in feature_importance[:10]]

print("\n=== TOP 10 FEATURES ===")
print(top_features)

# Find column indexes of top features
top_indices = [
    list(data.feature_names).index(feature)
    for feature in top_features
]

X_top = X[:, top_indices]

# Evaluate Random Forest using only top 10 features
scores = cross_val_score(
    model,
    X_top,
    y,
    cv=5,
    scoring="accuracy"
)

print("\n=== TOP 10 FEATURES MODEL ===")
print("CV Accuracy:", scores)
print("Average Accuracy:", scores.mean())


print("\n=== FEATURE SELECTION COMPARISON ===")

for n in [5, 10, 15, 20, 30]:

    # Select top n features
    top_indices = [
        list(data.feature_names).index(feature)
        for feature, score in feature_importance[:n]
    ]

    X_selected = X[:, top_indices]

    # Cross-validation
    scores = cross_val_score(
        model,
        X_selected,
        y,
        cv=5,
        scoring="accuracy"
    )

    print(f"Top {n} features: {scores.mean():.4f}")


print("\n=== FEATURE SELECTION COMPARISON ===")

for n in [5, 10, 15, 20, 30]:

    # Select top n features
    top_indices = [
        list(data.feature_names).index(feature)
        for feature, score in feature_importance[:n]
    ]

    X_selected = X[:, top_indices]

    # Cross-validation
    scores = cross_val_score(
        model,
        X_selected,
        y,
        cv=5,
        scoring="accuracy"
    )

    print(
        f"Top {n} features: "
        f"Mean Accuracy = {scores.mean():.4f}, "
        f"Std = {scores.std():.4f}"
    )