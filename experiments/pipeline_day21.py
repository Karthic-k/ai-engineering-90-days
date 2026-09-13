from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

# Load dataset
data = load_breast_cancer()

X = data.data
y = data.target

# Create Logistic Regression model
model = LogisticRegression(max_iter=10000)

# Cross-validation
scores = cross_val_score(
    model,
    X,
    y,
    cv=5,
    scoring="accuracy"
)

print("=== LOGISTIC REGRESSION WITHOUT SCALING ===")
print("Scores:", scores)
print("Mean Accuracy:", scores.mean())
print("Std:", scores.std())

# Create pipeline with scaling + Logistic Regression
scaled_model = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=10000))
])

# Cross-validation
scaled_scores = cross_val_score(
    scaled_model,
    X,
    y,
    cv=5,
    scoring="accuracy"
)

print("\n=== LOGISTIC REGRESSION WITH SCALING ===")
print("Scores:", scaled_scores)
print("Mean Accuracy:", scaled_scores.mean())
print("Std:", scaled_scores.std())

rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf_scores = cross_val_score(
    rf_model,
    X,
    y,
    cv=5,
    scoring="accuracy"
)

print("\n=== RANDOM FOREST WITHOUT SCALING ===")
print("Mean Accuracy:", rf_scores.mean())
print("Std:", rf_scores.std())

scaled_rf_model = Pipeline([
    ("scaler", StandardScaler()),
    ("model", RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ))
])

scaled_rf_scores = cross_val_score(
    scaled_rf_model,
    X,
    y,
    cv=5,
    scoring="accuracy"
)

print("\n=== RANDOM FOREST WITH SCALING ===")
print("Mean Accuracy:", scaled_rf_scores.mean())
print("Std:", scaled_rf_scores.std())

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=10000))
])

param_grid = {
    "model__C": [0.01, 0.1, 1, 10, 100]
}

grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring="accuracy"
)

grid_search.fit(X, y)

print("Best Parameters:", grid_search.best_params_)
print("Best CV Accuracy:", grid_search.best_score_)

print("\n=== FINAL COMPARISON ===")
print("Logistic Regression:", scores.mean())
print("Scaled Logistic Regression:", scaled_scores.mean())
print("Random Forest:", rf_scores.mean())
print("Tuned Logistic Regression:", grid_search.best_score_)