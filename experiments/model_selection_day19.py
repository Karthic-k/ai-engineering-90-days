from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    cross_val_score,
    GridSearchCV
)


data=load_breast_cancer()
X=data.data
y=data.target

models={
    "Logistic Regression":LogisticRegression(max_iter=10000),
    "Decision Tree":DecisionTreeClassifier(
        max_depth=5,
        random_state=42
    ),
    "Random Forest":RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
}

print("====Models====")

for name,model in models.items():
    print(name)

print("\n====CROSS-VALIDATION RESULT====")
for name,model in models.items():

    scores=cross_val_score(
        model,
        X,
        y,
        cv=5,
        scoring="accuracy"
    )

    print(f"{name}")
    print(f"Scores: {scores}")
    print(f"Mean Accuracy:{scores.mean():4f}")
    print(f"std:{scores.std():4f}")

print("\n===TUNED RANDOM FOREST===")    
param_grid={
    "n_estimators":[50,100,200],
    "max_depth":[3,5,10,None]
}

grid_search=GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring="accuracy"
)

grid_search.fit(X,y)

print("Best Parameters:",grid_search.best_params_)
print("Best CV Accuracy:",grid_search.best_score_)

print("\n=== TOP 20 + TUNED RANDOM FOREST ===")

importance_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

importance_model.fit(X, y)
feature_importance = list(
    zip(data.feature_names, importance_model.feature_importances_)
)

feature_importance.sort(
    key=lambda x: x[1],
    reverse=True
)

top_20_indices=[
    list(data.feature_names).index(feature)
    for feature,score in feature_importance[:20]
]

X_top20=X[:,top_20_indices]

grid_search_top20=GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring="accuracy"
)

grid_search_top20.fit(X_top20,y)

print("Best Parameters:",grid_search_top20.best_params_)
print("Best CV Accuracy:",grid_search_top20.best_score_)

print("\n=== FINAL MODEL COMPARISON ===")

models_final = {
    "Logistic Regression": LogisticRegression(max_iter=10000),
    "Decision Tree": DecisionTreeClassifier(
        max_depth=5,
        random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ),
    "Tuned Random Forest": grid_search.best_estimator_,
    "Top 20 Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
}

for name, model in models_final.items():

    if name == "Top 20 Random Forest":
        scores = cross_val_score(
            model,
            X_top20,
            y,
            cv=5,
            scoring="accuracy"
        )
    else:
        scores = cross_val_score(
            model,
            X,
            y,
            cv=5,
            scoring="accuracy"
        )

    print(
        f"{name}: "
        f"Mean = {scores.mean():.4f}, "
        f"Std = {scores.std():.4f}"
    )