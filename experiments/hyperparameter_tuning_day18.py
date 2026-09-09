from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    cross_val_score,
    GridSearchCV
)
data=load_breast_cancer()


X=data.data
y=data.target
model=RandomForestClassifier(
      random_state=42
)
param_grid ={
    "n_estimators":[50,100,200],
    "max_depth":[3,5,10,None]
}

grid_search=GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=5,
    scoring="accuracy"
)

grid_search.fit(X,y)

print("=== BEST HYPERPARAMETERS ===")
print("Best Parameters:",grid_search.best_params_)
print("Best CV Accuracy:",grid_search.best_score_)

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


param_grid ={
    "n_estimators":[50,100,200],
    "max_depth":[3,5,10,None]
}

grid_search_top20=GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=5,
    scoring="accuracy"
)

grid_search_top20.fit(X_top20,y)

print("=== TOP20 + GRID SEARCH ===")
print("Best Parameters:",grid_search_top20.best_params_)
print("Best CV Accuracy:",grid_search_top20.best_score_)
