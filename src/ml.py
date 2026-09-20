from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
def detect_ml_features(df):
    numeric_columns = df.select_dtypes(include="number").columns.tolist()

    categorical_columns = df.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    return {
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns
    }
def detect_problem_type(df, target_column):
    target = df[target_column]

    unique_values = target.nunique()

    if target.dtype == "object":
        return "classification"

    if unique_values <= 10 and target.value_counts().min() >= 2:
        return "classification"

    return "regression"

def train_classification_model(df, target_column):
    data = df.copy()

    X = data.drop(columns=[target_column])
    y = data[target_column]

        # Validate target for classification
    if y.nunique() < 2:
        raise ValueError(
            "Target column must contain at least 2 classes."
        )

    if y.value_counts().min() < 2:
        raise ValueError(
            "Each target class must have at least 2 samples. "
            "Please select a suitable classification target."
        )
    X = X.select_dtypes(include="number")

    if y.dtype == "object":
        encoder = LabelEncoder()
        y = encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = LogisticRegression(max_iter=10000)

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, zero_division=0)
    recall = recall_score(y_test, predictions, zero_division=0)
    f1 = f1_score(y_test, predictions, zero_division=0)
    matrix = confusion_matrix(y_test, predictions)

    return {
        "model": model,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": matrix
    }
def compare_classification_models(df, target_column):
    data = df.copy()

    X = data.drop(columns=[target_column])
    y = data[target_column]

    X = X.select_dtypes(include="number")

    if y.dtype == "object":
        encoder = LabelEncoder()
        y = encoder.fit_transform(y)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=10000),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )
    }

    results = {}

    for name, model in models.items():
        scores = cross_val_score(
            model,
            X,
            y,
            cv=5,
            scoring="accuracy"
        )

        results[name] = {
            "mean_accuracy": scores.mean(),
            "std_accuracy": scores.std()
        }
    best_model = max(
    results,
    key=lambda name: results[name]["mean_accuracy"]
    )

    return {
        "models": results,
        "best_model": best_model
    }

def compare_regression_models(df, target_column):
    data = df.copy()

    X = data.drop(columns=[target_column])
    y = data[target_column]

    X = X.select_dtypes(include="number")

    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
        "Random Forest": RandomForestRegressor(
            n_estimators=100,
            random_state=42
        )
    }

    results = {}

    for name, model in models.items():

        mae_scores = -cross_val_score(
            model,
            X,
            y,
            cv=5,
            scoring="neg_mean_absolute_error"
        )

        rmse_scores = np.sqrt(
            -cross_val_score(
                model,
                X,
                y,
                cv=5,
                scoring="neg_mean_squared_error"
            )
        )

        r2_scores = cross_val_score(
            model,
            X,
            y,
            cv=5,
            scoring="r2"
        )

        results[name] = {
            "mean_mae": mae_scores.mean(),
            "std_mae": mae_scores.std(),
            "mean_rmse": rmse_scores.mean(),
            "std_rmse": rmse_scores.std(),
            "mean_r2": r2_scores.mean(),
            "std_r2": r2_scores.std()
        }

    best_model = max(
        results,
        key=lambda name: results[name]["mean_r2"]
    )

    return {
        "models": results,
        "best_model": best_model
    }
def train_regression_model(df, target_column):
    data = df.copy()

    X = data.drop(columns=[target_column])
    y = data[target_column]

    X = X.select_dtypes(include="number")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    model = LinearRegression()

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)

    return {
        "model": model,
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "r2": r2
    }