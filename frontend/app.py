import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import streamlit as st
import pandas as pd

from src.loader import load_dataset
from src.analyzer import run_analysis
from src.ml import (
    detect_ml_features,
    detect_problem_type,
    train_classification_model,
    train_regression_model
)
from src.visualizer import (
    plot_numeric_distribution,
    plot_categorical_numeric,
    plot_correlation_heatmap
)

st.set_page_config(
    page_title="AI Dataset Analyzer",
    page_icon="📊",
    layout="wide"
)

st.title("📊 AI Dataset Analyzer")
st.write("Upload a CSV or Excel file to analyze your dataset.")

uploaded_file = st.file_uploader(
    "Choose a dataset",
    type=["csv", "xlsx", "xls"]
)

if uploaded_file is not None:

    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.success(f"Successfully loaded: {uploaded_file.name}")

    st.subheader("Dataset Preview")
    st.dataframe(df)

    result = run_analysis(df)

    st.subheader("Dataset Overview")

    overview = result["overview"]

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Rows", overview["rows"])
        st.metric("Columns", overview["columns"])

    with col2:
        st.metric("Duplicate Rows", overview["duplicate_rows"])
        st.metric(
            "Total Missing Values",
            sum(overview["missing_values"].values())
        )
    st.subheader("Statistical Summary")

    statistics = result["statistics"]

    st.dataframe(
        pd.DataFrame(statistics).T
    )

    st.subheader("Data Quality")

    quality = result["data_quality"]

    st.write("Missing Values")
    st.json(quality["missing_values"])

    st.metric("Duplicate Rows", quality["duplicate_rows"])

    st.write("Outliers")
    st.json(quality["outliers"])

    st.subheader("Categorical Analysis")

    categorical = result["categorical"]

    for column, values in categorical.items():
        st.write(f"**{column}**")
        st.dataframe(
            pd.DataFrame(
                list(values.items()),
                columns=["Value", "Count"]
            )
        )
    st.subheader("Visualizations")

    st.write("### Numeric Distribution")

    numeric_columns = df.select_dtypes(include="number").columns.tolist()

    selected_column = st.selectbox(
        "Select a numeric column",
        numeric_columns
    )

    fig = plot_numeric_distribution(df, selected_column)
    st.pyplot(fig)

    st.write("### Categorical vs Numeric")

    categorical_columns = df.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    numeric_columns = df.select_dtypes(
        include="number"
    ).columns.tolist()

    if categorical_columns and numeric_columns:

        categorical_column = st.selectbox(
            "Select categorical column",
            categorical_columns
        )

        numeric_column = st.selectbox(
            "Select numeric column",
            numeric_columns
        )

        fig = plot_categorical_numeric(
            df,
            categorical_column,
            numeric_column
        )

        st.pyplot(fig)

    else:
        st.info(
            "This dataset does not contain both categorical "
            "and numeric columns for this visualization."
        )

    st.write("### Correlation Heatmap")
    fig = plot_correlation_heatmap(df)
    st.pyplot(fig)

    st.subheader("🤖 ML Analysis")

    ml_features = detect_ml_features(df)

    st.write("### Numeric Features")
    st.write(ml_features["numeric_columns"])

    st.write("### Categorical Features")
    st.write(ml_features["categorical_columns"])

    target_column = st.selectbox(
        "Select target column",
        df.columns
    )

    st.write("Selected target:", target_column)

    if st.button("Run ML Model"):
        try:
            problem_type = detect_problem_type(
                df,
                target_column
            )

            st.write("### Problem Type")
            st.write(problem_type.capitalize())

            if problem_type == "classification":

                result = train_classification_model(
                    df,
                    target_column
                )

                st.success("Classification model trained successfully!")

                st.write("### Model")
                st.write(type(result["model"]).__name__)

                col1, col2, col3, col4 = st.columns(4)

                col1.metric(
                    "Accuracy",
                    f"{result['accuracy']:.2%}"
                )

                col2.metric(
                    "Precision",
                    f"{result['precision']:.2%}"
                )

                col3.metric(
                    "Recall",
                    f"{result['recall']:.2%}"
                )

                col4.metric(
                    "F1 Score",
                    f"{result['f1']:.2%}"
                )

                st.write("### Confusion Matrix")
                st.write(result["confusion_matrix"])

            else:

                result = train_regression_model(
                    df,
                    target_column
                )

                st.success("Regression model trained successfully!")

                st.write("### Model")
                st.write(type(result["model"]).__name__)

                col1, col2, col3, col4 = st.columns(4)

                col1.metric(
                    "MAE",
                    f"{result['mae']:.2f}"
                )

                col2.metric(
                    "MSE",
                    f"{result['mse']:.2f}"
                )

                col3.metric(
                    "RMSE",
                    f"{result['rmse']:.2f}"
                )

                col4.metric(
                    "R² Score",
                    f"{result['r2']:.2f}"
                )

        except ValueError as e:
            st.error(str(e))