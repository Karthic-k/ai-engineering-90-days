import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import streamlit as st
import pandas as pd

from src.loader import load_dataset
from src.analyzer import run_analysis

from src.visualizer import (
    plot_numeric_distribution,
    plot_department_salary,
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

    st.write("### Salary Distribution")
    fig = plot_numeric_distribution(df, "salary")
    st.pyplot(fig)

    st.write("### Average Salary by Department")
    fig = plot_department_salary(df)
    st.pyplot(fig)

    st.write("### Correlation Heatmap")
    fig = plot_correlation_heatmap(df)
    st.pyplot(fig)