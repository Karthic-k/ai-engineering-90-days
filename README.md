# AI Dataset Analyzer

A Python-based data analysis project that explores employee datasets using Pandas, NumPy, and Matplotlib.

The project performs data inspection, department analysis, salary analysis, correlation analysis, outlier detection, and data visualization.

## Features

- Dataset inspection
- Missing value detection
- Average salary analysis
- Highest-paid employee detection
- Department statistics
- Correlation analysis
- Salary outlier detection
- Data visualization
- Reusable analysis functions

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Git & GitHub

## Project Structure

01-ai-dataset-analyzer/
├── api/
│   ├── __init__.py
│   └── main.py
├── data/
│   ├── employee_data.csv
│   └── students.csv
├── experiments/
├── scripts/
│   ├── __init__.py
│   └── main.py
├── src/
│   ├── __init__.py
│   └── analyzer.py
├── tests/
├── breast_cancer_model.pkl
├── breast_cancer_pipeline.pkl
├── requirements.txt
└── README.md

## How to Run

### 1. Clone the repository

git clone https://github.com/Karthic-k/ai-engineering-90-days.git

### 2. Navigate to the project

cd ai-engineering-90-days/01-ai-dataset-analyzer

### 3. Install dependencies

pip install -r requirements.txt

### 4. Run the analyzer

python -m scripts.main

## Example Analysis

Using the included employee dataset, the analyzer produces the following results:

### Average Salary by Department

- Finance: ₹73,000
- HR: ₹45,333.33
- IT: ₹44,250

### Highest Paid Employee

- Name: Suresh
- Department: Finance
- Experience: 15 years
- Salary: ₹95,000

### Experience vs Salary

Correlation between experience and salary:

0.9965

This indicates a very strong positive linear relationship between experience and salary in this dataset.

### Missing Values

No missing values were found in the dataset.

### Salary Outliers

No salary outliers were detected using the IQR method.

## What I Learned

Through this project, I learned:

- Python fundamentals and functions
- NumPy arrays and numerical operations
- Pandas DataFrame manipulation
- Data inspection and cleaning
- GroupBy and aggregation
- Pivot tables
- Correlation analysis
- Outlier detection using IQR
- Data visualization with Matplotlib
- Writing reusable Python functions
- Modular programming
- Importing functions from modules
- Structuring a Python project
- Using Git and GitHub for version control

## Project Status

Completed as part of my 90-Day AI Engineering learning journey.

Days completed: 1–24

## ML Prediction API

This project includes a FastAPI endpoint for making predictions using the saved breast cancer ML pipeline.

### Run the API

From the project root:

```bash
uvicorn api.main:app --reload

The API will run at:

http://127.0.0.1:8000

API Documentation

FastAPI provides interactive documentation at:

http://127.0.0.1:8000/docs

Prediction Endpoint

POST /predict

The endpoint accepts the 30 breast cancer features as JSON and returns the model prediction.

Example response:

{
  "prediction": 0,
  "prediction_label": "malignant"
}
API Flow

Raw JSON → FastAPI → Pydantic Validation → Saved ML Pipeline → Prediction → JSON Response


