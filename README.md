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
|
├── src/
│   └── analyzer.py
|
├── employee_data.csv
├── main.py
├── numpy_test.py
├── pandas_test.py
├── real_data_test.py
├── students.csv
├── requirements.txt
├── .gitignore
└── README.md

## How to Run

### 1. Clone the repository

git clone https://github.com/Karthic-k/ai-engineering-90-days.git

### 2. Navigate to the project

cd ai-engineering-90-days/01-ai-dataset-analyzer

### 3. Install dependencies

pip install -r requirements.txt

### 4. Run the analyzer

python main.py

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

Days completed: 1–6