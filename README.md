Employee Attrition Analysis
This project analyzes employee attrition trends at Green Destinations. The goal is to determine factors influencing employee attrition, such as age, years at the company, and monthly income.

Table of Contents
Introduction
Dataset
Installation
Usage
Features
Exploratory Data Analysis (EDA)
Machine Learning Model
Results
Visualizations
Contributing
License
Repository
Introduction
Employee attrition is a key challenge for companies. This project utilizes machine learning to analyze and predict employee attrition based on important factors.

Dataset
The dataset contains employee details with the following key columns:

Age: Age of the employee.
YearsAtCompany: Number of years the employee has worked at the company.
MonthlyIncome: Employee’s monthly income.
Attrition: Target variable (Yes = 1, No = 0).
Installation
Ensure you have Python installed, then install the required dependencies:

bash
Copy
Edit
pip install pandas numpy matplotlib seaborn scikit-learn
Usage
Run the script using:

bash
Copy
Edit
python employee_attrition_analysis.py
Features
Data Cleaning

Handles missing values using forward fill.
Exploratory Data Analysis (EDA)

Generates summary statistics.
Uses Seaborn for visualizing trends.
Machine Learning Model

Logistic Regression to predict attrition.
Evaluates performance using Accuracy, Confusion Matrix, and Classification Report.
Exploratory Data Analysis (EDA)
The script performs the following analysis:

Displays basic information and summary statistics.
Checks for missing values and fills them.
Computes the attrition rate.
Visualizes key insights using box plots and count plots.
Machine Learning Model
Splits data into training (70%) and testing (30%).
Trains a Logistic Regression model.
Predicts employee attrition based on Age, Years at Company, and Monthly Income.
Evaluates the model using:
Accuracy Score
Confusion Matrix
Classification Report
Results
Attrition rate percentage is displayed.
Logistic Regression predicts employee attrition based on key factors.
Model performance is measured using evaluation metrics.
Visualizations
The project includes the following plots:

Attrition Distribution: Bar chart showing attrition counts.
Age vs. Attrition: Box plot to analyze age differences.
Years at Company vs. Attrition: Box plot showing attrition trends over tenure.
Monthly Income vs. Attrition: Box plot to visualize income differences.
Contributing
If you want to contribute, fork the repository and submit a pull request.
