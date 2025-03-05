# Employee Attrition Analysis

## Overview
This project analyzes employee attrition trends using a dataset from Green Destinations. The goal is to determine attrition rates and identify key influencing factors such as age, years at the company, and monthly income. The analysis uses Python with pandas, NumPy, matplotlib, seaborn, and scikit-learn.

## Features
- Loads and cleans the dataset
- Performs exploratory data analysis (EDA) with visualizations
- Implements a logistic regression model to predict attrition
- Evaluates model performance with accuracy, confusion matrix, and classification report

---

## Dataset
The dataset includes the following key columns:
- `Age`: Employee's age
- `YearsAtCompany`: Number of years the employee has worked at the company
- `MonthlyIncome`: Employee's monthly salary
- `Attrition`: Whether the employee has left the company (`Yes` → 1, `No` → 0)

---

## Prerequisites
Ensure you have Python installed along with the required libraries:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

---

## Usage
1. Download the dataset and place it in the appropriate directory.
2. Run the script:
   ```bash
   python employee-attrition.py
   ```
3. The script will:
   - Display dataset statistics
   - Show attrition distribution and boxplots
   - Train a logistic regression model
   - Print model accuracy and classification metrics

---

## Exploratory Data Analysis
### Attrition Distribution
This visualization shows the number of employees who stayed vs. those who left.

![Screenshot 2025-03-05 155028](https://github.com/user-attachments/assets/726cf9b8-5e7e-4748-baf9-b416fe3d9866)


### Age vs. Attrition
Examining how age affects employee attrition.

![Screenshot 2025-03-05 155041](https://github.com/user-attachments/assets/8449f097-dc2b-4a43-a8f4-2a40100114d3)


### Years at Company vs. Attrition
This visualization shows how long employees stayed before leaving.

![Screenshot 2025-03-05 155053](https://github.com/user-attachments/assets/9db9063c-d011-4842-9057-8d936bbc22da)


### Monthly Income vs. Attrition
Does salary impact attrition?

![Screenshot 2025-03-05 155102](https://github.com/user-attachments/assets/24246569-7f2c-4be1-a328-c8936f4427c4)

### Other output Images 

![Screenshot 2025-03-05 155154](https://github.com/user-attachments/assets/9015d4fe-c44e-4f50-acd4-ade17eadc39e)


![Screenshot 2025-03-05 155137](https://github.com/user-attachments/assets/8224c151-ec6e-426b-bc0f-3afd7de479d4)


![Screenshot 2025-03-05 155122](https://github.com/user-attachments/assets/d07049fe-8f2f-41b1-ac62-b45fa1cdc1b1)





---

## Model Performance
After training the logistic regression model, the results include:
- **Accuracy Score**
- **Confusion Matrix**
- **Classification Report**

Example output:
```bash
Accuracy: 78.5%
Confusion Matrix:
[[500  30]
 [ 90  80]]
Classification Report:
              precision    recall  f1-score   support
           0       0.85      0.94      0.89       530
           1       0.73      0.47      0.57       170
```

---

## Conclusion
This project provides insights into employee attrition using exploratory data analysis and machine learning. Future improvements can include:
- Adding more features such as job role, department, and work-life balance
- Using advanced models like decision trees or neural networks

---

## Author
**Rahul Patel**

---
