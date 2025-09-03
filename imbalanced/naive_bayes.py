# naive_bayes.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 1. Load dataset
data = pd.read_csv(r'C:\Users\Rahul Patel\Downloads\greendestination (1) (1).csv')

# 2. Quick overview
print("Data Shape:", data.shape)
print(data.head())
print(data.info())
print(data.describe())
print("Missing values:\n", data.isnull().sum())

# 3. Handle missing values
data['Age'] = data['Age'].ffill()   # forward fill for Age

# 4. Convert categorical columns
data['Attrition'] = data['Attrition'].map({'Yes': 1, 'No': 0})
data['OverTime'] = data['OverTime'].map({'Yes': 1, 'No': 0})

# 5. Check attrition rate
attrition_rate = data['Attrition'].mean() * 100
print(f"\nAttrition Rate: {attrition_rate:.2f}%")

# EDA (same plots as other scripts)
sns.countplot(x='Attrition', data=data, palette="Set2")
plt.xticks([0, 1], ['No', 'Yes'])
plt.title("Attrition Distribution")
plt.show()

# (boxplots omitted for brevity — same as logistic script if you want)

# -------------------
# 🤖 Model Building (Gaussian Naive Bayes)
# -------------------

X = data[['Age', 'YearsAtCompany', 'MonthlyIncome', 'JobSatisfaction', 'WorkLifeBalance', 'OverTime']]
y = data['Attrition']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

pipe = Pipeline([('scaler', StandardScaler()), ('gnb', GaussianNB())])
pipe.fit(X_train, y_train)

y_pred = pipe.predict(X_test)

# -------------------
# 📈 Model Evaluation
# -------------------

print("\nModel Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(4, 3))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=['No Attrition', 'Attrition'],
            yticklabels=['No Attrition', 'Attrition'])
plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.title("Confusion Matrix - Naive Bayes")
plt.show()

# Final prediction
new_employee = pd.DataFrame({
    'Age': [30],
    'YearsAtCompany': [2],
    'MonthlyIncome': [4000],
    'JobSatisfaction': [3],
    'WorkLifeBalance': [2],
    'OverTime': [1]
})
pred = pipe.predict(new_employee)[0]
proba = pipe.predict_proba(new_employee)[0][1]
print("\nFinal Prediction for new employee:",
      "Attrition = Yes" if pred==1 else "Attrition = No",
      f"(Probability of leaving: {proba:.2f})")
