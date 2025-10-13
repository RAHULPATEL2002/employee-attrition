# ==========================
# 📘 Logistic Regression with Random UnderSampling
# ==========================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from imblearn.under_sampling import RandomUnderSampler

# -------------------
# 1️⃣ Load the dataset
# -------------------
data = pd.read_csv(r'C:\Users\Rahul Patel\Downloads\greendestination (1) (1).csv')

print("✅ Data Loaded Successfully")
print("Data Shape:", data.shape)
print(data.head(), "\n")
print(data.info(), "\n")
print("Missing values:\n", data.isnull().sum())

# -------------------
# 2️⃣ Handle missing values
# -------------------
data['Age'] = data['Age'].ffill()  # forward fill missing Age

# -------------------
# 3️⃣ Encode categorical columns
# -------------------
data['Attrition'] = data['Attrition'].map({'Yes': 1, 'No': 0})
data['OverTime'] = data['OverTime'].map({'Yes': 1, 'No': 0})

# -------------------
# 4️⃣ Check imbalance
# -------------------
print("\nClass Distribution before UnderSampling:")
print(data['Attrition'].value_counts())

plt.figure(figsize=(5,4))
sns.countplot(x='Attrition', data=data, palette="coolwarm")
plt.title("Attrition Class Distribution (Before Undersampling)")
plt.xticks([0,1], ['No', 'Yes'])
plt.show()

# -------------------
# 5️⃣ Feature Selection
# -------------------
X = data[['Age', 'YearsAtCompany', 'MonthlyIncome', 'JobSatisfaction', 'WorkLifeBalance', 'OverTime']]
y = data['Attrition']

# -------------------
# 6️⃣ Split before undersampling
# -------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# -------------------
# 7️⃣ Apply Random UnderSampling
# -------------------
rus = RandomUnderSampler(random_state=42)
X_train_res, y_train_res = rus.fit_resample(X_train, y_train)

print("\nClass Distribution after UnderSampling:")
print(y_train_res.value_counts())

plt.figure(figsize=(5,4))
sns.countplot(x=y_train_res, palette="Set2")
plt.title("Attrition Class Distribution (After Undersampling)")
plt.xticks([0,1], ['No', 'Yes'])
plt.show()

# -------------------
# 8️⃣ Model Building - Logistic Regression
# -------------------
model = LogisticRegression(max_iter=1000)
model.fit(X_train_res, y_train_res)

# -------------------
# 9️⃣ Predictions
# -------------------
y_pred = model.predict(X_test)

# -------------------
# 🔟 Model Evaluation
# -------------------
print("\n📈 Model Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Confusion matrix visualization
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(4,3))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=['No Attrition', 'Attrition'],
            yticklabels=['No Attrition', 'Attrition'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix Heatmap (Undersampled Data)")
plt.show()

# -------------------
# 1️⃣1️⃣ Final Prediction Example
# -------------------
new_employee = pd.DataFrame({
    'Age': [30],
    'YearsAtCompany': [2],
    'MonthlyIncome': [4000],
    'JobSatisfaction': [3],
    'WorkLifeBalance': [2],
    'OverTime': [1]   # 1 = Yes, 0 = No
})

prediction = model.predict(new_employee)[0]
prediction_proba = model.predict_proba(new_employee)[0][1]

print("\nFinal Prediction for new employee:")
print("Attrition =", "Yes" if prediction==1 else "No")
print(f"Probability of leaving: {prediction_proba:.2f}")
