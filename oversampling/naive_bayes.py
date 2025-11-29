# ==========================================
# 🤖 Naive Bayes Classifier with Random OverSampling
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from imblearn.over_sampling import RandomOverSampler   # 🔁 Using OverSampler

# -------------------
# 1️⃣ Load the dataset
# -------------------
data = pd.read_csv(r'C:\Users\Rahul Patel\Downloads\greendestination (1) (1).csv')

print("✅ Data Loaded Successfully")
print("Data Shape:", data.shape)
print(data.head(), "\n")
print(data.info(), "\n")
print("Missing Values:\n", data.isnull().sum(), "\n")

# -------------------
# 2️⃣ Handle missing values
# -------------------
data['Age'] = data['Age'].ffill()  # Forward fill missing Age values

# -------------------
# 3️⃣ Encode categorical columns
# -------------------
data['Attrition'] = data['Attrition'].map({'Yes': 1, 'No': 0})
data['OverTime'] = data['OverTime'].map({'Yes': 1, 'No': 0})

# -------------------
# 4️⃣ Check class imbalance before oversampling
# -------------------
print("\nClass Distribution before Oversampling:")
print(data['Attrition'].value_counts())

plt.figure(figsize=(5,4))
sns.countplot(x='Attrition', data=data, palette='coolwarm')
plt.title("Attrition Class Distribution (Before Oversampling)")
plt.xticks([0,1], ['No', 'Yes'])
plt.show()

# -------------------
# 5️⃣ Feature selection
# -------------------
X = data[['Age', 'YearsAtCompany', 'MonthlyIncome', 'JobSatisfaction', 'WorkLifeBalance', 'OverTime']]
y = data['Attrition']

# -------------------
# 6️⃣ Train-test split
# -------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# -------------------
# 7️⃣ Apply Random OverSampling
# -------------------
ros = RandomOverSampler(random_state=42)
X_train_res, y_train_res = ros.fit_resample(X_train, y_train)

print("\nClass Distribution after OverSampling:")
print(y_train_res.value_counts())

plt.figure(figsize=(5,4))
sns.countplot(x=y_train_res, palette='Set2')
plt.title("Attrition Class Distribution (After OverSampling)")
plt.xticks([0,1], ['No', 'Yes'])
plt.show()

# -------------------
# 8️⃣ Feature Scaling (Optional but helpful for NB)
# -------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_res)
X_test_scaled = scaler.transform(X_test)

# -------------------
# 9️⃣ Model Building - Gaussian Naive Bayes
# -------------------
nb_model = GaussianNB()
nb_model.fit(X_train_scaled, y_train_res)

# -------------------
# 🔟 Predictions
# -------------------
y_pred = nb_model.predict(X_test_scaled)

# -------------------
# 1️⃣1️⃣ Model Evaluation
# -------------------
print("\n📈 Model Evaluation (Naive Bayes - Oversampled Data):")
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:\n", cm)
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix Heatmap
plt.figure(figsize=(4,3))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['No Attrition', 'Attrition'],
            yticklabels=['No Attrition', 'Attrition'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix Heatmap (Naive Bayes - Oversampled Data)")
plt.show()

# -------------------
# 1️⃣2️⃣ Predict for a new employee
# -------------------
new_employee = pd.DataFrame({
    'Age': [30],
    'YearsAtCompany': [2],
    'MonthlyIncome': [4000],
    'JobSatisfaction': [3],
    'WorkLifeBalance': [2],
    'OverTime': [1]   # 1 = Yes, 0 = No
})

new_employee_scaled = scaler.transform(new_employee)
prediction = nb_model.predict(new_employee_scaled)[0]
probability = nb_model.predict_proba(new_employee_scaled)[0][1]

print("\nFinal Prediction for new employee (Naive Bayes - Oversampled Model):")
print("Attrition =", "Yes" if prediction == 1 else "No")
print(f"Probability of leaving: {probability:.2f}")
