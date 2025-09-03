import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
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

# -------------------
# 📊 Exploratory Data Analysis
# -------------------

# Countplot of attrition
sns.countplot(x='Attrition', data=data, palette="Set2")
plt.xticks([0, 1], ['No', 'Yes'])
plt.title("Attrition Distribution")
plt.show()

# Boxplots: Key features vs Attrition
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

sns.boxplot(x='Attrition', y='Age', data=data, ax=axs[0], palette="Set3")
axs[0].set_title("Age vs Attrition")

sns.boxplot(x='Attrition', y='YearsAtCompany', data=data, ax=axs[1], palette="Set3")
axs[1].set_title("Years at Company vs Attrition")

sns.boxplot(x='Attrition', y='MonthlyIncome', data=data, ax=axs[2], palette="Set3")
axs[2].set_title("Income vs Attrition")

plt.tight_layout()
plt.show()

# Boxplots: Additional Key features vs Attrition
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

sns.boxplot(x='Attrition', y='JobSatisfaction', data=data, ax=axs[0], palette="Set2")
axs[0].set_title("Job Satisfaction vs Attrition")

sns.boxplot(x='Attrition', y='WorkLifeBalance', data=data, ax=axs[1], palette="Set2")
axs[1].set_title("Work-Life Balance vs Attrition")

sns.boxplot(x='Attrition', y='OverTime', data=data, ax=axs[2], palette="Set2")
axs[2].set_title("OverTime vs Attrition")

plt.tight_layout()
plt.show()

# Correlation heatmap
plt.figure(figsize=(6, 4))
sns.heatmap(
    data[['Age','YearsAtCompany','MonthlyIncome','JobSatisfaction','WorkLifeBalance','OverTime','Attrition']].corr(), 
    annot=True, cmap="coolwarm"
)
plt.title("Correlation Heatmap")
plt.show()

# -------------------
# 🤖 Model Building
# -------------------

# Features & target (added new features)
X = data[['Age', 'YearsAtCompany', 'MonthlyIncome', 'JobSatisfaction', 'WorkLifeBalance', 'OverTime']]
y = data['Attrition']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Logistic Regression Model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# -------------------
# 📈 Model Evaluation
# -------------------

print("\nModel Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Confusion matrix heatmap
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(4, 3))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=['No Attrition', 'Attrition'],
            yticklabels=['No Attrition', 'Attrition'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix Heatmap")
plt.show()

# -------------------
# ✅ Final Prediction Example
# -------------------

# Example: Predict attrition for a new employee
new_employee = pd.DataFrame({
    'Age': [30],
    'YearsAtCompany': [2],
    'MonthlyIncome': [4000],
    'JobSatisfaction': [3],
    'WorkLifeBalance': [2],
    'OverTime': [1]   # 1 = Yes, 0 = No
})

prediction = model.predict(new_employee)[0]
prediction_proba = model.predict_proba(new_employee)[0][1]  # probability of Attrition=Yes

print("\nFinal Prediction for new employee:",
      "Attrition = Yes" if prediction==1 else "Attrition = No",
      f"(Probability of leaving: {prediction_proba:.2f})")
