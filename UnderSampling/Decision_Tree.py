# ==========================================
# 🌳 Decision Tree Classifier with Random UnderSampling
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from imblearn.under_sampling import RandomUnderSampler

# -------------------
# 1️⃣ Load the dataset
# -------------------
data = pd.read_csv(r'C:\Users\Rahul Patel\Downloads\greendestination (1) (1).csv')

print("✅ Data Loaded Successfully")
print("Shape of Data:", data.shape)
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
# 4️⃣ Check class imbalance before undersampling
# -------------------
print("\nClass Distribution before Undersampling:")
print(data['Attrition'].value_counts())

plt.figure(figsize=(5,4))
sns.countplot(x='Attrition', data=data, palette='coolwarm')
plt.title("Attrition Class Distribution (Before Undersampling)")
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
# 7️⃣ Apply Random UnderSampling
# -------------------
rus = RandomUnderSampler(random_state=42)
X_train_res, y_train_res = rus.fit_resample(X_train, y_train)

print("\nClass Distribution after Undersampling:")
print(y_train_res.value_counts())

plt.figure(figsize=(5,4))
sns.countplot(x=y_train_res, palette='Set2')
plt.title("Attrition Class Distribution (After Undersampling)")
plt.xticks([0,1], ['No', 'Yes'])
plt.show()

# -------------------
# 8️⃣ Model Building - Decision Tree Classifier
# -------------------
dt_model = DecisionTreeClassifier(
    criterion='gini',      # or 'entropy'
    max_depth=None,         # Grow until all leaves are pure
    random_state=42
)

dt_model.fit(X_train_res, y_train_res)

# -------------------
# 🔟 Predictions
# -------------------
y_pred = dt_model.predict(X_test)

# -------------------
# 1️⃣1️⃣ Model Evaluation
# -------------------
print("\n📈 Model Evaluation:")
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
plt.title("Confusion Matrix Heatmap (Decision Tree - Undersampled Data)")
plt.show()

# -------------------
# 1️⃣2️⃣ Visualize Decision Tree
# -------------------
plt.figure(figsize=(15,8))
plot_tree(dt_model, feature_names=X.columns, class_names=['No', 'Yes'],
          filled=True, rounded=True, fontsize=10)
plt.title("Decision Tree Visualization")
plt.show()

# -------------------
# 1️⃣3️⃣ Predict for a new employee
# -------------------
new_employee = pd.DataFrame({
    'Age': [30],
    'YearsAtCompany': [2],
    'MonthlyIncome': [4000],
    'JobSatisfaction': [3],
    'WorkLifeBalance': [2],
    'OverTime': [1]   # 1 = Yes, 0 = No
})

prediction = dt_model.predict(new_employee)[0]
probability = dt_model.predict_proba(new_employee)[0][1]

print("\nFinal Prediction for new employee:")
print("Attrition =", "Yes" if prediction == 1 else "No")
print(f"Probability of leaving: {probability:.2f}")
