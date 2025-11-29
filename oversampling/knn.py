# ==========================================
# 🤖 K-Nearest Neighbors (KNN) Classifier with Random OverSampling
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from imblearn.over_sampling import RandomOverSampler   # 🔁 OverSampling instead of UnderSampling

# -------------------
# 1️⃣ Load the dataset
# -------------------
data = pd.read_csv(r'C:\Users\Rahul Patel\Downloads\greendestination (1) (1).csv')

print("✅ Data Loaded Successfully")
print("Shape of Data:", data.shape)
print(data.head(), "\n")
print(data.info(), "\n")
print("Missing values:\n", data.isnull().sum(), "\n")

# -------------------
# 2️⃣ Handle missing values
# -------------------
data['Age'] = data['Age'].ffill()  # Forward fill missing values

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

print("\nClass Distribution after Oversampling:")
print(y_train_res.value_counts())

plt.figure(figsize=(5,4))
sns.countplot(x=y_train_res, palette='Set2')
plt.title("Attrition Class Distribution (After Oversampling)")
plt.xticks([0,1], ['No', 'Yes'])
plt.show()

# -------------------
# 8️⃣ Feature Scaling (Important for KNN)
# -------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_res)
X_test_scaled = scaler.transform(X_test)

# -------------------
# 9️⃣ Model Building - KNN Classifier
# -------------------
knn_model = KNeighborsClassifier(
    n_neighbors=5,        # Number of neighbors (you can tune this)
    metric='minkowski',   # Default distance metric (Euclidean)
    p=2                   # p=2 → Euclidean distance
)
knn_model.fit(X_train_scaled, y_train_res)

# -------------------
# 🔟 Make Predictions
# -------------------
y_pred = knn_model.predict(X_test_scaled)

# -------------------
# 1️⃣1️⃣ Model Evaluation
# -------------------
print("\n📈 Model Evaluation (KNN - Oversampled Data):")
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
plt.title("Confusion Matrix Heatmap (KNN - Oversampled Data)")
plt.show()

# -------------------
# 1️⃣2️⃣ Elbow Method (Optional)
# -------------------
# Helps to find optimal 'k' value by checking error rate
error_rate = []
for k in range(1, 21):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train_res)
    pred_k = knn.predict(X_test_scaled)
    error_rate.append(np.mean(pred_k != y_test))

plt.figure(figsize=(6,4))
plt.plot(range(1,21), error_rate, marker='o', linestyle='--', color='b')
plt.title('Elbow Method - Optimal k for KNN (Oversampled Data)')
plt.xlabel('Number of Neighbors (k)')
plt.ylabel('Error Rate')
plt.show()

# -------------------
# 1️⃣3️⃣ Prediction for a new employee
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
prediction = knn_model.predict(new_employee_scaled)[0]
probability = knn_model.predict_proba(new_employee_scaled)[0][1]

print("\nFinal Prediction for new employee (KNN - Oversampled Model):")
print("Attrition =", "Yes" if prediction == 1 else "No")
print(f"Probability of leaving: {probability:.2f}")
