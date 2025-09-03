# random_forest.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
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
# 🤖 Model Building (Random Forest)
# -------------------

X = data[['Age', 'YearsAtCompany', 'MonthlyIncome', 'JobSatisfaction', 'WorkLifeBalance', 'OverTime']]
y = data['Attrition']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

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
plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.title("Confusion Matrix - Random Forest")
plt.show()

# Feature importance
importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\nFeature importances:\n", importances)

# Final prediction
new_employee = pd.DataFrame({
    'Age': [30],
    'YearsAtCompany': [2],
    'MonthlyIncome': [4000],
    'JobSatisfaction': [3],
    'WorkLifeBalance': [2],
    'OverTime': [1]
})
pred = model.predict(new_employee)[0]
proba = model.predict_proba(new_employee)[0][1]
print("\nFinal Prediction for new employee:",
      "Attrition = Yes" if pred==1 else "Attrition = No",
      f"(Probability of leaving: {proba:.2f})")
