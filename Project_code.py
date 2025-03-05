import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
data = pd.read_csv(r'C:\Users\Rahul Patel\Downloads\greendestination (1) (1).csv')

print(data.head())
print(data.info())
print(data.describe())
print(data.isnull().sum())
data['Age'] = data['Age'].ffill()  # Forward fill for the 'Age' column
  # Example for forward fill
sns.boxplot(data['Age'])
data['Attrition'] = data['Attrition'].map({'Yes': 1, 'No': 0})
attrition_rate = data['Attrition'].mean() * 100
print(f"Attrition Rate: {attrition_rate:.2f}%")
sns.countplot(x='Attrition', data=data)
plt.title('Attrition Distribution')
plt.show()
sns.boxplot(x='Attrition', y='Age', data=data)
plt.title('Age vs. Attrition')
plt.show()
sns.boxplot(x='Attrition', y='YearsAtCompany', data=data)


plt.title('Years at Company vs. Attrition')
plt.show()
sns.boxplot(x='Attrition', y='MonthlyIncome', data=data)


plt.title('Income vs. Attrition')
plt.show()
from sklearn.model_selection import train_test_split

X = data[['Age', 'YearsAtCompany', 'MonthlyIncome']]  # Independent variables

y = data['Attrition']  # Dependent variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
