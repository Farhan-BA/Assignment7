import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
df = pd.read_csv('C:\python\glass.csv')
print("Sample Data:")
print(df.head())
print("\nMissing values:")
print(df.isnull().sum())
print("\nData Types:")
print(df.dtypes)
print("\nDuplicate Rows:", df.duplicated().sum())
df.drop_duplicates(inplace=True)
print("\nSummary Stats:")
print(df.describe())
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()
# Bar plot for target variable
plt.figure(figsize=(7, 5))
sns.countplot(x='Type', data=df)
plt.title("Glass Type Distribution")
plt.xlabel("Glass Type")
plt.ylabel("Count")
plt.show()
X = df.drop('Type', axis=1)
y = df['Type']
# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# Model: Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)
# Evaluation
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("Accuracy Score:", accuracy_score(y_test, y_pred)
# Confusion Matrix
conf_mat = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 5))
sns.heatmap(conf_mat, annot=True, fmt="d", cmap='Blues', xticklabels=np.unique(y), yticklabels=np.unique(y))
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
joblib.dump(model, 'glass_model.pkl')
joblib.dump(scaler, "scaler.pkl")