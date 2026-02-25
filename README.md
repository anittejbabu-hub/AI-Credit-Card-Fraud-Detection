# AI-Credit-Card-Fraud-Detection
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# ===============================
# 1️⃣ Load Dataset
# ===============================
print("Loading Dataset...")

data = pd.read_csv("data/creditcard.csv")

print("Dataset Loaded Successfully ✅")
print("Dataset Shape:", data.shape)

# ===============================
# 2️⃣ Check Class Distribution
# ===============================
print("\nClass Distribution:")
print(data["Class"].value_counts())

# ===============================
# 3️⃣ Split Features and Target
# ===============================
X = data.drop("Class", axis=1)
y = data["Class"]

# ===============================
# 4️⃣ Feature Scaling
# ===============================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ===============================
# 5️⃣ Train Test Split
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

print("\nData Preprocessing Done ✅")

# ===============================
# 6️⃣ Logistic Regression (Balanced)
# ===============================
print("\nTraining Logistic Regression Model...")

lr_model = LogisticRegression(class_weight='balanced', max_iter=1000)
lr_model.fit(X_train, y_train)

lr_pred = lr_model.predict(X_test)

print("\nLogistic Regression Results:")
print("Confusion Matrix:")
print(confusion_matrix(y_test, lr_pred))

print("\nClassification Report:")
print(classification_report(y_test, lr_pred))

# ===============================
# 7️⃣ Random Forest Model
# ===============================
print("\nTraining Random Forest Model...")

rf_model = RandomForestClassifier(
    n_estimators=20,
    max_depth=10,
    n_jobs=-1,
    random_state=42
)

rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)

print("\nRandom Forest Results:")
print("Confusion Matrix:")
print(confusion_matrix(y_test, rf_pred))

print("\nClassification Report:")
print(classification_report(y_test, rf_pred))

# ===============================
# 8️⃣ Save Best Model (Random Forest)
# ===============================
pickle.dump(rf_model, open("model.pkl", "wb"))

print("\nModel Saved Successfully as model.pkl ✅")

print("\n🎉 Project Completed Successfully 🎉")
# ===============================
# 9️⃣ Confusion Matrix Heatmap
# ===============================

cm = confusion_matrix(y_test, rf_pred)

plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')

plt.title("Random Forest Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.show()
# ===============================
# 🔟 Feature Importance
# ===============================

importances = rf_model.feature_importances_
features = X.columns

feature_importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10,6))
sns.barplot(x='Importance', y='Feature',
            data=feature_importance_df.head(10))

plt.title("Top 10 Important Features (Random Forest)")
plt.show()
