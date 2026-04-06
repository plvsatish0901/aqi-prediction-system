# train_model.py

# 1️⃣ Import Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor


# 2️⃣ Load Dataset
data = pd.read_csv("data/india_city_aqi_2015_2023.csv")
# Create plots directory
os.makedirs("plots", exist_ok=True)

print("Dataset Loaded Successfully")
print("Shape:", data.shape)
print("\nDataset Info:")
print(data.info())

print("\nStatistical Summary:")
print(data.describe())


# 3️⃣ Data Preprocessing

# Convert date column to datetime (if exists)
if "date" in data.columns:
    data["date"] = pd.to_datetime(data["date"], dayfirst=True)
    data = data.sort_values("date")

# Drop duplicates
data = data.drop_duplicates()

# Check missing values
print("\nMissing Values:\n", data.isnull().sum())

# Drop rows with missing values (simple clean approach)
data = data.dropna()

# Correlation Matrix step - 3
plt.figure(figsize=(10,6))
sns.heatmap(data.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.tight_layout()
plt.savefig("plots/correlation_matrix.png")
plt.close()


# 4️⃣ Feature Selection
features = ["pm25", "pm10", "no2", "so2", "co", "o3"]
target = "aqi"

X = data[features]
y = data[target]

# 5️⃣ Train-Test Split (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTraining samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])

# 6️⃣ Feature Scaling
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 7️⃣ Model Training
model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

model.fit(X_train_scaled, y_train)

# Feature Importance step-5
importances = model.feature_importances_
feature_names = features

# randon forest 
importances = model.feature_importances_
feature_names = features

plt.figure(figsize=(8,5))
sns.barplot(x=importances, y=feature_names)
plt.title("Feature Importance (Random Forest)")
plt.xlabel("Importance")
plt.ylabel("Features")
plt.tight_layout()
plt.savefig("plots/feature_importance.png")
plt.close()


# 8️⃣ Model Evaluation
y_pred = model.predict(X_test_scaled)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation:")
print("MAE :", round(mae, 2))
print("RMSE:", round(rmse, 2))
print("R²  :", round(r2, 4))

# 🔵 Gradient Boosting Model
gb_model = GradientBoostingRegressor(
    n_estimators=200,
    learning_rate=0.1,
    random_state=42
)

gb_model.fit(X_train_scaled, y_train)

gb_pred = gb_model.predict(X_test_scaled)

gb_mae = mean_absolute_error(y_test, gb_pred)
gb_rmse = np.sqrt(mean_squared_error(y_test, gb_pred))
gb_r2 = r2_score(y_test, gb_pred)

print("\nGradient Boosting Evaluation:")
print("MAE :", round(gb_mae, 2))
print("RMSE:", round(gb_rmse, 2))
print("R²  :", round(gb_r2, 4))


# 9️⃣ Save Model & Scaler
os.makedirs("model", exist_ok=True)

joblib.dump(model, "model/aqi_model.pkl")
joblib.dump(scaler, "model/scaler.pkl")

print("\nModel and Scaler saved successfully inside 'model/' folder.")

#step - 6 actual vs prediction
plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred, alpha=0.3)
plt.xlabel("Actual AQI")
plt.ylabel("Predicted AQI")
plt.title("Actual vs Predicted AQI")
plt.tight_layout()
plt.savefig("plots/actual_vs_predicted.png")
plt.close()


#step-4 AQI distibution
plt.figure(figsize=(8,5))
sns.histplot(data["aqi"], bins=30, kde=True)
plt.title("AQI Distribution")
plt.xlabel("AQI")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("plots/aqi_distribution.png")
plt.close()



