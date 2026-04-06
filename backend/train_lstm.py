# train_lstm.py (Multivariate LSTM Upgrade)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import joblib

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping

# Create folders
os.makedirs("model", exist_ok=True)
os.makedirs("plots", exist_ok=True)

# Load dataset
data = pd.read_csv("data/india_city_aqi_2015_2023.csv")
data["date"] = pd.to_datetime(data["date"], dayfirst=True)
data = data.sort_values("date")

# Select multivariate features
features = ["pm25", "pm10", "no2", "so2", "co", "o3", "aqi"]
data = data[features].dropna()

# Scale all features
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data.values)

# Save scaler
joblib.dump(scaler, "model/lstm_scaler.pkl")

# Create sequences
def create_sequences(data, window_size=14):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size])
        y.append(data[i+window_size, -1])  # target = AQI (last column)
    return np.array(X), np.array(y)

window_size = 14
X, y = create_sequences(scaled_data, window_size)

# Train-test split (80/20)
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

print("Training shape:", X_train.shape)
print("Testing shape:", X_test.shape)

# Build Improved LSTM Model
model = Sequential([
    Input(shape=(window_size, len(features))),
    LSTM(64, return_sequences=True),
    Dropout(0.2),
    LSTM(32),
    Dropout(0.2),
    Dense(16, activation="relu"),
    Dense(1)
])

model.compile(optimizer="adam", loss="mse")
model.summary()

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True
)

history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=[early_stop],
    verbose=1
)

# Predict
y_pred = model.predict(X_test)

# Inverse scale only AQI
# To inverse, we rebuild full feature vector with predicted AQI
def inverse_aqi(pred, X_part):
    temp = X_part[:, -1, :].copy()  # last timestep features
    temp[:, -1] = pred.flatten()
    inv = scaler.inverse_transform(temp)
    return inv[:, -1]

y_test_inv = inverse_aqi(y_test.reshape(-1,1), X_test)
y_pred_inv = inverse_aqi(y_pred, X_test)

mae = mean_absolute_error(y_test_inv, y_pred_inv)
rmse = np.sqrt(mean_squared_error(y_test_inv, y_pred_inv))

print("\nImproved LSTM Evaluation:")
print("MAE :", round(mae, 2))
print("RMSE:", round(rmse, 2))

# Save model
model.save("model/lstm_model.keras")
print("Improved LSTM model saved.")

# Save forecast plot
plt.figure(figsize=(10,5))
plt.plot(y_test_inv[:200], label="Actual")
plt.plot(y_pred_inv[:200], label="Predicted")
plt.title("Improved LSTM Forecast vs Actual")
plt.legend()
plt.tight_layout()
plt.savefig("plots/lstm_forecast_improved.png")
plt.close()
