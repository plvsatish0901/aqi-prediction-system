# main.py

from fastapi import FastAPI
from fastapi import HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import os
import logging


from datetime import datetime
from tensorflow.keras.models import load_model
from fastapi.middleware.cors import CORSMiddleware

from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


#Initialize FastAPI App step 2
app = FastAPI(title="AQI Prediction API")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000","http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------
# DATABASE CONFIGURATION
# ----------------------------

DATABASE_URL = "sqlite:///./aqi_history.db"

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False}  # Needed for SQLite
)

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)

Base = declarative_base()


class PredictionHistory(Base):
    __tablename__ = "prediction_history"

    id = Column(Integer, primary_key=True, index=True)
    pm25 = Column(Float)
    pm10 = Column(Float)
    no2 = Column(Float)
    so2 = Column(Float)
    co = Column(Float)
    o3 = Column(Float)
    predicted_aqi = Column(Float)
    category = Column(String)
    timestamp = Column(DateTime, default=datetime.utcnow)


# Create the database tables
Base.metadata.create_all(bind=engine)


#Load Models step-3
# Load Random Forest model and scaler
rf_model = joblib.load("model/aqi_model.pkl")
rf_scaler = joblib.load("model/scaler.pkl")

# Load LSTM model and scaler
lstm_model = load_model("model/lstm_model.keras")
lstm_scaler = joblib.load("model/lstm_scaler.pkl")

# LSTM window size (must match training)
WINDOW_SIZE = 14
FEATURE_COUNT = 7


#define pollutants step 4
# Input schema for Random Forest
class PollutantInput(BaseModel):
    pm25: float
    pm10: float
    no2: float
    so2: float
    co: float
    o3: float


# Input schema for LSTM Forecast
class ForecastInput(BaseModel):
    last_14_days: list


#AQI Category Function step 5
def get_aqi_category(aqi):
    if aqi <= 50:
        return "Good"
    elif aqi <= 100:
        return "Satisfactory"
    elif aqi <= 200:
        return "Moderate"
    elif aqi <= 300:
        return "Poor"
    elif aqi <= 400:
        return "Very Poor"
    else:
        return "Severe"


#/predict Endpoint (Random Forest) step 6
@app.post("/predict")
def predict_aqi(data: PollutantInput):
    try:
        logger.info(f"Prediction request received: {data}")

        # Convert input to numpy array
        input_data = np.array([[
            data.pm25,
            data.pm10,
            data.no2,
            data.so2,
            data.co,
            data.o3
        ]], dtype=float)

        # Validate numeric values
        if np.any(np.isnan(input_data)) or np.any(np.isinf(input_data)):
            logger.warning("Invalid numeric values detected.")
            raise ValueError("Invalid numeric values provided.")

        if np.any(input_data < 0):
            logger.warning("Negative pollutant value detected.")
            raise ValueError("Pollutant values cannot be negative.")

        # Scale input  Uses same scaler from training.
        scaled_input = rf_scaler.transform(input_data)


        # Predict Returns AQI value.
        prediction = rf_model.predict(scaled_input)[0]

        if prediction is None or np.isnan(prediction):
            logger.error("Model returned invalid prediction.")
            raise ValueError("Model returned invalid prediction.")

        category = get_aqi_category(prediction)

        # Save prediction to database
        db = SessionLocal()  #Creates session.

        history_entry = PredictionHistory(                     #Creates row object
            pm25=data.pm25,
            pm10=data.pm10,
            no2=data.no2,
            so2=data.so2,
            co=data.co,
            o3=data.o3,
            predicted_aqi=float(prediction),
            category=category,
            timestamp=datetime.utcnow()
        )

        db.add(history_entry)           #Saves record permanently.
        db.commit()
        db.close()


        result = {
            "success": True,
            "model_used": "random_forest",
            "predicted_aqi": round(float(prediction), 2),
            "category": category,
            "timestamp": datetime.utcnow().isoformat()
        }

        logger.info(f"Prediction successful: {result}")

        return result

    except ValueError as ve:
        logger.error(f"Validation error: {str(ve)}")
        raise HTTPException(status_code=400, detail=str(ve))

    except Exception as e:
        logger.exception("Unexpected error during prediction.")
        raise HTTPException(
            status_code=500,
            detail="Internal server error during prediction."
        )

#/forecast Endpoint (LSTM) step 7

@app.post("/forecast")
def forecast_aqi(data: ForecastInput):
    try:
        # Validate length
        if len(data.last_14_days) != WINDOW_SIZE:
            raise ValueError(f"Provide exactly {WINDOW_SIZE} AQI values.")

        # Convert to numpy
        input_sequence = np.array(data.last_14_days, dtype=float)

        if np.any(np.isnan(input_sequence)) or np.any(np.isinf(input_sequence)):
            raise ValueError("Invalid AQI values provided.")

        # Reshape (WINDOW_SIZE, 1)
        input_sequence = input_sequence.reshape(-1, 1)

        # Prepare full feature array
        dummy_features = np.zeros((WINDOW_SIZE, FEATURE_COUNT))
        dummy_features[:, -1] = input_sequence.flatten()

        # Scale
        scaled_sequence = lstm_scaler.transform(dummy_features)

        # Reshape for LSTM
        scaled_sequence = scaled_sequence.reshape(1, WINDOW_SIZE, FEATURE_COUNT)

        # Predict
        pred_scaled = lstm_model.predict(scaled_sequence)

        if pred_scaled is None:
            raise ValueError("LSTM model returned invalid prediction.")

        # Inverse scale
        dummy_features[-1, -1] = pred_scaled[0][0]
        inv = lstm_scaler.inverse_transform(dummy_features)
        predicted_aqi = inv[-1, -1]

        category = get_aqi_category(predicted_aqi)

        return {
            "success": True,
            "model_used": "lstm_forecast",
            "forecast_aqi": round(float(predicted_aqi), 2),
            "category": category
        }

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Forecast failed: {str(e)}"
        )


@app.get("/")
def home():
    return {"message": "AQI Backend Running Successfully"}

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": True,
        "model_type": "Random Forest"
    }


@app.get("/model-info")
def model_info():
    return {
        "random_forest": {
            "type": "Regression",
            "features": 6,
            "status": "loaded"
        },
        "lstm": {
            "type": "Time Series Forecasting",
            "window_size": WINDOW_SIZE,
            "features": FEATURE_COUNT,
            "status": "loaded"
        }
    }

@app.get("/")
def root():
    return {
        "service": "AQI Prediction API",
        "version": "1.0.0",
        "model": "Random Forest"
    }


# ----------------------------
# GET PREDICTION HISTORY
# ----------------------------

@app.get("/history")
def get_prediction_history(limit: int = 20):
    try:
        db = SessionLocal()

        records = (
            db.query(PredictionHistory)
            .order_by(PredictionHistory.timestamp.desc())
            .limit(limit)
            .all()
        )

        db.close()

        return {
            "success": True,
            "count": len(records),
            "history": [
                {
                    "id": r.id,
                    "pm25": r.pm25,
                    "pm10": r.pm10,
                    "no2": r.no2,
                    "so2": r.so2,
                    "co": r.co,
                    "o3": r.o3,
                    "predicted_aqi": r.predicted_aqi,
                    "category": r.category,
                    "timestamp": r.timestamp.isoformat()
                }
                for r in records
            ]
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch history: {str(e)}"
        )
