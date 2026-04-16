# AQI Prediction System

This project predicts Air Quality Index (AQI) using machine learning models and provides real-time predictions through a FastAPI backend integrated with a frontend dashboard.

## Tech Stack
- Python, Scikit-learn
- FastAPI (Backend)
- React / TypeScript (Frontend)
- SQLite

## Features
- AQI prediction using ML models (Random Forest)
- Data preprocessing and feature scaling
- REST API for real-time predictions
- Interactive frontend dashboard
- Stores prediction history

## Project Structure
backend/   → Machine learning model & FastAPI API  
frontend/  → User interface and visualization  

## Run Backend
cd backend  
pip install -r requirements.txt  
uvicorn main:app --reload  

## Run Frontend
cd frontend  
npm install  
npm run dev  

Model Performance:0.95
