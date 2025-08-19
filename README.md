# House Price Prediction API

## 📌 Problem
Predict house prices based on:
- Square footage
- Number of bedrooms
- Number of bathrooms
- Age of the house

## 🧠 Model
- **Model**: Linear Regression
- **Library**: scikit-learn
- **Training Data**: Synthetic data generated with a pricing formula
- **Serialization**: Model saved using joblib

## 🚀 API Endpoints

### GET `/`
Health check endpoint

### POST `/predict`
Input:
```json
{
  "square_footage": 2400,
  "bedrooms": 3,
  "bathrooms": 2.5,
  "house_age": 10
}
