# 🏡 House Price Prediction API  

**Name:** Pabasara Senevirathne 
**Reg. No:** ITBIN-2211-0285  
**Course:** IT41043 - Intelligent Systems  
**GitHub Repository:** https://github.com/PabasaraSenevirathne/ML-Model-Inference-with-Fast-API.git  
**Localhost:** http://127.0.0.1:8000/docs  

---

## 📌 1. Project Overview  
This project implements a **machine learning regression model** to predict house prices based on property features:  
- Square footage  
- Number of bedrooms  
- Number of bathrooms  
- Age of the house  

The trained model is deployed as a **FastAPI web service**, enabling real-time predictions through API endpoints.  

---

## 🎯 2. Problem Statement  
Accurately predicting house prices helps **buyers, sellers, and real-estate businesses** in decision-making.  

**Objective:**  
- Build an ML regression model for house price prediction.  
- Deploy the trained model with FastAPI for **online inference**.  

---

## 📊 3. Dataset  
- **Source:** Synthetic dataset (Housing.csv / generated features).  
- **Features:**  
  - `square_footage`  
  - `bedrooms`  
  - `bathrooms`  
  - `house_age`  
- **Target:** House price  

**Preprocessing:**  
- Checked for missing values.  
- Feature scaling (optional for some models).  

---

## 🔎 4. Exploratory Data Analysis  
- **Summary statistics** for each feature.  
- **Scatter plot:** Square footage vs Price → positive correlation.  
- **Correlation heatmap** for feature–target relationships.  

---

## 🤖 5. Model Development  
- **Train/Test Split:** 80% training, 20% testing.  
- **Algorithms Tested:**  
  - Linear Regression ✅  
  - Random Forest Regressor  
- **Best Model:** Linear Regression (simple, interpretable, fast).  
- **Evaluation Metrics:**  
  - R² Score on test set  
  - Mean Squared Error (MSE)  

Model serialized with **joblib** as `model.pkl`.  

---

## ⚙️ 6. Model Pipeline  
`Input features → Preprocessing → Regression Model → Prediction`  

- Exported trained model as **joblib pipeline** for deployment.  

---

## 🚀 7. FastAPI Implementation  
**Endpoints:**  
1. `GET /` → Health check  
2. `POST /predict` → Predict house price from input JSON  
3. `GET /model-info` → Returns model metadata  

**Error Handling:**  
- Invalid inputs return **HTTP 400** with descriptive error message.  

**Deployment:**  
- Run locally:  
```bash
uvicorn main:app --reload
