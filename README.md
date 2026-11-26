# Food-Delivery-ETA-Prediction-System
End-to-end Machine Learning &amp; Deep Learning regression system to predict food delivery ETA using real-world logistics data (45.5K+ orders). Features geospatial Haversine distance engineering, XGBoost/LightGBM/ANN benchmarking, a FastAPI backend, and an interactive Streamlit web app.

A regression system that predicts food delivery time (in minutes) from order, courier, traffic and weather information.

**Live demo:** https://food-delivery-eta-prediction-system-yjmsp7s9gtjarkw3vahdnu.streamlit.app

## Business problem
Delivery platforms need accurate arrival estimates. Wrong estimates hurt customer trust, courier planning and restaurant timing. This project predicts the delivery time of an order so the platform can show realistic ETAs and plan courier capacity.

## Dataset
Food Delivery Dataset from Kaggle (45K+ orders, 15+ features): https://www.kaggle.com/datasets/gauravmalik26/food-delivery-dataset

- Target: delivery time in minutes
- Features: courier age and rating, vehicle type and condition, traffic density, weather, order type, festival, city type, order time, distance and restaurant preparation time

## Pipeline
1. Data cleaning: missing values, duplicates and outliers
2. EDA: distributions, correlations, boxplots and insights
3. Feature engineering: distance, order hour, weekend flag, rush-hour flag, preparation time
4. Preprocessing: one-hot encoding and scaling inside a single scikit-learn pipeline
5. Models: Ridge Regression, Random Forest, Gradient Boosting
6. Tuning: RandomizedSearchCV on the Random Forest
7. Deployment: model saved with Joblib, served through a Streamlit app

## Results
| Model | MAE (min) | RMSE | R2 |
|---|---|---|---|
| Ridge Regression | 4.81 | - | 0.56 |
| Gradient Boosting | 3.66 | 4.58 | 0.75 |
| Random Forest (tuned) | 3.16 | 3.98 | 0.81 |

Random Forest was selected as the final model because it gave the lowest error and the highest R2 on the same test split.

## Run locally
```
pip install -r requirements.txt
streamlit run app.py
```

## Files
- `app.py`: Streamlit user interface
- `model.joblib`: trained pipeline
- `requirements.txt`: dependencies
