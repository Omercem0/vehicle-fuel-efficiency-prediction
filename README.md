#  AutoMPG Predictor - AI-Driven Fuel Efficiency Analysis

AutoMPG Predictor is an end-to-end Machine Learning web application designed to predict the fuel efficiency (Miles Per Gallon) of vehicles based on technical specifications. The project leverages advanced regression techniques and is deployed using a modular architecture.

🔗 **Live Demo:** [vehicle-fuel-efficiency-prediction-omer.streamlit.app](https://vehicle-fuel-efficiency-prediction-omer.streamlit.app/)

---

## 🚀 Key Features

- **Hybrid Regression Engine:** Instead of a single model, this project implements an Averaging Ensemble combining XGBoost and Lasso Regression to minimize prediction error.
- **Scientific Data Pipeline:** Features a robust preprocessing stage including outlier detection, handling skewed distributions (Log Transformation), and Robust Scaling to manage data variance.
- **Interactive Insights:** Users can input vehicle parameters such as cylinders, horsepower, weight, and model year to get real-time MPG estimations.

---

## 🛠️ Engineering Pipeline

This project is a complete data science product consisting of three core stages:

### 1. Exploratory Data Analysis (EDA) 📊
Located in `notebooks/eda.ipynb`, this phase involves:
- **Distribution Analysis:** Identifying skewness in target variables and applying normal distribution transforms.
- **Correlation Mapping:** Visualizing how weight and displacement impact fuel consumption using Seaborn heatmaps.
- **Data Cleaning:** Managing missing values and removing outliers that degrade model generalization.

### 2. Model Development (The Engine) ⚙️
The training pipeline in `src/train.py` handles:
- **Advanced Modeling:** Utilizing Lasso (L1 Regularization) for feature selection and XGBoost for capturing non-linear patterns.
- **Performance Metrics:** Models are evaluated using Mean Squared Error (MSE), achieving high accuracy through the Averaging ensemble technique.
- **Artifact Export:** Scalers and model weights are serialized into `.pkl` files for production use.

### 3. Deployment (The Product) 💻
- **Streamlit UI:** A lightweight, interactive frontend built to serve the model predictions.
- **Modular Architecture:** Clean separation of concerns between data processing (`data_preprocessing.py`), model logic (`predict.py`), and the user interface.

---

## 📂 Project Structure

```
ml-project/
├── data/
│   └── raw/                  # Raw dataset (auto-mpg.data)
├── notebooks/
│   └── eda.ipynb             # Exploratory Data Analysis
├── src/
│   ├── data_preprocessing.py # Data loading, cleaning & feature engineering
│   ├── train.py              # Model training & evaluation pipeline
│   └── predict.py            # Prediction API
├── models/
│   ├── model.pkl             # Trained model artifact
│   ├── scaler.pkl            # Fitted scaler
│   └── metadata.json         # Model metadata & feature columns
├── app/
│   └── app.py                # Streamlit web UI
├── requirements.txt
└── README.md
```

---

## 🛠️ Tech Stack

| Layer | Tools |
|---|---|
| Frontend | Streamlit |
| Machine Learning | Scikit-Learn, XGBoost |
| Data Processing | Pandas, NumPy, SciPy |
| Visualization | Seaborn, Matplotlib |
| Serialization | Joblib / Pickle |

---

## 👨‍💻 Author

**Ömer Cem Tanrıkulu** - Computer Engineering Student

[![LinkedIn](https://img.shields.io/badge/LinkedIn-blue?logo=linkedin)](https://linkedin.com/in/omercemtanrikulu/)
[![GitHub](https://img.shields.io/badge/GitHub-black?logo=github)](https://github.com/Omercem0)
[![Kaggle](https://img.shields.io/badge/Kaggle-blue?logo=kaggle)](https://kaggle.com/omercemtanrikulu)
