# Vehicle Fuel Efficiency Prediction

End-to-end machine learning project that predicts a vehicle's fuel efficiency (MPG) using the [Auto MPG dataset](https://archive.ics.uci.edu/ml/datasets/auto+mpg).

## Project Structure

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

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Add data

Place the `auto-mpg.data` file into `data/raw/`.

### 3. Train the model

```bash
python src/train.py
```

This will:
- Load and preprocess the data
- Train Linear Regression, Ridge, Lasso, ElasticNet, and XGBoost models
- Build an averaging ensemble (XGBoost + Lasso)
- Save the best model, scaler, and metadata to `models/`

### 4. Make predictions (CLI)

```bash
python src/predict.py
```

### 5. Launch Streamlit app

```bash
streamlit run app/app.py
```

## Models

| Model | Description |
|---|---|
| Linear Regression | Baseline OLS |
| Ridge (L2) | Regularized with GridSearch alpha tuning |
| Lasso (L1) | Feature selection via L1 penalty |
| ElasticNet (L1+L2) | Combined regularization |
| XGBoost | Gradient boosting with hyperparameter search |
| Averaged Ensemble | Mean of XGBoost + Lasso predictions |

## Dataset

- **Source:** UCI Machine Learning Repository — Auto MPG
- **Samples:** 398 vehicles
- **Target:** MPG (Miles Per Gallon)
- **Features:** Cylinders, Displacement, Horsepower, Weight, Acceleration, Model Year, Origin

## License

This project is for educational purposes.
