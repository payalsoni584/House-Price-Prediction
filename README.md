
# 🏠 House Price Prediction

A Machine Learning project that predicts house prices based on property details such as area, bedrooms, bathrooms, stories, and parking spaces.

## 🌟 Project Overview

This project uses regression algorithms to estimate house prices from property features. A Streamlit web application allows users to enter house details and get a predicted price.

## 🎯 Objectives

- Predict house prices using Machine Learning.
- Train and compare regression models.
- Evaluate model performance using regression metrics.
- Build an interactive web application using Streamlit.
- Deploy the application online.

## 🛠️ Technologies Used

- Python
- Pandas
- Scikit-learn
- Streamlit
- Joblib
- Matplotlib (if used in your code)

## 📊 Dataset

The dataset contains 545 records and 6 columns.

Features:
- `area`
- `bedrooms`
- `bathrooms`
- `stories`
- `parking`

Target variable:
- `price`

## 🤖 Machine Learning Models

The following regression approaches were evaluated:

1. Linear Regression
2. Polynomial Regression (degree 2)

## 📈 Model Evaluation

The dataset was split into training and testing sets using an 80:20 ratio with `random_state=42`.

| Metric | Linear Regression | Polynomial Regression |
|---|---:|---:|
| MAE | 1,127,483.35 | 1,132,221.08 |
| MSE | 2,292,721,545,725.36 | 2,323,567,024,084.57 |
| RMSE | 1,514,173.55 | 1,524,325.10 |
| R² Score | 0.5464 | 0.5403 |

Linear Regression performed slightly better than Polynomial Regression on this test split, with lower MAE and RMSE and a higher R² score. Therefore, Linear Regression was selected as the saved model.

## 🚀 Live Demo

[Open House Price Prediction App](https://house-price-prediction-qnymnuedxrscptyrfxwfp.streamlit.app/)

## 💻 Run the Project Locally

1. Clone the repository:

   ```bash
   git clone https://github.com/payalsoni584/House-Price-Prediction.git
   ```

2. Open the project folder:

   ```bash
   cd House-Price-Prediction
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Run the Streamlit app:

   ```bash
   streamlit run app.py
   ```

## 📁 Project Structure

```text
House-Price-Prediction/
├── app.py
├── main.py
├── house_price_model.pkl
├── Housing_Price_Dataset.csv
├── requirements.txt
└── README.md
```

## 🔮 Future Improvements

- Evaluate additional regression models.
- Improve prediction accuracy with feature engineering.
- Add visualizations and exploratory data analysis.
- Improve the user interface.

## 👩‍💻 Author

Payal Soni

GitHub: [payalsoni584](https://github.com/payalsoni584)

