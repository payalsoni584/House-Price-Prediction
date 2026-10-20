import pandas as pd

# Load dataset
df = pd.read_csv("Housing_Price_Dataset.csv")

# Dataset ki basic information
print("First 5 rows:")
print(df.head())

print("\nDataset shape:")
print(df.shape)

print("\nColumn names:")
print(df.columns)

print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isnull().sum())
from sklearn.model_selection import train_test_split

# Features and target
X = df.drop("price", axis=1)
y = df["price"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTraining data shape:")
print(X_train.shape)

print("\nTesting data shape:")
print(X_test.shape)
from sklearn.linear_model import LinearRegression

# Model create
lr_model = LinearRegression()

# Training
lr_model.fit(X_train, y_train)

print("Linear Regression Model Trained Successfully!")
y_pred = lr_model.predict(X_test)

print("\nFirst 5 Predictions:")
print(y_pred[:5])
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation:")
print("MAE =", mae)
print("MSE =", mse)
print("RMSE =", rmse)
print("R2 Score =", r2)
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

poly_model = make_pipeline(
    PolynomialFeatures(degree=2),
    LinearRegression()
)

poly_model.fit(X_train, y_train)

poly_pred = poly_model.predict(X_test)

print("\nPolynomial Regression Model Trained Successfully!")
poly_mae = mean_absolute_error(y_test, poly_pred)
poly_mse = mean_squared_error(y_test, poly_pred)
poly_rmse = np.sqrt(poly_mse)
poly_r2 = r2_score(y_test, poly_pred)

print("\nPolynomial Regression Evaluation:")
print("MAE =", poly_mae)
print("MSE =", poly_mse)
print("RMSE =", poly_rmse)
print("R2 Score =", poly_r2)
import joblib

joblib.dump(lr_model, "house_price_model.pkl")

print("Model saved successfully!")
new_house = pd.DataFrame({
    "area": [5000],
    "bedrooms": [3],
    "bathrooms": [2],
    "stories": [2],
    "parking": [1]
})

predicted_price = lr_model.predict(new_house)

print("Predicted House Price =", predicted_price[0])

# Compare both models

results = pd.DataFrame({
    "Model": ["Linear Regression", "Polynomial Regression"],
    "MAE": [mae, poly_mae],
    "MSE": [mse, poly_mse],
    "RMSE": [rmse, poly_rmse],
    "R2 Score": [r2, poly_r2]
})

print("\nModel Comparison:")
print(results)

# Select best model based on lower RMSE
if poly_rmse < rmse:
    best_model = poly_model
    best_model_name = "Polynomial Regression"
else:
    best_model = lr_model
    best_model_name = "Linear Regression"

print("\nBest Model:", best_model_name)

# Save best model
import joblib

joblib.dump(best_model, "best_model.pkl")

print("Best model saved successfully!")

