import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
import joblib

# load dataset
df = pd.read_csv("data/kc_house_data.csv")

# first 5 row
print(df.head())

# Shape
print("Shape:",df.shape)

# columns
print("Columns")
print(df.columns)

# Dataset info
print("Dataset Info")
print(df.info())

# Remove Unwanted columns
df = df.drop(["id","date"],axis=1)

# Check Missing Values
print("\nMissing Values:")
print(df.isnull().sum())

# Check Duplicate Rows
print("\nDuplicate Rows:")
print(df.duplicated().sum())

# Statistical Summary
print("\nStatistical Summary:")
print(df.describe())

# Remove Duplicates
df = df.drop_duplicates()

# Which features have the strongest relationship with price
print(df.corr(numeric_only=True)["price"].sort_values(ascending=False))

# Create Feature(x) and Target(y)
x = df.drop("price",axis=1)
y = df["price"]

print("x shape:",x.shape)
print("y shape:",y.shape)

# Train Test Split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=42)
print("X_train:", x_train.shape)
print("X_test :", x_test.shape)
print("y_train:", y_train.shape)
print("y_test :", y_test.shape)

# Train the Model
model = LinearRegression()
model.fit(x_train,y_train)

# Predict House Prices
y_pred = model.predict(x_test)

# Compare
comparison = pd.DataFrame({
    "Actual Price": y_test,
    "Predicted Price": y_pred
})

print(comparison.head(10))

# Evaluate the Model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("MAE :", mae)
print("MSE :", mse)
print("RMSE:", rmse)
print("R² Score:", r2)


joblib.dump(model, "models/house_price_model.pkl")
print("Model Saved Successfully!")









# # remove id column
# df = df.drop('Id',axis=1)

# # Encode Categorical Columns
# df = pd.get_dummies(
#     df,
#     columns=["Location", "Condition", "Garage"],
#     drop_first=True
# )

# # Feature and Target
# x = df.drop("Price",axis=1)
# y = df["Price"]

# print(x.head())
# print(y.head())
# print(x.shape)
# print(y.shape)

# # train-test split
# x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

# # chack shape
# print("X_train:", x_train.shape)
# print("X_test :", x_test.shape)

# print("y_train:", y_train.shape)
# print("y_test :", y_test.shape)

# # Train the model
# model = LinearRegression()
# model.fit(x_train,y_train)
# print("model trained successfully")

# # Predict house price 
# y_pred = model.predict(x_test)

# # comparison = pd.DataFrame({
# #     "Actual Price": y_test,
# #     "Predicted Price": y_pred
# # })

# # print(comparison.head(10))

# # Evaluate the Model
# mae = mean_absolute_error(y_test, y_pred)
# mse = mean_squared_error(y_test, y_pred)
# rmse = np.sqrt(mse)
# r2 = r2_score(y_test, y_pred)

# print("MAE :", mae)
# print("MSE :", mse)
# print("RMSE:", rmse)
# print("R² Score:", r2)
# print(df.corr(numeric_only=True)["Price"].sort_values(ascending=False))