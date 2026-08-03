import joblib
import pandas as pd

# Load Trained Model
model = joblib.load("models/house_price_model.pkl")


def predict_price(
    bedrooms,
    bathrooms,
    sqft_living,
    sqft_lot,
    floors,
    waterfront,
    view,
    condition,
    grade,
    sqft_above,
    sqft_basement,
    yr_built,
    yr_renovated,
    zipcode,
    lat,
    long,
    sqft_living15,
    sqft_lot15
):
    """
    Predict house price using the trained Linear Regression model.
    """

    # Create DataFrame (same columns as used during training)
    data = pd.DataFrame({
        "bedrooms": [bedrooms],
        "bathrooms": [bathrooms],
        "sqft_living": [sqft_living],
        "sqft_lot": [sqft_lot],
        "floors": [floors],
        "waterfront": [waterfront],
        "view": [view],
        "condition": [condition],
        "grade": [grade],
        "sqft_above": [sqft_above],
        "sqft_basement": [sqft_basement],
        "yr_built": [yr_built],
        "yr_renovated": [yr_renovated],
        "zipcode": [zipcode],
        "lat": [lat],
        "long": [long],
        "sqft_living15": [sqft_living15],
        "sqft_lot15": [sqft_lot15]
    })

    # Make Prediction
    prediction = model.predict(data)

    # Return Predicted Price
    return prediction[0]