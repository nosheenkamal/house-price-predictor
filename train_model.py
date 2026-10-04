import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression

# Load dataset
df = pd.read_csv("data (1).csv")

# Remove invalid zero-price houses
df = df[df["price"] > 0].copy()

# Features
features = [
    "bedrooms",
    "bathrooms",
    "sqft_living",
    "sqft_lot",
    "floors",
    "waterfront",
    "view",
    "condition",
    "sqft_above",
    "sqft_basement",
    "yr_built",
    "yr_renovated",
    "city",
    "statezip"
]

X = df[features]
y = df["price"]

# Separate numerical and categorical features
categorical_features = ["city", "statezip"]

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ],
    remainder="passthrough"
)

# Build model
model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression())
    ]
)

# Train model
model.fit(X, y)

# Save model
joblib.dump(model, "house_price_model.pkl")

print("Model trained successfully!")
print("Rows used:", len(df))
print("Model saved as house_price_model.pkl")