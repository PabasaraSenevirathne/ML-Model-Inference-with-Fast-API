import numpy as np # type: ignore
import pandas as pd # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.linear_model import LinearRegression # type: ignore
import joblib # type: ignore

# ✅ Generate synthetic data
np.random.seed(42)
data_size = 1000

square_footage = np.random.randint(500, 5000, size=data_size)
bedrooms = np.random.randint(1, 6, size=data_size)
bathrooms = np.round(np.random.uniform(1, 4, size=data_size), 1)
house_age = np.random.randint(0, 100, size=data_size)

# 🧠 Price formula (synthetic)
price = (
    square_footage * 150
    + bedrooms * 10000
    + bathrooms * 7000
    - house_age * 500
    + np.random.normal(0, 20000, size=data_size)
)


# 🧾 Create DataFrame
df = pd.DataFrame({
    "square_footage": square_footage,
    "bedrooms": bedrooms,
    "bathrooms": bathrooms,
    "house_age": house_age,
    "price": price
})

# 🔀 Train/test split
X = df[["square_footage", "bedrooms", "bathrooms", "house_age"]]
y = df["price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🎯 Train model
model = LinearRegression()
model.fit(X_train, y_train)

# 💾 Save model
joblib.dump(model, "model.pkl")
print("✅ Model saved as model.pkl")
