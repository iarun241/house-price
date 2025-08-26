import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib

# Load data
data = pd.read_csv("C:\my _app\house_prices.csv")

# Example: assume columns
X = data[['bedrooms', 'bathrooms', 'sqft_living']]
y = data['price']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "model.pkl")
print("✅ Model saved as model.pkl")
