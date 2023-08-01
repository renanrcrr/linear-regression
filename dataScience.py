import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
data_url = "https://example.com/dataset.csv"
df = pd.read_csv(data_url)

# Data preprocessing
df.dropna(inplace=True)  # Remove rows with missing values
df['Date'] = pd.to_datetime(df['Date'])  # Convert 'Date' column to datetime type
df['Year'] = df['Date'].dt.year  # Extract year from 'Date' to a new 'Year' column

# Feature engineering
def extract_features(row):
    # Add custom features based on existing columns
    return row['Feature1'] * row['Feature2'] + np.log(row['Feature3'] + 1)

df['CustomFeature'] = df.apply(extract_features, axis=1)

# Prepare data for modeling
X = df[['Feature1', 'Feature2', 'Feature3', 'CustomFeature']]
y = df['Target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Model evaluation
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.4f}")
print(f"R-squared: {r2:.4f}")

# Feature importance
feature_importance = pd.Series(model.coef_, index=X.columns)
sorted_feature_importance = feature_importance.abs().sort_values(ascending=False)
print("\nFeature Importance:")
print(sorted_feature_importance)

# Save the model
import joblib
joblib.dump(model, 'linear_regression_model.pkl')
