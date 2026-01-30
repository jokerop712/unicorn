import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


# Create synthetic data
np.random.seed(0)

X = np.random.rand(100, 1)          # Independent variable
y = 20 * X + 1 + np.random.randn(100, 1)  # Dependent variable with noise


# Convert data to DataFrame
data = pd.DataFrame({
    'X': X.flatten(),
    'y': y.flatten()
})


# Visualize the data
plt.scatter(X, y)
plt.xlabel('Independent Variable (X)')
plt.ylabel('Dependent Variable (y)')
plt.title('Sample Data for Linear Regression')
plt.show()


# Split data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# Create and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)


# Make predictions
y_pred = model.predict(X_test)


# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r_squared = r2_score(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R-squared (R²): {r_squared:.2f}")


# Plot regression line
plt.scatter(X, y, label='Data')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Regression Line')
plt.xlabel('Independent Variable (X)')
plt.ylabel('Dependent Variable (y)')
plt.title('Linear Regression Model')
plt.legend()
plt.show()


# Model parameters
coefficients = model.coef_
intercept = model.intercept_

print(f"Coefficient: {coefficients[0][0]:.2f}")
print(f"Intercept: {intercept[0]:.2f}")
