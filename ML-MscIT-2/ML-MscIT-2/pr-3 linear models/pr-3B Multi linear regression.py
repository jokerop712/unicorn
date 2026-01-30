import numpy as np
import pandas as pd

np.random.seed(0)

# Generate dataset with 3 features
X = np.random.rand(100, 3)
y = X[:, 0] + 2 * X[:, 1] + 3 * X[:, 2] + np.random.randn(100) * 0.1


# Create DataFrame
df = pd.DataFrame(X, columns=['x1', 'x2', 'x3'])
df['y'] = y


# Imports for modeling
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm


# Split data
X_train, X_test, y_train, y_test = train_test_split(
    df[['x1', 'x2', 'x3']],
    df['y'],
    test_size=0.2,
    random_state=42
)


# Train Multiple Linear Regression model
mlr = LinearRegression()
mlr.fit(X_train, y_train)


# Predictions
y_pred = mlr.predict(X_test)


# Evaluation
print("Multiple Linear Regression")
print("Coefficients:", mlr.coef_)
print("Intercept:", mlr.intercept_)
print("MSE:", mean_squared_error(y_test, y_pred))
print("R²:", r2_score(y_test, y_pred))


# Variance Inflation Factor (VIF)
X_vif = sm.add_constant(df[['x1', 'x2', 'x3']])

vif_data = pd.DataFrame()
vif_data["Feature"] = X_vif.columns
vif_data["VIF"] = [
    variance_inflation_factor(X_vif.values, i)
    for i in range(X_vif.shape[1])
]


print("\nVIF Values:")
print(vif_data)
