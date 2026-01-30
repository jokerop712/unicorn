import numpy as np 
import pandas as pd 
np.random.seed(0) 
X = np.random.rand(100, 3) # Generate dataset with 3 features 
y = X[:, 0] + 2*X[:, 1] + 3*X[:, 2] + np.random.randn(100)*0.1 
df = pd.DataFrame(X, columns=['x1', 'x2', 'x3']) # Create DataFrame 
df['y'] = y 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import Ridge, Lasso, ElasticNet 
from sklearn.preprocessing import StandardScaler 
from sklearn.metrics import mean_squared_error, r2_score 
scaler = StandardScaler() # Feature scaling (important for regularization) 
X_scaled = scaler.fit_transform(df[['x1','x2','x3']]) 
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42) 
ridge = Ridge(alpha=1.0) 
ridge.fit(X_train, y_train) 
ridge_pred = ridge.predict(X_test) 
print("\nRidge Regression") 
print("Coefficients:", ridge.coef_) 
print("MSE:", mean_squared_error(y_test, ridge_pred)) 
print("R²:", r2_score(y_test, ridge_pred)) 

lasso = Lasso(alpha=0.1) 
lasso.fit(X_train, y_train) 
lasso_pred = lasso.predict(X_test) 
print("\nLasso Regression") 
print("Coefficients:", lasso.coef_) 
print("MSE:", mean_squared_error(y_test, lasso_pred)) 
print("R²:", r2_score(y_test, lasso_pred)) 

elastic = ElasticNet(alpha=0.1, l1_ratio=0.5) 
elastic.fit(X_train, y_train) 
elastic_pred = elastic.predict(X_test) 
print("\nElasticNet Regression") 
print("Coefficients:", elastic.coef_) 
print("MSE:", mean_squared_error(y_test, elastic_pred)) 
print("R²:", r2_score(y_test, elastic_pred))
