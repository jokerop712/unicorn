import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# Load dataset
df = pd.read_csv("Ecommerce Customers.csv")


# Define features and target
features = [
    'Avg. Session Length',
    'Time on App',
    'Time on Website',
    'Length of Membership'
]
label = "Yearly Amount Spent"

X = df[features]
y = df[label]


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.4,
    random_state=42
)


# Train Linear Regression model
regression_model = LinearRegression()
regression_model.fit(X_train, y_train)


# Save the trained model
with open("model.pkl", "wb") as file:
    pickle.dump(regression_model, file)


print("Model training complete and saved as model.pkl")
