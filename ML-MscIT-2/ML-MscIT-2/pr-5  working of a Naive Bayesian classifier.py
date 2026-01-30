from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics


# Load Iris dataset
iris = load_iris()

# Feature matrix (X) and target vector (y)
X = iris.data
y = iris.target


# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.4,
    random_state=1
)


# Train the Gaussian Naive Bayes model
gnb = GaussianNB()
gnb.fit(X_train, y_train)


# Make predictions
y_pred = gnb.predict(X_test)


# Evaluate the model
accuracy = metrics.accuracy_score(y_test, y_pred) * 100
print("Gaussian Naive Bayes model accuracy (in %):", accuracy)
