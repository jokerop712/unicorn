import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets


# Parameters
n_neighbors = 15
h = 0.02  # Step size in the mesh


# Load Iris dataset
iris = datasets.load_iris()

# Use only first two features
X = iris.data[:, :2]
y = iris.target


# Create color maps
cmap_light = ListedColormap(["orange", "cyan", "cornflowerblue"])
cmap_bold = ["darkorange", "c", "darkblue"]


# Train and plot for different weight types
for weights in ["uniform", "distance"]:

    # Create KNN classifier
    clf = neighbors.KNeighborsClassifier(
        n_neighbors=n_neighbors,
        weights=weights
    )
    clf.fit(X, y)

    # Create mesh grid
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, h),
        np.arange(y_min, y_max, h)
    )

    # Predict for mesh grid
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot decision boundary
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, cmap=cmap_light)

    # Plot training points
    sns.scatterplot(
        x=X[:, 0],
        y=X[:, 1],
        hue=iris.target_names[y],
        palette=cmap_bold,
        alpha=1.0,
        edgecolor="black"
    )

    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())

    plt.title(
        "3-Class classification (k = %i, weights = '%s')"
        % (n_neighbors, weights)
    )

    plt.xlabel(iris.feature_names[0])
    plt.ylabel(iris.feature_names[1])


plt.show()
