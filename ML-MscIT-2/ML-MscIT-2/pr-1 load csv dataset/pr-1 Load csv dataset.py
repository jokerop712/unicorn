import pandas as pd

# Load dataset
df = pd.read_csv("Student_data.csv")

# Basic exploration
df.head()
df.shape   # (10, 7)

df.info()
df.describe()

# Check missing values
df.isnull().sum()


# Fill missing values
df['Age'] = df['Age'].fillna(df['Age'].mean())
df['Math_Score'] = df['Math_Score'].fillna(df['Math_Score'].mean())
df['Science_Score'] = df['Science_Score'].fillna(df['Science_Score'].mean())
df['English_Score'] = df['English_Score'].fillna(df['English_Score'].mean())
df['Gender'] = df['Gender'].fillna(df['Gender'].mode()[0])

# Verify missing values are handled
df.isnull().sum()


# Feature and target separation
X = df.drop(['Student_ID', 'Passed'], axis=1)
y = df['Passed']


# Encode categorical variables
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df['Gender'] = le.fit_transform(df['Gender'])
df['Passed'] = le.fit_transform(df['Passed'])


# Feature scaling
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df[['Age', 'Math_Score', 'Science_Score', 'English_Score']] = scaler.fit_transform(
    df[['Age', 'Math_Score', 'Science_Score', 'English_Score']]
)


# Binarization
from sklearn.preprocessing import Binarizer

binarizer = Binarizer(threshold=0)
df['Math_Binary'] = binarizer.fit_transform(df[['Math_Score']])


# Data visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Histogram
plt.hist(df['Math_Score'])
plt.xlabel("Math Score")
plt.ylabel("Frequency")
plt.title("Distribution of Math Scores")
plt.show()

# Box plot
sns.boxplot(x=df['Science_Score'])
plt.show()

# Scatter plot
plt.scatter(df['Math_Score'], df['Science_Score'])
plt.xlabel("Math Score")
plt.ylabel("Science Score")
plt.title("Math vs Science Scores")
plt.show()

# Count plot
sns.countplot(x='Passed', data=df)
plt.show()

# Correlation heatmap
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.show()

# Final data preview
df.head()
