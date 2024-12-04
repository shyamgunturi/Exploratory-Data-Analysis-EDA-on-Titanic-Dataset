import pandas as pd

# Load the Titanic dataset
data = pd.read_csv('train.csv')

# Display the first few rows of the dataset
print(data.head())

print(data.isnull().sum())

print(data.describe())

# Fill missing 'Age' with the median age
data['Age'].fillna(data['Age'].median(), inplace=True)

# Drop rows with missing values in 'Embarked' column
data.dropna(subset=['Embarked'], inplace=True)

import seaborn as sns
import matplotlib.pyplot as plt

# Count plot for survival
sns.countplot(x='Survived', data=data)
plt.title('Survival Distribution')
plt.show()

sns.barplot(x='Sex', y='Survived', data=data)
plt.title('Survival by Gender')
plt.show()

sns.barplot(x='Pclass', y='Survived', data=data)
plt.title('Survival by Passenger Class')
plt.show()

sns.histplot(data[data['Survived'] == 1]['Age'], kde=True, color='blue', label='Survived')
sns.histplot(data[data['Survived'] == 0]['Age'], kde=True, color='red', label='Did not survive')
plt.legend()
plt.title('Age Distribution by Survival')
plt.show()
