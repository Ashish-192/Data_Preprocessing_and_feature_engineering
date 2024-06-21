import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('/content/titanic.csv')
# Display the first few rows of the dataset
data.head()
data.info()
# Fill missing values
data['Age'].fillna(data['Age'].median(), inplace=True)
data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)

# Feature engineering
# Create new features
data['FamilySize'] = data['SibSp'] + data['Parch'] + 1

# Categorical features encoding
data = pd.get_dummies(data, columns=['Embarked', 'Sex'], drop_first=True)

# Drop unnecessary columns
data.drop(['Name', 'Ticket', 'Cabin', 'PassengerId', 'SibSp', 'Parch'], axis=1, inplace=True)

# Visualize the results
sns.barplot(x='Pclass', y='Survived', data=data) 
plt.show()
sns.scatterplot(x='Age', y='Fare', hue='Survived', data=data)
plt.figure(figsize=(10, 8))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.show()
