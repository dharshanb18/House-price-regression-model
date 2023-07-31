import csv
import pandas as pd
from sklearn.linear_model import LinearRegression

# Read the data from the csv file.
with open('medals.csv', 'r') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    data = list(reader)

# Create a Pandas DataFrame from the data.
df = pd.DataFrame(data)

# Select the features that you want to use to predict the medals.
features = ['Country', 'Year', 'Training', 'Funding', 'Natural Talent']

# Create a target variable that indicates whether the country won a medal.
target = df['Medal']

# Split the data into a training set and a test set.
X_train, X_test, y_train, y_test = train_test_split(df[features], target, test_size=0.25)

# Create a linear regression model.
model = LinearRegression()

# Train the model on the training data.
model.fit(X_train, y_train)

# Predict the medals for the test set.
predictions = model.predict(X_test)

# Calculate the accuracy of the model.
accuracy = accuracy_score(y_test, predictions)

print('Accuracy:', accuracy)
