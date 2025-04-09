import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the dataset
df = pd.read_csv('diabetes.csv')
print("Dataset loaded successfully!")
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns

# Show the first few rows of data
print(df.head())

# Plot class distribution
sns.countplot(x='Outcome', data=df)
plt.title("Diabetes Class Distribution (0 = No, 1 = Yes)")
plt.show()

# Plot correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.show()

# Compare glucose levels by outcome
sns.boxplot(x='Outcome', y='Glucose', data=df)
plt.title("Glucose Levels by Diabetes Status")
plt.show()

# Split data into features and target
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train Decision Tree model
tree_model = DecisionTreeClassifier(random_state=42)
tree_model.fit(X_train, y_train)

# Make predictions
tree_predictions = tree_model.predict(X_test)

# Evaluate Decision Tree model
tree_accuracy = accuracy_score(y_test, tree_predictions)
print(f"Decision Tree Accuracy: {tree_accuracy * 100:.2f}%")

# Create and train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Evaluate model
accuracy = accuracy_score(y_test, predictions)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Predict diabetes for a new patient using the Decision Tree model
new_patient = [[2, 70, 70, 25, 80, 20.0, 0.5, 25]]
prediction = tree_model.predict(new_patient)

# Print the result
if int(prediction[0]) == 1:
    print("Prediction for new patient: Diabetic 🩺\n")

else:

    print("Prediction for new patient: Not Diabetic ✅\n")
