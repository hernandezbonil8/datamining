import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the dataset with the correct encoding and delimiter
data = pd.read_csv('2021-2022 Football Team Stats.csv', encoding='latin1', sep=';')

# Create a binary target variable "High_Performing"
# Teams with wins greater than the median are labeled 1 (high performing), others 0.
median_wins = data['W'].median()
data['High_Performing'] = (data['W'] > median_wins).astype(int)

# Define the feature columns based on available dataset columns
# Here we use Goals For (GF), Goals Against (GA), Points per Game (Pts/G), and Matches Played (MP)
feature_columns = ['GF', 'GA', 'Pts/G', 'MP']
X = data[feature_columns]
y = data['High_Performing']

# Split the data: 70% training, 15% validation, and 15% test
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Function to train and evaluate a model on the validation set
def train_and_evaluate(model, model_name):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    print(f"{model_name} Accuracy on Validation Set: {accuracy_score(y_val, y_pred):.2f}")
    print(f"Classification Report for {model_name}:\n", classification_report(y_val, y_pred))
    print(f"Confusion Matrix for {model_name}:\n", confusion_matrix(y_val, y_pred))
    print("------------------------------------------------------")

# Logistic Regression
lr_model = LogisticRegression(max_iter=1000)
train_and_evaluate(lr_model, "Logistic Regression")

# Decision Tree Classifier
dt_model = DecisionTreeClassifier(random_state=42)
train_and_evaluate(dt_model, "Decision Tree")

# K-Nearest Neighbors
knn_model = KNeighborsClassifier()
train_and_evaluate(knn_model, "K-Nearest Neighbors")

# Evaluate the best model on the test set. Here we assume Logistic Regression is best.
best_model = lr_model
best_model.fit(X_train, y_train)
y_test_pred = best_model.predict(X_test)
print("Final Evaluation on Test Set (Logistic Regression):")
print("Accuracy:", accuracy_score(y_test, y_test_pred))
print("Classification Report:\n", classification_report(y_test, y_test_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_test_pred))
