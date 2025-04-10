import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib  # for saving the model

# Load the dataset
data = pd.read_csv('2021-2022 Football Team Stats.csv', encoding='latin1', sep=';')

# For this example, we create a binary target "High_Performing" based on whether wins ('W') exceed the median value.
median_wins = data['W'].median()
data['High_Performing'] = (data['W'] > median_wins).astype(int)

# Define feature columns (adjust these as needed)
feature_columns = ['GF', 'GA', 'Pts/G', 'MP']
X = data[feature_columns]
y = data['High_Performing']

# Split the data: 70% training, 15% validation, 15% test
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

def train_and_evaluate(model, model_name):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    print(f"{model_name} Accuracy on Validation Set: {accuracy:.2f}")
    print(f"Classification Report for {model_name}:\n", classification_report(y_val, y_pred))
    print(f"Confusion Matrix for {model_name}:\n", confusion_matrix(y_val, y_pred))
    print("------------------------------------------------------")
    return accuracy

# Train and evaluate baseline models
lr_model = LogisticRegression(max_iter=1000)
dt_model = DecisionTreeClassifier(random_state=42)
knn_model = KNeighborsClassifier()

acc_lr = train_and_evaluate(lr_model, "Logistic Regression")
acc_dt = train_and_evaluate(dt_model, "Decision Tree")
acc_knn = train_and_evaluate(knn_model, "K-Nearest Neighbors")

# Choose the best model (here we assume Logistic Regression is best if its accuracy is highest)
if acc_lr >= acc_dt and acc_lr >= acc_knn:
    best_model = lr_model
    best_name = "Logistic Regression"
elif acc_dt >= acc_lr and acc_dt >= acc_knn:
    best_model = dt_model
    best_name = "Decision Tree"
else:
    best_model = knn_model
    best_name = "K-Nearest Neighbors"

print(f"Best model on validation set: {best_name}")

# Evaluate the best model on the test set
best_model.fit(X_train, y_train)
y_test_pred = best_model.predict(X_test)
print("\nFinal Evaluation on Test Set:")
print("Accuracy:", accuracy_score(y_test, y_test_pred))
print("Classification Report:\n", classification_report(y_test, y_test_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_test_pred))

# Save the best model to disk
joblib.dump(best_model, 'best_model.pkl')
print("\nBest model saved as 'best_model.pkl'.")

# --- Head-to-Head Prediction Function ---
def predict_head_to_head(team1_features, team2_features, model):
    """
    Given the feature vectors for two teams, predict and compare their probabilities
    of being 'high performing'. Returns a string with the result.
    """
    # Assumes model.predict_proba returns probability for class 1 (high performing) at index 1.
    prob_team1 = model.predict_proba([team1_features])[0][1]
    prob_team2 = model.predict_proba([team2_features])[0][1]
    print(f"Team 1 Win Probability: {prob_team1:.2f}")
    print(f"Team 2 Win Probability: {prob_team2:.2f}")
    if prob_team1 > prob_team2:
        return "Team 1 is more likely to win."
    elif prob_team2 > prob_team1:
        return "Team 2 is more likely to win."
    else:
        return "Both teams are equally matched."

# For demonstration, predict head-to-head for two teams from the dataset:
# Let's choose team rows based on index; adjust indices or selection as needed.
team1_features = data.loc[0, feature_columns].tolist()  # e.g., first team
team2_features = data.loc[1, feature_columns].tolist()  # e.g., second team
print("\nHead-to-Head Prediction between Team 1 and Team 2:")
print(predict_head_to_head(team1_features, team2_features, best_model))
