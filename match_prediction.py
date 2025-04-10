import pandas as pd
import numpy as np
import csv  # for quoting
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# Define file paths (these files should reside in a folder called "data")
CLUBS1_CSV = 'data/2021-2022 Football Team Stats.csv'
CLUBS2_CSV = 'data/Soccer_Football Clubs Ranking in june.csv'
PLAYERS_CSV = 'data/players_15.csv'

# ------------------------------
# 1. Load Club-Level Datasets
# ------------------------------

# Load clubs1_df from 2021-2022 Football Team Stats.csv (assumed semicolon-separated)
clubs1_df = pd.read_csv(CLUBS1_CSV, encoding='latin1', sep=';')

# Load clubs2_df from Soccer_Football Clubs Ranking in june.csv using comma as the delimiter
clubs2_df = pd.read_csv(CLUBS2_CSV, encoding='latin1', sep=',', engine='python')

# Debug print: show clubs2_df columns
print("clubs2_df columns:", clubs2_df.columns.tolist())

# Strip extra whitespace from column names
clubs1_df.columns = clubs1_df.columns.str.strip()
clubs2_df.columns = clubs2_df.columns.str.strip()

# Standardize the club name columns:
# For clubs1_df, assume the club name is in the column "Squad" – rename it to "club"
if 'Squad' in clubs1_df.columns:
    clubs1_df.rename(columns={'Squad': 'club'}, inplace=True)
elif 'Club' in clubs1_df.columns:
    clubs1_df.rename(columns={'Club': 'club'}, inplace=True)
else:
    raise Exception("No appropriate club name column found in clubs1_df.")

# For clubs2_df, we check for "club name", "Team", or "Club" and rename to "club"
if 'club name' in clubs2_df.columns:
    clubs2_df.rename(columns={'club name': 'club'}, inplace=True)
elif 'Team' in clubs2_df.columns:
    clubs2_df.rename(columns={'Team': 'club'}, inplace=True)
elif 'Club' in clubs2_df.columns:
    clubs2_df.rename(columns={'Club': 'club'}, inplace=True)
else:
    raise Exception("No appropriate club name column found in clubs2_df.")

# Merge the two club datasets on the "club" column
clubs_merged = pd.merge(clubs1_df, clubs2_df, on='club', how='inner', suffixes=('_stats', '_rank'))

# -------------------------------------
# 2. Load Player-Level Dataset and Aggregate
# -------------------------------------
players_df = pd.read_csv(PLAYERS_CSV, encoding='latin1')
players_df.columns = players_df.columns.str.strip()

# IMPORTANT: Adjust these column names to match your players dataset.
# In this code, we assume players_15.csv uses 'overall' and 'potential'.
if 'overall' not in players_df.columns or 'potential' not in players_df.columns:
    raise Exception("Expected columns 'overall' and 'potential' not found in players dataset. Please verify your file.")

agg_players = players_df.groupby('club').agg({
    'overall': 'mean',
    'potential': 'mean'
}).reset_index()
agg_players.rename(columns={
    'overall': 'avg_overall_rating',
    'potential': 'avg_potential'
}, inplace=True)

# Merge aggregated player data with the club-level merged data
merged_df = pd.merge(clubs_merged, agg_players, on='club', how='inner')

# --------------------------
# 3. Define Performance Metric
# --------------------------
# We'll use "point score" as our performance metric. Check if it exists.
if 'point score' not in merged_df.columns:
    print("Column 'point score' not found. Attempting to use 'Pts' instead.")
    if 'Pts' in merged_df.columns:
        merged_df.rename(columns={'Pts': 'point score'}, inplace=True)
    else:
        raise Exception("No column for team points found. Please verify your datasets.")

# --------------------------
# 4. Define Feature Columns
# --------------------------
# We'll use the club-level point score, average overall rating, and average potential.
feature_cols = ['point score', 'avg_overall_rating', 'avg_potential']

# =============================================================================
# 5. Create Synthetic Match-Level Dataset
# For every pair of clubs (i, j), create a training example:
#   - Input: difference between their feature vectors (club_i - club_j).
#   - Label: 1 if club_i's point score > club_j's, else 0.
# Also create the reverse matchup (club_j - club_i) with the opposite label.
# =============================================================================
synthetic_data = []
num_clubs = merged_df.shape[0]

for i in range(num_clubs):
    for j in range(i + 1, num_clubs):
        team_i_features = merged_df.loc[i, feature_cols].values.astype(float)
        team_j_features = merged_df.loc[j, feature_cols].values.astype(float)
        
        # Create example: club_i minus club_j
        diff_features = team_i_features - team_j_features
        label = 1 if merged_df.loc[i, 'point score'] > merged_df.loc[j, 'point score'] else 0
        synthetic_data.append((diff_features, label))
        
        # Also create inverse example: club_j minus club_i
        diff_features_rev = team_j_features - team_i_features
        label_rev = 1 - label
        synthetic_data.append((diff_features_rev, label_rev))

X_synthetic = np.array([x for x, y in synthetic_data])
y_synthetic = np.array([y for x, y in synthetic_data])
print("Synthetic dataset shape:", X_synthetic.shape)

# -----------------------------------------------------------
# 6. Split Data and Train a Logistic Regression Model
# -----------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(X_synthetic, y_synthetic, test_size=0.2, random_state=42)

match_model = LogisticRegression(max_iter=1000)
match_model.fit(X_train, y_train)

y_pred = match_model.predict(X_test)
print("Synthetic Test Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Save the trained model to disk
joblib.dump(match_model, 'match_prediction_model.pkl')
print("Match prediction model saved as 'match_prediction_model.pkl'.")

# =============================================================================
# 7. Function for Predicting Head-to-Head Match Outcome
# =============================================================================
def predict_match(club1, club2):
    """
    Given two club names, predict which club is more likely to win based on the difference
    in their feature vectors.
    """
    row1 = merged_df[merged_df['club'] == club1]
    row2 = merged_df[merged_df['club'] == club2]
    
    if row1.empty or row2.empty:
        raise Exception("One or both club names not found in the merged dataset.")
    
    team1_features = row1.iloc[0][feature_cols].values.astype(float)
    team2_features = row2.iloc[0][feature_cols].values.astype(float)
    
    diff = team1_features - team2_features
    prob = match_model.predict_proba([diff])[0][1]
    print(f"Predicted probability that {club1} wins over {club2}: {prob:.2f}")
    if prob > 0.5:
        print(f"{club1} is predicted to win against {club2}.")
        return club1, prob
    else:
        print(f"{club2} is predicted to win against {club1}.")
        return club2, 1 - prob

# Example usage: Predict outcome between the first two clubs in merged_df
if merged_df.shape[0] >= 2:
    club_a = merged_df['club'].iloc[0]
    club_b = merged_df['club'].iloc[1]
    print("\nExample Match Prediction:")
    winner, win_prob = predict_match(club_a, club_b)
    print(f"Result: {winner} is predicted to win with probability {win_prob:.2f}")
else:
    print("Not enough clubs in the dataset for a match prediction.")
