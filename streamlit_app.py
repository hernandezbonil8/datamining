import streamlit as st
import pandas as pd
import joblib

# Define file paths for datasets; ensure they are in the 'data' folder.
CLUBS1_CSV = 'data/2021-2022 Football Team Stats.csv'
CLUBS2_CSV = 'data/Soccer_Football Clubs Ranking in june.csv'
PLAYERS_CSV = 'data/players_15.csv'

def load_merged_data():
    # Load the club-level datasets
    clubs1_df = pd.read_csv(CLUBS1_CSV, encoding='latin1', sep=';')
    clubs2_df = pd.read_csv(CLUBS2_CSV, encoding='latin1', sep=',', engine='python')
    
    # Strip whitespace from column names
    clubs1_df.columns = clubs1_df.columns.str.strip()
    clubs2_df.columns = clubs2_df.columns.str.strip()
    
    # Rename club name columns
    if 'Squad' in clubs1_df.columns:
        clubs1_df.rename(columns={'Squad': 'club'}, inplace=True)
    else:
        clubs1_df.rename(columns={'Club': 'club'}, inplace=True)
    
    if 'club name' in clubs2_df.columns:
        clubs2_df.rename(columns={'club name': 'club'}, inplace=True)
    elif 'Team' in clubs2_df.columns:
        clubs2_df.rename(columns={'Team': 'club'}, inplace=True)
    else:
        raise Exception("No appropriate club name column found in clubs2_df.")
    
    # Merge club data
    clubs_merged = pd.merge(clubs1_df, clubs2_df, on='club', how='inner', suffixes=('_stats', '_rank'))
    
    # Load and aggregate player-level data
    players_df = pd.read_csv(PLAYERS_CSV, encoding='latin1')
    players_df.columns = players_df.columns.str.strip()
    agg_players = players_df.groupby('club').agg({
        'overall': 'mean',      # Change to your column name if needed
        'potential': 'mean'     # Change to your column name if needed
    }).reset_index()
    agg_players.rename(columns={
        'overall': 'avg_overall_rating',
        'potential': 'avg_potential'
    }, inplace=True)
    
    # Merge with club data
    merged_df = pd.merge(clubs_merged, agg_players, on='club', how='inner')
    
    # Use performance metric "point score" or "Pts"
    if 'point score' not in merged_df.columns:
        if 'Pts' in merged_df.columns:
            merged_df.rename(columns={'Pts': 'point score'}, inplace=True)
        else:
            st.error("No performance metric found in the club data!")
    return merged_df

# Load merged data and define features
merged_df = load_merged_data()
feature_cols = ['point score', 'avg_overall_rating', 'avg_potential']

# Load the trained model
model = joblib.load('match_prediction_model.pkl')

def main():
    st.title("Head-to-Head Football Match Outcome Predictor")
    st.write("Select two clubs to predict the match outcome based on combined club and player statistics.")
    
    clubs = merged_df['club'].tolist()
    clubs_sorted = sorted(clubs)
    club1 = st.selectbox("Select Club 1", clubs_sorted, index=0)
    club2 = st.selectbox("Select Club 2", clubs_sorted, index=1)
    
    if st.button("Predict Outcome"):
        row1 = merged_df[merged_df['club'] == club1]
        row2 = merged_df[merged_df['club'] == club2]
        if row1.empty or row2.empty:
            st.error("One or both club names not found in the dataset.")
        else:
            team1_features = row1.iloc[0][feature_cols].values.astype(float)
            team2_features = row2.iloc[0][feature_cols].values.astype(float)
            diff = team1_features - team2_features
            prob = model.predict_proba([diff])[0][1]
            if prob > 0.5:
                winner = club1
                win_prob = prob
            else:
                winner = club2
                win_prob = 1 - prob
            st.success(f"Predicted Winner: **{winner}**")
            st.write(f"Win Probability: **{win_prob:.2f}**")

if __name__ == "__main__":
    main()
