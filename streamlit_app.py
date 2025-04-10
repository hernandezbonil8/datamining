import pandas as pd
import joblib
import streamlit as st

# Load the dataset (ensure the CSV file is in the same directory)
data = pd.read_csv('2021-2022 Football Team Stats.csv', encoding='latin1', sep=';')

# Load the trained best model
model = joblib.load('best_model.pkl')

# Define the feature columns used in model_training.py
feature_columns = ['GF', 'GA', 'Pts/G', 'MP']

# Streamlit app title and description
st.title("Head-to-Head Team Outcome Predictor")
st.write("Select two teams to predict which team is more likely to be 'high performing' based on season stats.")

# Create dropdowns for team selection from the 'Squad' column
teams = data['Squad'].tolist()
team1 = st.selectbox("Select Team 1", teams, index=0)
team2 = st.selectbox("Select Team 2", teams, index=1)

if st.button("Predict Head-to-Head Outcome"):
    # Get the features for the selected teams
    team1_row = data[data['Squad'] == team1].iloc[0]
    team2_row = data[data['Squad'] == team2].iloc[0]
    team1_features = team1_row[feature_columns].tolist()
    team2_features = team2_row[feature_columns].tolist()
    
    # Predict probabilities using the best model
    prob_team1 = model.predict_proba([team1_features])[0][1]
    prob_team2 = model.predict_proba([team2_features])[0][1]
    
    st.write(f"**{team1}** High Performing Probability: **{prob_team1:.2f}**")
    st.write(f"**{team2}** High Performing Probability: **{prob_team2:.2f}**")
    
    if prob_team1 > prob_team2:
        st.success(f"{team1} is more likely to win!")
    elif prob_team2 > prob_team1:
        st.success(f"{team2} is more likely to win!")
    else:
        st.info("Both teams are equally matched.")
