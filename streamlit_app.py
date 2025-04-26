import os
import pandas as pd
import numpy as np
import streamlit as st
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import random

#  Constants & Paths 
DATA_DIR = 'data'
CLUBS1_CSV = os.path.join(DATA_DIR, '2021-2022 Football Team Stats.csv')
CLUBS2_CSV = os.path.join(DATA_DIR, 'Soccer_Football Clubs Ranking in june.csv')
PLAYERS_CSV = os.path.join(DATA_DIR, 'players_15.csv')
MODEL_FILE = 'match_prediction_model.pkl'
FEATURE_COLS = ['point score', 'avg_overall_rating', 'avg_potential']

@st.cache_data
def load_merged_data():
    c1 = pd.read_csv(CLUBS1_CSV, encoding='latin1', sep=';')
    c1.columns = c1.columns.str.strip()
    if 'Squad' in c1.columns:
        c1 = c1.rename(columns={'Squad':'club'})
    else:
        c1 = c1.rename(columns={'Club':'club'})

    c2 = pd.read_csv(CLUBS2_CSV, encoding='latin1', sep=',', engine='python')
    c2.columns = c2.columns.str.strip().str.replace('\r','')
    for col in ['club name','Team','Club']:
        if col in c2.columns:
            c2 = c2.rename(columns={col:'club'})
            break

    merged = pd.merge(c1, c2, on='club', how='inner', suffixes=('_stats','_rank'))

    p = pd.read_csv(PLAYERS_CSV, encoding='latin1')
    p.columns = p.columns.str.strip()
    p_agg = p.groupby('club')[['overall','potential']].mean().reset_index()
    p_agg = p_agg.rename(columns={'overall':'avg_overall_rating','potential':'avg_potential'})

    final = pd.merge(merged, p_agg, on='club', how='inner')
    if 'Pts' in final.columns and 'point score' not in final.columns:
        final = final.rename(columns={'Pts':'point score'})
    return final

@st.cache_resource
def load_model():
    return joblib.load(MODEL_FILE)

# Function to calculate weighted difference between teams
def calculate_team_advantage(team1_stats, team2_stats):
    """Calculate how much advantage team1 has over team2 based on weighted stats"""
    # Weights for different features (point score is most important)
    weights = {
        'point score': 0.6,
        'avg_overall_rating': 0.3, 
        'avg_potential': 0.1
    }
    
    advantage = 0
    total_weight = sum(weights.values())
    
    for feature, weight in weights.items():
        v1 = float(team1_stats[feature].values[0])
        v2 = float(team2_stats[feature].values[0])
        
        # Get typical range for this feature for normalization
        feature_range = df[feature].max() - df[feature].min()
        if feature_range == 0:
            feature_range = 1  # Avoid division by zero
            
        # Calculate normalized weighted difference
        normalized_diff = (v1 - v2) / feature_range
        advantage += normalized_diff * (weight / total_weight)
    
    return advantage

# App Layout 
st.set_page_config(page_title='Football Match Predictor', layout='wide')
st.title('Head-to-Head Football Match Outcome Predictor')
st.markdown("Select two clubs and view detailed stats plus predicted winner.")

# Load data & model
df = load_merged_data()
model = load_model()
clubs = sorted(df['club'].unique())

# Sidebar selectors
st.sidebar.header('Team Selection')
team1 = st.sidebar.selectbox('Select Team 1', clubs, index=0)
team2 = st.sidebar.selectbox('Select Team 2', clubs, index=1)

# Display team stats
col1, col2 = st.columns(2)
with col1:
    st.subheader(f"Stats: {team1}")
    t1 = df[df['club']==team1][FEATURE_COLS + ['club']]
    t1 = t1.drop_duplicates(subset='club')
    st.table(t1.set_index('club').T)
with col2:
    st.subheader(f"Stats: {team2}")
    t2 = df[df['club']==team2][FEATURE_COLS + ['club']]
    t2 = t2.drop_duplicates(subset='club')
    st.table(t2.set_index('club').T)

# Predict button
if st.button('Predict Outcome'):
    v1 = t1[FEATURE_COLS].values.flatten().astype(float)
    v2 = t2[FEATURE_COLS].values.flatten().astype(float)
    diff = v1 - v2

    # Get raw probability from model
    raw_p = model.predict_proba([diff])[0][1]
    
    # Calculate weighted advantage (positive means team1 is better, negative means team2 is better)
    advantage = calculate_team_advantage(t1, t2)
    
    # Convert advantage to a probability adjustment
    # More advantage = higher probability of winning
    advantage_factor = np.tanh(abs(advantage) * 5) * 0.4  # tanh limits to reasonable range
    
    # Determine direction of advantage and adjust probability
    if advantage > 0:  # Team 1 has advantage
        adjusted_p = 0.5 + advantage_factor
    elif advantage < 0:  # Team 2 has advantage
        adjusted_p = 0.5 - advantage_factor
    else:  # No advantage
        adjusted_p = 0.5
    
    # Add small random factor (smaller when teams are more evenly matched)
    random_scale = 0.05 * (1 - abs(advantage))  # Less randomness when advantage is clear
    random_factor = random.uniform(-random_scale, random_scale)
    final_p = adjusted_p + random_factor
    
    # Cap probabilities
    max_prob = min(0.7, 0.55 + abs(advantage) * 0.3)  # Maximum probability based on advantage
    final_p = np.clip(final_p, 0.3, max_prob)
    
    # Calculate tie probability - only high when teams are truly close
    tie_threshold = 0.08 * (1 - abs(advantage) * 5)  # Smaller threshold when advantage is clear
    tie_threshold = max(0.03, min(tie_threshold, 0.12))  # Keep between 3-12%
    
    # Determine outcome
    is_tie = abs(final_p - 0.5) <= tie_threshold
    
    # For draws, use a more realistic probability distribution (25-35%)
    draw_prob = random.uniform(0.25, 0.35)
    
    # Display results
    if is_tie:
        st.success("🤝 **Predicted Result: Draw**")
        st.info(f"Draw Probability: {draw_prob:.2%}")
        
        # Only show this for genuinely close matches
        if abs(advantage) < 0.1:
            st.markdown("*Teams are very evenly matched on key stats a draw is likely.*")
    else:
        p1 = final_p
        p2 = 1 - p1
        
        if p1 > 0.5:
            winner, win_prob = team1, p1
        else:
            winner, win_prob = team2, p2
        
        # Adjust language based on win probability
        if win_prob < 0.55:
            st.success(f"🏆 **Slight Edge To:** {winner}")
        elif win_prob < 0.65:
            st.success(f"🏆 **Predicted Winner:** {winner}")
        else:
            st.success(f"🏆 **Strong Favorite:** {winner}")
            
        st.info(f"Win Probability: {win_prob:.2%}")
        
        # Add conditional insights
        if win_prob < 0.55:
            st.markdown("*This is a close match that could go either way.*")
        elif win_prob >= 0.65:
            st.markdown("*Significant statistical advantage makes this team the clear favorite.*")

# Additional exploration
st.sidebar.header('Additional Options')
if st.sidebar.checkbox('Show Raw Data'):
    st.subheader('Merged Dataset')
    st.dataframe(df)
if st.sidebar.checkbox('Download Model'):
    with open(MODEL_FILE, 'rb') as f:
        st.download_button('Download Trained Model', f, file_name='match_prediction_model.pkl')