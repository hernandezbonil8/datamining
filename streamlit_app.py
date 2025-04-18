
import os
import pandas as pd
import numpy as np
import streamlit as st
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# -- Constants & Paths --
DATA_DIR = 'data'
CLUBS1_CSV = os.path.join(DATA_DIR, '2021-2022 Football Team Stats.csv')
CLUBS2_CSV = os.path.join(DATA_DIR, 'Soccer_Football Clubs Ranking in june.csv')
PLAYERS_CSV = os.path.join(DATA_DIR, 'players_15.csv')
MODEL_FILE = 'match_prediction_model.pkl'
FEATURE_COLS = ['point score', 'avg_overall_rating', 'avg_potential']

@st.cache_data
def load_merged_data():
    # Load club stats
    c1 = pd.read_csv(CLUBS1_CSV, encoding='latin1', sep=';')
    c1.columns = c1.columns.str.strip()
    if 'Squad' in c1.columns:
        c1 = c1.rename(columns={'Squad':'club'})
    else:
        c1 = c1.rename(columns={'Club':'club'})
    
    # Load club rankings
    c2 = pd.read_csv(CLUBS2_CSV, encoding='latin1', sep=',', engine='python')
    c2.columns = c2.columns.str.strip().str.replace('\r','')
    for col in ['club name','Team','Club']:
        if col in c2.columns:
            c2 = c2.rename(columns={col:'club'})
            break

    # Merge club-level
    merged = pd.merge(c1, c2, on='club', how='inner', suffixes=('_stats','_rank'))

    # Load and aggregate player data
    p = pd.read_csv(PLAYERS_CSV, encoding='latin1')
    p.columns = p.columns.str.strip()
    p_agg = p.groupby('club')[['overall','potential']].mean().reset_index()
    p_agg = p_agg.rename(columns={'overall':'avg_overall_rating','potential':'avg_potential'})

    # Final merge
    final = pd.merge(merged, p_agg, on='club', how='inner')
    # Rename Pts to point score
    if 'Pts' in final.columns and 'point score' not in final.columns:
        final = final.rename(columns={'Pts':'point score'})
    return final

@st.cache_resource
def load_model():
    return joblib.load(MODEL_FILE)

# -- App Layout --
st.set_page_config(page_title='Football Match Predictor', layout='wide')
st.title('⚽️ Head-to-Head Football Match Outcome Predictor')
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
    st.table(t1.set_index('club').T)
with col2:
    st.subheader(f"Stats: {team2}")
    t2 = df[df['club']==team2][FEATURE_COLS + ['club']]
    st.table(t2.set_index('club').T)

# Predict button
if st.button('Predict Outcome'):
    # Compute feature difference
    v1 = t1[FEATURE_COLS].values.flatten().astype(float)
    v2 = t2[FEATURE_COLS].values.flatten().astype(float)
    diff = v1 - v2
    prob = model.predict_proba([diff])[0][1]
    winner = team1 if prob>0.5 else team2
    win_prob = prob if prob>0.5 else 1-prob

    # Display result
    st.success(f"**Predicted Winner:** {winner}")
    st.info(f"Win Probability: {win_prob:.2%}")

    # Difference bar chart
    diff_df = pd.DataFrame({'feature':FEATURE_COLS, 'difference': diff})
    plt.figure(figsize=(6,4))
    sns.barplot(x='difference', y='feature', data=diff_df, palette='viridis')
    plt.axvline(0, color='black', linestyle='--')
    plt.title('Feature Differences (Team1 - Team2)')
    st.pyplot(plt)

# Additional exploration
st.sidebar.header('Additional Options')
if st.sidebar.checkbox('Show Raw Data'):
    st.subheader('Merged Dataset')
    st.dataframe(df)
if st.sidebar.checkbox('Download Model'):
    with open(MODEL_FILE, 'rb') as f:
        st.download_button('Download Trained Model', f, file_name='match_prediction_model.pkl')


