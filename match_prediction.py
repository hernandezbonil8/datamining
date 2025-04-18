#!/usr/bin/env python3
# Match Prediction Script for Football Clubs
# Student project: merges club & player data, builds logistic regression, and predicts head-to-head

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score
)
import joblib
import argparse

# file paths
CLUBS1 = 'data/2021-2022 Football Team Stats.csv'
CLUBS2 = 'data/Soccer_Football Clubs Ranking in june.csv'
PLAYERS = 'data/players_15.csv'
OUTPUT_DIR = 'match_output'
MODEL_FILE = 'match_prediction_model.pkl'

# 1) load and clean club-level tables
def load_club_data():
    print('Loading club stats...')
    df1 = pd.read_csv(CLUBS1, encoding='latin1', sep=';')
    df1.columns = df1.columns.str.strip()
    # rename squad to club
    if 'Squad' in df1.columns:
        df1.rename(columns={'Squad':'club'}, inplace=True)
    elif 'Club' in df1.columns:
        df1.rename(columns={'Club':'club'}, inplace=True)

    print('Loading club rankings...')
    df2 = pd.read_csv(CLUBS2, encoding='latin1', sep=',', engine='python')
    df2.columns = df2.columns.str.strip().str.replace('\r','')
    # rename possible columns to club
    for c in ['club name','Team','Club']:
        if c in df2.columns:
            df2.rename(columns={c:'club'}, inplace=True)
            break

    merged = pd.merge(df1, df2, on='club', how='inner')
    print(f'Merged club data: {merged.shape[0]} rows, {merged.shape[1]} columns')
    return merged

# 2) load and aggregate player ratings
def load_player_data():
    print('Loading player data...')
    df = pd.read_csv(PLAYERS, encoding='latin1')
    df.columns = df.columns.str.strip()
    if 'club' not in df.columns:
        raise ValueError('players_15.csv missing club column')
    agg = df.groupby('club')[['overall','potential']].mean().reset_index()
    agg.rename(columns={'overall':'avg_overall','potential':'avg_potential'}, inplace=True)
    print(f'Aggregated player ratings: {agg.shape[0]} clubs')
    return agg

# 3) merge all into one df
def merge_all():
    clubs = load_club_data()
    players = load_player_data()
    df = pd.merge(clubs, players, on='club', how='inner')
    # ensure a points column exists
    if 'point score' not in df.columns and 'Pts' in df.columns:
        df.rename(columns={'Pts':'point score'}, inplace=True)
    print(f'Final merged dataset: {df.shape[0]} rows, {df.shape[1]} columns')
    return df

# 4) generate synthetic head-to-head dataset
def make_synthetic(df, feature_cols, save_csv=False):
    print('Creating synthetic match data...')
    data = []
    n = len(df)
    for i in range(n):
        for j in range(i+1, n):
            v1 = df.iloc[i][feature_cols].values.astype(float)
            v2 = df.iloc[j][feature_cols].values.astype(float)
            label = int(df.iloc[i]['point score'] > df.iloc[j]['point score'])
            data.append((v1-v2, label))
            data.append((v2-v1, 1-label))
    X = np.vstack([d[0] for d in data])
    y = np.array([d[1] for d in data])
    print(f'Synthetic data shape: {X.shape}, Labels: {y.shape}')
    if save_csv:
        syn_df = pd.DataFrame(X, columns=feature_cols)
        syn_df['label'] = y
        syn_df.to_csv(os.path.join(OUTPUT_DIR, 'synthetic_match_data.csv'), index=False)
        print('Saved synthetic dataset to synthetic_match_data.csv')
    return X, y

# 5) train logistic regression model
def train_model(X, y, feature_cols, cv_folds=5):
    print('Training logistic regression...')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    model = LogisticRegression(max_iter=1000)
    # cross-validation
    cv_scores = cross_val_score(model, X, y, cv=cv_folds)
    print(f'CV Accuracy scores ({cv_folds}-fold): {cv_scores.round(3)}')
    print(f'Mean CV accuracy: {cv_scores.mean():.3f}')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f'Test Accuracy: {acc:.3f}')
    print('Confusion matrix:\n', confusion_matrix(y_test, y_pred))
    print('Classification report:\n', classification_report(y_test, y_pred))
    # ROC curve
    probs = model.predict_proba(X_test)[:,1]
    fpr, tpr, _ = roc_curve(y_test, probs)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
    plt.plot([0,1],[0,1],'k--')
    plt.title('ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR,'roc_curve.png'))
    plt.close()
    print(f'ROC AUC: {roc_auc:.3f}')
    # Precision-Recall
    precision, recall, _ = precision_recall_curve(y_test, probs)
    ap = average_precision_score(y_test, probs)
    plt.figure()
    plt.plot(recall, precision, label=f'AP = {ap:.2f}')
    plt.title('Precision-Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR,'precision_recall_curve.png'))
    plt.close()
    print(f'Average Precision: {ap:.3f}')
    # coefficients bar chart
    coefs = pd.Series(model.coef_[0], index=feature_cols)
    coefs.sort_values().plot(kind='barh')
    plt.title('Feature Coefficients')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR,'feature_coefs.png'))
    plt.close()
    # save model
    joblib.dump(model, MODEL_FILE)
    print(f'Model saved to {MODEL_FILE}')
    return model

# 6) predict head-to-head match outcome
def predict_match(model, df, feature_cols, c1, c2):
    print(f'Predicting: {c1} vs {c2}')
    r1 = df[df['club']==c1]
    r2 = df[df['club']==c2]
    if r1.empty or r2.empty:
        print('Club name not found!')
        return
    diff = r1.iloc[0][feature_cols].values.astype(float) - r2.iloc[0][feature_cols].values.astype(float)
    prob = model.predict_proba([diff])[0][1]
    winner = c1 if prob>0.5 else c2
    print(f'{winner} wins with probability {max(prob,1-prob):.2f}')
    return winner, max(prob,1-prob)

# 7) Command-line interface
def main():
    parser = argparse.ArgumentParser(description='Football Club Head-to-Head Predictor')
    parser.add_argument('--save-synthetic', action='store_true', help='Save synthetic match dataset to CSV')
    parser.add_argument('--club1', type=str, help='Name of club 1 to predict')
    parser.add_argument('--club2', type=str, help='Name of club 2 to predict')
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = merge_all()
    feature_cols = ['point score','avg_overall','avg_potential']
    X, y = make_synthetic(df, feature_cols, save_csv=args.save_synthetic)
    model = train_model(X, y, feature_cols)

    if args.club1 and args.club2:
        predict_match(model, df, feature_cols, args.club1, args.club2)
    else:
        # example default
        clubs = df['club'].tolist()
        if len(clubs)>=2:
            predict_match(model, df, feature_cols, clubs[0], clubs[1])
    print('Done! All outputs are in', OUTPUT_DIR)

if __name__ == '__main__':
    main()
