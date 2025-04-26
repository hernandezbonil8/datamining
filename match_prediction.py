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
    if 'Squad' in df1.columns:
        df1.rename(columns={'Squad':'club'}, inplace=True)
    else:
        df1.rename(columns={'Club':'club'}, inplace=True)

    print('Loading club rankings...')
    df2 = pd.read_csv(CLUBS2, encoding='latin1', sep=',', engine='python')
    df2.columns = df2.columns.str.strip().str.replace('\r','')
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
    agg = df.groupby('club')[['overall','potential']].mean().reset_index()
    agg.rename(columns={'overall':'avg_overall_rating','potential':'avg_potential'}, inplace=True)
    print(f'Aggregated player ratings: {agg.shape[0]} clubs')
    return agg

# 3) merge all into one df
def merge_all():
    clubs = load_club_data()
    players = load_player_data()
    df = pd.merge(clubs, players, on='club', how='inner')
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
            try:
                v1 = df.iloc[i][feature_cols].values.astype(float)
                v2 = df.iloc[j][feature_cols].values.astype(float)
                label = int(df.iloc[i]['point score'] > df.iloc[j]['point score'])
                data.append((v1-v2, label))
                data.append((v2-v1, 1-label))
            except Exception as e:
                print(f"Error processing clubs {df.iloc[i]['club']} vs {df.iloc[j]['club']}: {e}")
                continue
                
    X = np.vstack([d[0] for d in data])
    y = np.array([d[1] for d in data])
    print(f'Synthetic data shape: {X.shape}, Labels: {y.shape}')
    if save_csv:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        syn_df = pd.DataFrame(X, columns=feature_cols)
        syn_df['label'] = y
        syn_df.to_csv(os.path.join(OUTPUT_DIR, 'synthetic_match_data.csv'), index=False)
        print('Saved synthetic dataset to synthetic_match_data.csv')
    return X, y

# 5) train logistic regression model
def train_model(X, y, feature_cols, cv_folds=5):
    print('Training logistic regression...')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    model = LogisticRegression(max_iter=1000, random_state=42)
    cv_scores = cross_val_score(model, X, y, cv=cv_folds)
    print(f'CV Accuracy scores ({cv_folds}-fold): {cv_scores.round(3)}')
    print(f'Mean CV accuracy: {cv_scores.mean():.3f}')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f'Test Accuracy: {accuracy_score(y_test, y_pred):.3f}')
    print('Confusion matrix:\n', confusion_matrix(y_test, y_pred))
    print('Classification report:\n', classification_report(y_test, y_pred))
    
    # Save coefficients and feature importance
    try:
        coef_df = pd.DataFrame({
            'Feature': feature_cols,
            'Coefficient': model.coef_[0]
        })
        coef_df['AbsImportance'] = abs(coef_df['Coefficient'])
        coef_df = coef_df.sort_values('AbsImportance', ascending=False)
        print("Feature importance:")
        print(coef_df)
        
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        coef_df.to_csv(os.path.join(OUTPUT_DIR, 'feature_importance.csv'), index=False)
    except Exception as e:
        print(f"Couldn't save feature importance: {e}")
    
    # ROC curve
    try:
        y_probs = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_probs)
        roc_auc = auc(fpr, tpr)
        print(f'ROC AUC: {roc_auc:.3f}')
        
        # Save the model
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        joblib.dump(model, MODEL_FILE)
        print(f'Model saved to {MODEL_FILE}')
    except Exception as e:
        print(f"Error during evaluation: {e}")
    
    return model

# 6) predict head-to-head match outcome
def predict_match(model, df, feature_cols, c1, c2):
    print(f'Predicting: {c1} vs {c2}')
    r1 = df[df['club']==c1]
    r2 = df[df['club']==c2]
    if r1.empty:
        print(f'Club not found: {c1}')
        print(f'Available clubs: {", ".join(sorted(df["club"].unique()[:10]))}...')
        return None, None
    if r2.empty:
        print(f'Club not found: {c2}')
        print(f'Available clubs: {", ".join(sorted(df["club"].unique()[:10]))}...')
        return None, None
    
    try:
        diff = r1.iloc[0][feature_cols].values.astype(float) - r2.iloc[0][feature_cols].values.astype(float)
        # compute both direction probabilities
        prob1 = model.predict_proba([diff])[0][1]
        prob2 = model.predict_proba([(-diff).tolist()])[0][1]
        
        # Print features for inspection
        print(f"\nTeam comparison ({c1} vs {c2}):")
        for i, feat in enumerate(feature_cols):
            print(f"{feat}: {r1.iloc[0][feat]:.2f} vs {r2.iloc[0][feat]:.2f}, diff: {diff[i]:.2f}")
        
        tie_threshold = 0.05
        if abs(prob1 - 0.5) <= tie_threshold and abs(prob2 - 0.5) <= tie_threshold:
            print('Head-to-Head result: Evenly matched.')
            return None, 0.5
            
        if prob1 > prob2:
            winner, win_prob = c1, prob1
        else:
            winner, win_prob = c2, prob2
            
        print(f'{winner} wins with probability {win_prob:.2f}')
        return winner, win_prob
    except Exception as e:
        print(f"Error during prediction: {e}")
        return None, None

# 7) Command-line interface
def main():
    parser = argparse.ArgumentParser(description='Football Club Head-to-Head Predictor')
    parser.add_argument('--save-synthetic', action='store_true', help='Save synthetic match data to CSV')
    parser.add_argument('--club1', type=str, help='First club name')
    parser.add_argument('--club2', type=str, help='Second club name')
    parser.add_argument('--train-only', action='store_true', help='Only train model without prediction')
    parser.add_argument('--predict-only', action='store_true', help='Only predict using existing model')
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load data
    df = merge_all()
    feature_cols = ['point score', 'avg_overall_rating', 'avg_potential']
    
    # Check if we're only predicting using existing model
    if args.predict_only:
        if not os.path.exists(MODEL_FILE):
            print(f"Error: Model file {MODEL_FILE} not found. Train a model first.")
            return
        model = joblib.load(MODEL_FILE)
        if args.club1 and args.club2:
            predict_match(model, df, feature_cols, args.club1, args.club2)
        else:
            clubs = df['club'].unique().tolist()
            if len(clubs) >= 2:
                predict_match(model, df, feature_cols, clubs[0], clubs[1])
        return
    
    # Generate synthetic data and train model
    X, y = make_synthetic(df, feature_cols, save_csv=args.save_synthetic)
    model = train_model(X, y, feature_cols)

    # Skip prediction if train-only
    if args.train_only:
        print('Training completed. Skipping prediction.')
        return
    
    # Make predictions if clubs specified
    if args.club1 and args.club2:
        predict_match(model, df, feature_cols, args.club1, args.club2)
    else:
        clubs = df['club'].unique().tolist()
        if len(clubs) >= 2:
            predict_match(model, df, feature_cols, clubs[0], clubs[1])
    
    print('Done! All outputs are in', OUTPUT_DIR)

if __name__ == '__main__':
    main()