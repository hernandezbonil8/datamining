
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# 1) Load and prepare data

def load_data(path):
    print(f"Loading data from {path}")
    df = pd.read_csv(path, encoding='latin1', sep=';')
    # Rename squad column if present
    if 'Squad' in df.columns:
        df = df.rename(columns={'Squad': 'Club'})
    # Create a few basic stats if possible
    if 'GF' in df.columns and 'GA' in df.columns:
        df['GoalDiff'] = df['GF'] - df['GA']
    if {'W','D','L'}.issubset(df.columns):
        df['WinRate'] = df['W'] / (df['W'] + df['D'] + df['L'])
    if 'MP' in df.columns and 'GF' in df.columns:
        df['GoalsPerGame'] = df['GF'] / df['MP']
        df['ConcedePerGame'] = df['GA'] / df['MP']
    return df

# 2) Exploratory data analysis

def explore_data(df):
    print("\n=== EXPLORATORY DATA ANALYSIS ===")
    print(f"Shape: {df.shape}")
    print("Columns and types:")
    print(df.dtypes)

    numeric = df.select_dtypes(include=['int64','float64'])
    print("\nHead:\n", df.head())
    print("\nSummary stats for numeric columns:")
    print(numeric.describe())

    # Check missing
    missing = df.isnull().sum()
    if missing.any():
        print("\nMissing values found:")
        print(missing[missing>0])
        for col in missing[missing>0].index:
            if df[col].dtype in ['float64','int64']:
                df[col].fillna(df[col].median(), inplace=True)
            else:
                df[col].fillna(df[col].mode()[0], inplace=True)
        print("Missing values filled.")
    else:
        print("\nNo missing values.")
    return numeric

# 3) Visualizations

def make_plots(df, numeric_df, out_dir='output'):
    os.makedirs(out_dir, exist_ok=True)
    print(f"\nGenerating plots in {out_dir}")

    # Goals histogram
    if 'GF' in df.columns:
        plt.figure(figsize=(8,6))
        sns.histplot(df['GF'], kde=True)
        plt.title('Goals For (GF) Distribution')
        plt.savefig(os.path.join(out_dir, 'gf_hist.png'))
        plt.close()

    # Correlation heatmap
    plt.figure(figsize=(10,8))
    corr = numeric_df.corr()
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.savefig(os.path.join(out_dir, 'corr_matrix.png'))
    plt.close()

    # Radar chart for top teams by points
    if 'Pts' in df.columns:
        top = df.nlargest(8, 'Pts')
        # pick metrics
        metrics = ['Pts', 'GF', 'GA']
        stats = top[metrics]
        stats_norm = (stats - stats.min())/(stats.max()-stats.min())
        labels = metrics + [metrics[0]]
        angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]
        fig, ax = plt.subplots(subplot_kw=dict(polar=True))
        for i, club in enumerate(top['Club']):
            vals = stats_norm.iloc[i].tolist()
            vals += vals[:1]
            ax.plot(angles, vals, label=club)
            ax.fill(angles, vals, alpha=0.1)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics)
        plt.legend(bbox_to_anchor=(1.1,1.05))
        plt.title('Top 8 Teams Performance Radar')
        plt.savefig(os.path.join(out_dir, 'radar_chart.png'))
        plt.close()

# 4) Clustering with KMeans

def cluster_teams(df, out_dir='output'):
    print("\nClustering teams...")
    feats = [c for c in ['Pts','GF','GA','WinRate','GoalsPerGame','GoalDiff'] if c in df.columns]
    if len(feats) < 2:
        print("Not enough features for clustering.")
        return df
    X = df[feats].dropna()
    X_scaled = StandardScaler().fit_transform(X)
    # elbow
    inertias = []
    ks = range(1,6)
    for k in ks:
        km = KMeans(n_clusters=k, random_state=0).fit(X_scaled)
        inertias.append(km.inertia_)
    plt.figure()
    plt.plot(ks, inertias, 'o-')
    plt.xlabel('k')
    plt.ylabel('Inertia')
    plt.title('Elbow Plot')
    plt.savefig(os.path.join(out_dir,'elbow.png'))
    plt.close()
    k_opt = 3
    km = KMeans(n_clusters=k_opt, random_state=0).fit(X_scaled)
    df['Cluster'] = km.labels_
    # PCA plot
    pca = PCA(2)
    coords = pca.fit_transform(X_scaled)
    plt.figure()
    for c in sorted(df['Cluster'].unique()):
        mask = df['Cluster']==c
        plt.scatter(coords[mask,0], coords[mask,1], label=f'Cluster {c}')
    plt.legend()
    plt.title('Team Clusters (PCA)')
    plt.savefig(os.path.join(out_dir,'clusters.png'))
    plt.close()
    return df

# 5) Simple regression model

def train_model(df, out_dir='output'):
    print("\nTraining regression model...")
    if 'Pts' not in df.columns:
        print("No Pts column.")
        return None
    cols = df.select_dtypes(include=['float64','int64']).columns.drop('Pts')
    X = df[cols].fillna(0)
    y = df['Pts']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    model = LinearRegression().fit(X_train, y_train)
    preds = model.predict(X_test)
    r2 = r2_score(y_test, preds)
    rmse = mean_squared_error(y_test, preds, squared=False)
    print(f"R2={r2:.3f}, RMSE={rmse:.3f}")
    plt.figure()
    plt.scatter(y_test, preds)
    plt.plot([y_test.min(),y_test.max()],[y_test.min(),y_test.max()], 'r--')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('Actual vs Predicted Points')
    plt.savefig(os.path.join(out_dir,'regression.png'))
    plt.close()
    return model

# 6) Main pipeline

def main():
    data_file = 'data/2021-2022 Football Team Stats.csv'
    out_dir = 'output'
    df = load_data(data_file)
    num_df = explore_data(df)
    make_plots(df, num_df, out_dir)
    df = cluster_teams(df, out_dir)
    model = train_model(df, out_dir)
    print("\nDone! Check the output folder.")

if __name__ == '__main__':
    main()
