import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
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

# Constants
data_path = 'data/2021-2022 Football Team Stats.csv'
output_dir = 'baseline_output'
best_model_file = os.path.join(output_dir, 'best_model.pkl')

# 1) Load and prepare data
def load_and_prepare():
    df = pd.read_csv(data_path, encoding='latin1', sep=';')
    # create binary target: High_Performing if W > median(W)
    median_w = df['W'].median()
    df['High_Performing'] = (df['W'] > median_w).astype(int)
    # features to use
    features = ['GF', 'GA', 'Pts/G', 'MP']
    X = df[features].values
    y = df['High_Performing'].values
    return df, X, y, features

# 2) Train & evaluate baseline models
def evaluate_classifiers(X_train, y_train, X_val, y_val):
    models = {
        'LogisticRegression': LogisticRegression(max_iter=1000),
        'DecisionTree': DecisionTreeClassifier(random_state=42),
        'KNN': KNeighborsClassifier()
    }
    val_scores = {}
    for name, model in models.items():
        # cross-validation on training data
        cv_scores = cross_val_score(model, X_train, y_train, cv=5)
        print(f"{name} CV acc: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")
        # train on train, evaluate on val
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        acc = accuracy_score(y_val, y_pred)
        print(f"{name} Val acc: {acc:.3f}")
        val_scores[name] = (acc, model)
        # detailed metrics
        print(classification_report(y_val, y_pred))
        print(confusion_matrix(y_val, y_pred))
        print('-'*40)
    # pick best by validation accuracy
    best_name = max(val_scores, key=lambda k: val_scores[k][0])
    print(f"Best model: {best_name} with acc={val_scores[best_name][0]:.3f}")
    return best_name, val_scores[best_name][1]

# 3) Test best model
def test_best_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print("\n--- Final Test Evaluation ---")
    print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.3f}")
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
    # ROC & PR curves
    probs = model.predict_proba(X_test)[:,1]
    fpr, tpr, _ = roc_curve(y_test, probs)
    roc_auc = auc(fpr, tpr)
    plt.figure(); plt.plot(fpr, tpr, label=f'AUC={roc_auc:.2f}'); plt.plot([0,1],[0,1],'k--')
    plt.title('ROC Curve'); plt.xlabel('FPR'); plt.ylabel('TPR'); plt.legend()
    plt.savefig(os.path.join(output_dir, 'roc_curve.png'))
    plt.close()
    precision, recall, _ = precision_recall_curve(y_test, probs)
    ap = average_precision_score(y_test, probs)
    plt.figure(); plt.plot(recall, precision, label=f'AP={ap:.2f}');
    plt.title('Precision-Recall Curve'); plt.xlabel('Recall'); plt.ylabel('Precision'); plt.legend()
    plt.savefig(os.path.join(output_dir, 'pr_curve.png'))
    plt.close()
    print(f"ROC AUC={roc_auc:.3f}, Avg Precision={ap:.3f}")

# 4) Head-to-head predictor
def predict_head_to_head(df, features, model, t1_idx=0, t2_idx=1):
    row1 = df.iloc[t1_idx]
    row2 = df.iloc[t2_idx]
    x1 = row1[features].values.reshape(1,-1)
    x2 = row2[features].values.reshape(1,-1)
    p1 = model.predict_proba(x1)[0,1]
    p2 = model.predict_proba(x2)[0,1]
    print(f"Team1 ({row1['Squad']}) prob high-perform: {p1:.2f}")
    print(f"Team2 ({row2['Squad']}) prob high-perform: {p2:.2f}")
    if p1>p2: return f"{row1['Squad']} likely better." 
    elif p2>p1: return f"{row2['Squad']} likely better." 
    else: return "Evenly matched."

# 5) Main CLI
def main():
    os.makedirs(output_dir, exist_ok=True)
    df, X, y, features = load_and_prepare()
    # 70/15/15 split
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    best_name, best_model = evaluate_classifiers(X_train, y_train, X_val, y_val)
    test_best_model(best_model, X_test, y_test)
    joblib.dump(best_model, best_model_file)
    print(f"Saved best model to {best_model_file}")

    # demonstration head-to-head
    result = predict_head_to_head(df, features, best_model)
    print("Head-to-Head result:", result)

if __name__ == '__main__':
    main()
