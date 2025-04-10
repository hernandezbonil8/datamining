import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset with appropriate encoding and delimiter
data = pd.read_csv('2021-2022 Football Team Stats.csv', encoding='latin1', sep=';')

# Display basic information
print("First 5 rows:")
print(data.head())
print("\nDataset Info:")
print(data.info())

# Print summary statistics for numeric columns only
numeric_data = data.select_dtypes(include=['int64', 'float64'])
print("\nSummary Statistics (Numeric Columns):")
print(numeric_data.describe())

# Check for missing values
print("\nMissing values in each column:")
print(data.isnull().sum())

# Save a histogram for a key numeric feature, e.g., 'GF' (Goals For)
if 'GF' in data.columns:
    plt.figure(figsize=(8, 6))
    sns.histplot(data['GF'], kde=True)
    plt.title('Distribution of Goals For (GF)')
    plt.xlabel('Goals For')
    plt.ylabel('Frequency')
    plt.savefig('goals_for_histogram.png')
    plt.close()
    print("\nSaved histogram for 'GF' as 'goals_for_histogram.png'.")

# Plot and save the correlation matrix using numeric columns only
plt.figure(figsize=(10, 8))
corr_matrix = numeric_data.corr()
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Correlation Matrix')
plt.savefig('correlation_matrix.png')
plt.close()
print("\nSaved correlation matrix as 'correlation_matrix.png'.")
