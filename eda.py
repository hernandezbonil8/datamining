import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset using the correct encoding and delimiter
data = pd.read_csv('2021-2022 Football Team Stats.csv', encoding='latin1', sep=';')

# Print the first 5 rows
print("First 5 rows:")
print(data.head())

# Print dataset info
print("\nDataset Info:")
print(data.info())

# Print summary statistics for numeric columns
print("\nSummary Statistics (Numeric Columns):")
numeric_data = data.select_dtypes(include=['int64', 'float64'])
print(numeric_data.describe())

# Check for missing values in all columns
print("\nMissing values in each column:")
print(data.isnull().sum())

# Generate and save a histogram for a key numeric feature, e.g., 'GF' (Goals For)
if 'GF' in data.columns:
    plt.figure(figsize=(8, 6))
    sns.histplot(data['GF'], kde=True)
    plt.title('Distribution of Goals For (GF)')
    plt.xlabel('Goals For')
    plt.ylabel('Frequency')
    plt.savefig('goals_for_histogram.png')
    plt.close()
    print("\nSaved histogram for 'GF' as 'goals_for_histogram.png'.")

# Generate and save a correlation matrix heatmap using numeric data only
plt.figure(figsize=(10, 8))
corr_matrix = numeric_data.corr()
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Correlation Matrix')
plt.savefig('correlation_matrix.png')
plt.close()
print("\nSaved correlation matrix as 'correlation_matrix.png'.")
