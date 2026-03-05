import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

print("Agentic AI AutoEDA Started")

# Step 1: Load Dataset
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
data = pd.read_csv(url)

print("\nDataset Loaded Successfully")

# Step 2: Observe Dataset
print("\nFirst 5 Rows")
print(data.head())

print("\nDataset Shape")
print(data.shape)

print("\nColumn Names")
print(data.columns)

# Step 3: Statistical Summary
print("\nStatistical Summary")
print(data.describe())

# Step 4: Missing Values
print("\nMissing Values")
print(data.isnull().sum())

# Step 5: Correlation Analysis
plt.figure(figsize=(10,6))
sns.heatmap(data.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()

# Step 6: Distribution Plots
data.hist(figsize=(10,8))
plt.show()

print("\nAgent Insight:")
print("EDA completed automatically by Agentic AI.")
