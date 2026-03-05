import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

st.title("Agentic AI AutoEDA Dashboard")

# Load dataset
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
data = pd.read_csv(url)

st.write("Dataset Preview")
st.dataframe(data.head())

st.write("Dataset Shape")
st.write(data.shape)

st.write("Missing Values")
st.write(data.isnull().sum())

st.write("Statistical Summary")
st.write(data.describe())

# Correlation heatmap
st.write("Correlation Heatmap")
fig, ax = plt.subplots()
sns.heatmap(data.corr(numeric_only=True), annot=True, cmap="coolwarm", ax=ax)
st.pyplot(fig)

# Histogram plots
st.write("Feature Distributions")
fig2 = data.hist(figsize=(10,8))
st.pyplot(plt.gcf())
