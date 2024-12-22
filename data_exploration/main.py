import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

st.title("Data Explorer")

# File uploader
uploaded_file = st.file_uploader("Upload CSV File", type="csv")

# Initialize DataFrame
df = None
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("File uploaded successfully!")
    st.write(f"DataFrame shape: {df.shape}")

if df is not None:
    # Show basic data exploration options
    st.subheader("Data Exploration")

    if st.button("Show First Rows"):
        st.write(df.head())

    if st.button("Show Last Rows"):
        st.write(df.tail())

    if st.button("Show Data Types"):
        st.write(df.dtypes)

    if st.button("Show Statistical Summary"):
        st.write(df.describe())

    if st.button("Show Missing Values"):
        st.write(df.isnull().sum())

    if st.button("Show Correlation Matrix"):
        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.empty:
            st.write("No numeric columns available for correlation.")
        else:
            corr = numeric_df.corr()
            st.write(corr)
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', ax=ax)
            st.pyplot(fig)

    # Column-based analysis
    columns = df.columns.tolist()
    if columns:
        st.subheader("Column-Based Analysis")

        # Value Counts
        col_val_counts = st.selectbox("Select a column for Value Counts:", columns)
        if st.button("Show Value Counts"):
            st.write(df[col_val_counts].value_counts())

        # Unique Values
        col_unique = st.selectbox("Select a column for Unique Values:", columns)
        if st.button("Show Unique Values"):
            uniques = df[col_unique].unique()
            st.write(f"Unique values in '{col_unique}':", uniques)

        # Histogram
        col_hist = st.selectbox("Select a column for Histogram:", columns)
        if st.button("Show Histogram"):
            fig, ax = plt.subplots(figsize=(8,6))
            sns.histplot(df[col_hist].dropna(), kde=True, ax=ax)
            ax.set_title(f"Histogram of {col_hist}")
            st.pyplot(fig)

        # Boxplot
        col_box = st.selectbox("Select a column for Boxplot:", columns)
        if st.button("Show Boxplot"):
            if df[col_box].dtype not in [np.float64, np.int64, np.float32, np.int32]:
                st.write(f"The selected column '{col_box}' is not numeric. Please select a numeric column.")
            else:
                fig, ax = plt.subplots(figsize=(8,6))
                sns.boxplot(y=df[col_box].dropna(), ax=ax)
                ax.set_title(f"Boxplot of {col_box}")
                st.pyplot(fig)
