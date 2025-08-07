# app.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Page Config
st.set_page_config(page_title="Titanic EDA Dashboard", layout="wide")

# Title
st.title("🚢 Titanic Data Analytics Dashboard")

# Load Data
df = pd.read_csv("cleaned_titanic.csv")

# Show Raw Data Option
if st.checkbox("Show Raw Data"):
    st.dataframe(df)

# Sidebar Filters
st.sidebar.header("🔎 Filter Options")

# Gender
gender = st.sidebar.selectbox("Select Gender", options=["All"] + list(df["Sex"].unique()))

# Passenger Class
pclass = st.sidebar.selectbox("Select Passenger Class", options=["All"] + sorted(df["Pclass"].astype(str).unique()))

# Embarked
embarked = st.sidebar.selectbox("Select Embarked", options=["All"] + list(df["Embarked"].unique()))

# Age Range Slider
min_age = int(df["Age"].min())
max_age = int(df["Age"].max())
age_range = st.sidebar.slider("Select Age Range", min_value=min_age, max_value=max_age, value=(min_age, max_age))

# Fare Range Slider
min_fare = float(df["Fare"].min())
max_fare = float(df["Fare"].max())
fare_range = st.sidebar.slider("Select Fare Range", min_value=float(min_fare), max_value=float(max_fare),
                               value=(float(min_fare), float(max_fare)))

# Apply Filters
filtered_df = df.copy()

if gender != "All":
    filtered_df = filtered_df[filtered_df["Sex"] == gender]
if pclass != "All":
    filtered_df = filtered_df[filtered_df["Pclass"] == int(pclass)]
if embarked != "All":
    filtered_df = filtered_df[filtered_df["Embarked"] == embarked]

# Age and Fare filters
filtered_df = filtered_df[(filtered_df["Age"] >= age_range[0]) & (filtered_df["Age"] <= age_range[1])]
filtered_df = filtered_df[(filtered_df["Fare"] >= fare_range[0]) & (filtered_df["Fare"] <= fare_range[1])]

# Label Survival for plotting
filtered_df["Survived_Label"] = filtered_df["Survived"].map({0: "Did Not Survive", 1: "Survived"})

# Filtered Data Preview
st.subheader("Filtered Data Preview")
st.write(filtered_df.head())

# Check if data is empty
if filtered_df.empty:
    st.warning("⚠️ No data matches the selected filters. Try adjusting the filter values.")
else:
    st.subheader("📊 Visual Analysis (Side-by-Side)")

    # Layout: 3 charts side by side
    col1, col2, col3 = st.columns(3)

    # Chart 1: Survival Count by Gender
    with col1:
        fig1, ax1 = plt.subplots(figsize=(5, 4))
        sns.countplot(data=filtered_df, x="Survived_Label", hue="Sex", ax=ax1)
        ax1.set_title("Survival Count by Gender")
        ax1.set_xlabel("Survival Status")
        ax1.set_ylabel("Count")
        st.pyplot(fig1)

    # Chart 2: Age Distribution by Survival
    with col2:
        fig2, ax2 = plt.subplots(figsize=(5, 4))
        sns.histplot(data=filtered_df, x="Age", hue="Survived_Label", multiple="stack", ax=ax2)
        ax2.set_title("Age Distribution by Survival")
        ax2.set_xlabel("Age")
        ax2.set_ylabel("Count")
        st.pyplot(fig2)

    # Chart 3: Fare Distribution by Class
    with col3:
        fig3, ax3 = plt.subplots(figsize=(5, 4))
        sns.boxplot(data=filtered_df, x="Pclass", y="Fare", ax=ax3)
        ax3.set_title("Fare Distribution by Class")
        ax3.set_xlabel("Passenger Class")
        ax3.set_ylabel("Fare")
        st.pyplot(fig3)
