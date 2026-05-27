
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

# -----------------------------
# PAGE CONFIG
# -----------------------------

st.set_page_config(
    page_title="Data Quality Dashboard",
    layout="wide"
)

st.title("📊 Data Quality Dashboard")
st.markdown("### Titanic Dataset Quality Assessment")

# -----------------------------
# LOAD DATA
# -----------------------------

df = pd.read_csv("dirty_titanic_dataset.csv")

# -----------------------------
# METRICS
# -----------------------------

total_cells = np.prod(df.shape)

missing_values = df.isnull().sum()

total_missing = missing_values.sum()

missing_percentage = (
    missing_values / len(df)
) * 100

duplicate_count = df.duplicated().sum()

completeness_score = (
    (total_cells - total_missing)
    / total_cells
) * 100

duplicate_penalty = (
    duplicate_count / len(df)
) * 100

quality_score = round(
    max(0, completeness_score - duplicate_penalty),
    2
)

# -----------------------------
# KPI CARDS
# -----------------------------

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Overall DQ Score",
    f"{quality_score}%"
)

col2.metric(
    "Missing Values",
    int(total_missing)
)

col3.metric(
    "Duplicate Rows",
    int(duplicate_count)
)

col4.metric(
    "Total Rows",
    len(df)
)

st.divider()

# -----------------------------
# MISSING VALUES TABLE
# -----------------------------

st.subheader("Missing Values Analysis")

missing_df = pd.DataFrame({
    "Column": df.columns,
    "Missing Values": missing_values.values,
    "Missing Percentage": missing_percentage.values
})

st.dataframe(missing_df)

# -----------------------------
# MISSING VALUES BAR CHART
# -----------------------------

fig1 = px.bar(
    missing_df,
    x="Column",
    y="Missing Percentage",
    title="Missing Values Percentage by Column",
    color="Missing Percentage"
)

st.plotly_chart(fig1, use_container_width=True)

# -----------------------------
# HEATMAP
# -----------------------------

st.subheader("Missing Values Heatmap")

fig, ax = plt.subplots(figsize=(12,6))

sns.heatmap(
    df.isnull(),
    cbar=False,
    ax=ax
)

st.pyplot(fig)

# -----------------------------
# AGE DISTRIBUTION
# -----------------------------

st.subheader("Age Distribution")

fig2 = px.histogram(
    df,
    x="Age",
    nbins=30,
    title="Age Distribution"
)

st.plotly_chart(fig2, use_container_width=True)

# -----------------------------
# FARE BOXPLOT
# -----------------------------

st.subheader("Fare Distribution")

fig3 = px.box(
    df,
    y="Fare",
    title="Fare Box Plot"
)

st.plotly_chart(fig3, use_container_width=True)

# -----------------------------
# INVALID AGE RECORDS
# -----------------------------

st.subheader("Invalid Age Records")

invalid_age = df[
    (df['Age'] < 0) |
    (df['Age'] > 100)
]

st.dataframe(invalid_age)

# -----------------------------
# NEGATIVE FARE RECORDS
# -----------------------------

st.subheader("Negative Fare Records")

negative_fare = df[
    df['Fare'] < 0
]

st.dataframe(negative_fare)

# -----------------------------
# PROBLEMATIC ROWS
# -----------------------------

st.subheader("Rows with Missing Values")

problematic_rows = df[
    df.isnull().any(axis=1)
]

st.dataframe(problematic_rows.head(20))

# -----------------------------
# BUSINESS SUMMARY
# -----------------------------

st.subheader("Business Insights")

st.markdown(f"""
- Overall dataset quality score is **{quality_score}%**
- Dataset contains **{total_missing} missing values**
- Duplicate records detected: **{duplicate_count}**
- Cabin column has severe incompleteness
- Invalid age values indicate data entry issues
- Negative fare values suggest corruption or validation failures
""")
