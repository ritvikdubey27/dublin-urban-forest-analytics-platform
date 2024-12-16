import streamlit as st
import pandas as pd
import plotly.express as px
import folium
from streamlit_folium import folium_static

# Load processed data
@st.cache_data
def load_data():
    return pd.read_parquet("../data/processed_dublin_trees.parquet")

df = load_data()

st.title("Dublin Urban Forest Analytics Dashboard")

# Sidebar for filtering
st.sidebar.header("Filters")
selected_species = st.sidebar.multiselect("Select Tree Species", df['species_desc'].unique())
selected_condition = st.sidebar.multiselect("Select Tree Condition", df['condition'].unique())

# Filter data based on selections
filtered_df = df[
    (df['species_desc'].isin(selected_species) if selected_species else True) &
    (df['condition'].isin(selected_condition) if selected_condition else True)
]

# Species Distribution
st.header("Species Distribution")
species_counts = filtered_df['species_desc'].value_counts().head(10)
fig_species = px.bar(species_counts, x=species_counts.index, y=species_counts.values)
st.plotly_chart(fig_species)

# Health Condition Distribution
st.header("Health Condition Distribution")
condition_counts = filtered_df['condition'].value_counts()
fig_condition = px.pie(values=condition_counts.values, names=condition_counts.index)
st.plotly_chart(fig_condition)

# Age Distribution
st.header("Age Distribution")
age_counts = filtered_df['age_category'].value_counts()
fig_age = px.bar(age_counts, x=age_counts.index, y=age_counts.values)
st.plotly_chart(fig_age)

# Tree Map
st.header("Tree Distribution Map")
m = folium.Map(location=[53.3498, -6.2603], zoom_start=11)
for _, row in filtered_df.iterrows():
    folium.CircleMarker(
        location=[row['lat'], row['long']],
        radius=3,
        popup=f"Species: {row['species_desc']}<br>Condition: {row['condition']}",
        color="green",
        fill=True
    ).add_to(m)
folium_static(m)

# Health by Area
st.header("Average Health Index by Area")
health_by_area = filtered_df.groupby('town')['health_index'].mean().sort_values(ascending=False).head(10)
fig_health_area = px.bar(health_by_area, x=health_by_area.index, y=health_by_area.values)
st.plotly_chart(fig_health_area)
