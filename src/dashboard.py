import streamlit as st
import pandas as pd
import plotly.express as px
import folium
from streamlit_folium import folium_static
from folium.plugins import MarkerCluster

st.set_page_config(layout="wide", page_title="Dublin Urban Forest Analytics")

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    div[data-testid="stHorizontalBlock"] {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    h1, h2, h3 {
        padding-top: 1rem;
        padding-bottom: 0.5rem;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    return pd.read_parquet("../data/processed_dublin_trees.parquet")

df = load_data()

st.title("🌳 Dublin Urban Forest Analytics")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Trees", f"{len(df):,}")
with col2:
    st.metric("Species Count", f"{df['species_desc'].nunique():,}")
with col3:
    healthy_trees = len(df[df['health_index'] >= 2])
    st.metric("Healthy Trees", f"{healthy_trees:,}")
with col4:
    avg_health = df['health_index'].mean()
    st.metric("Average Health", f"{avg_health:.2f}")

st.sidebar.header("📊 Filters")
with st.sidebar:
    st.markdown("### Filter Data")
    selected_species = st.multiselect(
        "Select Tree Species",
        options=sorted(df['species_desc'].unique()),
        default=df['species_desc'].value_counts().head().index.tolist()
    )
    
    selected_condition = st.multiselect(
        "Select Tree Condition",
        options=sorted(df['condition'].unique()),
        default=df['condition'].unique()
    )

# Filtered data
filtered_df = df[
    (df['species_desc'].isin(selected_species) if selected_species else True) &
    (df['condition'].isin(selected_condition) if selected_condition else True)
]

tab1, tab2, tab3 = st.tabs(["📊 Distribution", "🗺️ Map View", "📈 Analysis"])

with tab1:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Species Distribution")
        species_counts = filtered_df['species_desc'].value_counts().head(10)
        fig_species = px.bar(
            species_counts,
            orientation='h',
            title="Top 10 Tree Species",
            labels={'value': 'Count', 'index': 'Species'}
        )
        fig_species.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig_species, use_container_width=True)
    
    with col2:
        st.subheader("Health Condition Distribution")
        condition_counts = filtered_df['condition'].value_counts()
        fig_condition = px.pie(
            values=condition_counts.values,
            names=condition_counts.index,
            title="Tree Health Distribution"
        )
        fig_condition.update_layout(height=400)
        st.plotly_chart(fig_condition, use_container_width=True)

with tab2:
    st.subheader("Tree Distribution Map")
    m = folium.Map(location=[53.3498, -6.2603], zoom_start=11)
    
    marker_cluster = MarkerCluster().add_to(m)
    for _, row in filtered_df.sample(min(len(filtered_df), 1000)).iterrows():
        folium.CircleMarker(
            location=[row['lat'], row['long']],
            radius=3,
            popup=f"Species: {row['species_desc']}<br>Condition: {row['condition']}",
            color="green",
            fill=True
        ).add_to(marker_cluster)
    
    folium_static(m)

with tab3:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Age Distribution")
        age_counts = filtered_df['age_category'].value_counts()
        fig_age = px.bar(
            age_counts,
            title="Tree Age Distribution",
            labels={'value': 'Count', 'index': 'Age Category'}
        )
        fig_age.update_layout(height=400)
        st.plotly_chart(fig_age, use_container_width=True)
    
    with col2:
        st.subheader("Health by Area")
        health_by_area = filtered_df.groupby('town')['health_index'].mean().sort_values(ascending=False).head(10)
        fig_health = px.bar(
            health_by_area,
            title="Average Health Index by Area",
            labels={'value': 'Health Index', 'index': 'Area'}
        )
        fig_health.update_layout(height=400)
        st.plotly_chart(fig_health, use_container_width=True)
