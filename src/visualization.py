import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium.plugins import HeatMap
import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import count, desc

def create_spark_session():
    """Initialize Spark Session"""
    return SparkSession.builder \
        .appName("DublinTreesVisualization") \
        .getOrCreate()

def load_processed_data(spark):
    """Load processed data from parquet file"""
    return spark.read.parquet("../data/processed_dublin_trees.parquet")

def create_species_distribution_plot(df):
    """Create species distribution visualization"""
    plt.figure(figsize=(12, 6))
    species_counts = df.groupBy("species_desc") \
        .agg(count("*").alias("count")) \
        .orderBy(desc("count")) \
        .limit(10) \
        .toPandas()
    
    sns.barplot(data=species_counts, x="count", y="species_desc")
    plt.title("Top 10 Tree Species Distribution in Dublin")
    plt.xlabel("Number of Trees")
    plt.ylabel("Species")
    plt.tight_layout()
    plt.savefig("../visualizations/species_distribution.png")
    plt.close()

def create_health_condition_plot(df):
    """Create health condition visualization"""
    plt.figure(figsize=(10, 6))
    condition_counts = df.groupBy("condition") \
        .agg(count("*").alias("count")) \
        .toPandas()
    
    plt.pie(condition_counts["count"], 
            labels=condition_counts["condition"], 
            autopct='%1.1f%%')
    plt.title("Tree Health Condition Distribution")
    plt.savefig("../visualizations/health_condition.png")
    plt.close()

def create_age_distribution_plot(df):
    """Create age distribution visualization"""
    plt.figure(figsize=(10, 6))
    age_counts = df.groupBy("age_category") \
        .agg(count("*").alias("count")) \
        .toPandas()
    
    sns.barplot(data=age_counts, x="age_category", y="count")
    plt.title("Tree Age Distribution")
    plt.xlabel("Age Category")
    plt.ylabel("Number of Trees")
    plt.savefig("../visualizations/age_distribution.png")
    plt.close()

def create_tree_map(df):
    """Create interactive map visualization"""
    # Convert to pandas for mapping
    tree_locations = df.select("lat", "long", "species_desc", "condition") \
        .toPandas()
    
    # Create base map centered on Dublin
    dublin_map = folium.Map(
        location=[53.3498, -6.2603],
        zoom_start=11
    )
    
    # Add tree markers
    for idx, row in tree_locations.iterrows():
        folium.CircleMarker(
            location=[row["lat"], row["long"]],
            radius=3,
            popup=f"Species: {row['species_desc']}<br>Condition: {row['condition']}",
            color="green",
            fill=True
        ).add_to(dublin_map)
    
    # Create heatmap layer
    heat_data = tree_locations[["lat", "long"]].values.tolist()
    HeatMap(heat_data).add_to(dublin_map)
    
    dublin_map.save("../visualizations/tree_distribution_map.html")

def create_health_by_area_plot(df):
    """Create health by area visualization"""
    plt.figure(figsize=(12, 6))
    health_by_area = df.groupBy("town") \
        .agg({"health_index": "avg"}) \
        .orderBy(desc("avg(health_index)")) \
        .limit(10) \
        .toPandas()
    
    sns.barplot(data=health_by_area, 
                x="town", 
                y="avg(health_index)")
    plt.xticks(rotation=45)
    plt.title("Average Tree Health Index by Area")
    plt.tight_layout()
    plt.savefig("../visualizations/health_by_area.png")
    plt.close()

def main():
    # Create visualizations directory
    os.makedirs("visualizations", exist_ok=True)
    
    # Initialize Spark and load data
    spark = create_spark_session()
    df = load_processed_data(spark)
    
    # Create visualizations
    print("Creating species distribution plot...")
    create_species_distribution_plot(df)
    
    print("Creating health condition plot...")
    create_health_condition_plot(df)
    
    print("Creating age distribution plot...")
    create_age_distribution_plot(df)
    
    print("Creating tree distribution map...")
    create_tree_map(df)
    
    print("Creating health by area plot...")
    create_health_by_area_plot(df)
    
    print("All visualizations created successfully!")
    spark.stop()

if __name__ == "__main__":
    main()
