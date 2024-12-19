from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, round, avg, when, desc, sum, log
from pyspark.sql.window import Window
import matplotlib.pyplot as plt
import seaborn as sns

def create_spark_session():
    return SparkSession.builder \
        .appName("DublinTreesAnalysis") \
        .config("spark.driver.memory", "4g") \
        .getOrCreate()

def analyze_species_distribution(df):
    """Analyze species distribution and diversity"""
    # Calculating species distribution
    species_dist = df.groupBy("species_desc") \
        .agg(count("*").alias("count")) \
        .orderBy(desc("count"))
    
    # Calculating diversity metrics
    total_trees = df.count()
    species_richness = species_dist.count()
    
    # Calculating Shannon diversity index
    species_dist = species_dist.withColumn(
        "proportion", col("count") / total_trees
    ).withColumn(
        "shannon_component", 
        (col("proportion") * log(col("proportion"))).alias("shannon_component")
    )
    
    shannon_index = -species_dist.agg(sum("shannon_component")).collect()[0][0]
    
    return species_dist, species_richness, shannon_index

def analyze_health_condition(df):
    """Analyze tree health conditions"""
    health_stats = df.groupBy("condition") \
        .agg(
            count("*").alias("count"),
            round(avg("health_index"), 2).alias("avg_health_index")
        ).orderBy(desc("count"))
    
    # Calculating health metrics by area
    health_by_town = df.groupBy("town") \
        .agg(
            round(avg("health_index"), 2).alias("avg_health_index"),
            count("*").alias("tree_count")
        ).orderBy(desc("avg_health_index"))
    
    return health_stats, health_by_town

def analyze_spatial_patterns(df):
    """Analyze spatial distribution patterns"""
    # Calculating tree density by area
    density_by_town = df.groupBy("town") \
        .agg(count("*").alias("tree_count")) \
        .orderBy(desc("tree_count"))
    
    # Analyzing species clustering
    species_by_town = df.groupBy("town", "species_desc") \
        .agg(count("*").alias("count")) \
        .orderBy("town", desc("count"))
    
    return density_by_town, species_by_town

def analyze_age_distribution(df):
    """Analyze age distribution of trees"""
    age_dist = df.groupBy("age_category") \
        .agg(
            count("*").alias("count"),
            round(avg("health_index"), 2).alias("avg_health_index")
        ).orderBy("age_category")
    
    # Age distribution by species
    age_by_species = df.groupBy("species_desc", "age_category") \
        .agg(count("*").alias("count")) \
        .orderBy("species_desc", "age_category")
    
    return age_dist, age_by_species

def main():
    spark = create_spark_session()
    
    df = spark.read.parquet("../data/processed_dublin_trees.parquet")
    
    # Performing analyses
    species_dist, species_richness, shannon_index = analyze_species_distribution(df)
    health_stats, health_by_town = analyze_health_condition(df)
    density_by_town, species_by_town = analyze_spatial_patterns(df)
    age_dist, age_by_species = analyze_age_distribution(df)

    print(f"\nSpecies Richness: {species_richness}")
    print(f"Shannon Diversity Index: {shannon_index:.2f}")

    print("\nSpecies Distribution:")
    species_dist.show(5)
    
    print("\nHealth Conditions:")
    health_stats.show()
    
    print("\nSpatial Distribution:")
    density_by_town.show(5)
    
    print("\nAge Distribution:")
    age_dist.show()

    species_dist.toPandas().to_csv("../results/species_distribution.csv")
    health_stats.toPandas().to_csv("../results/health_statistics.csv")
    density_by_town.toPandas().to_csv("../results/spatial_distribution.csv")
    age_dist.toPandas().to_csv("../results/age_distribution.csv")
    
    spark.stop()

if __name__ == "__main__":
    main()
    