import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, SpectralClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import numpy as np
import matplotlib.pyplot as plt
from rich.console import Console
from rich.table import Table

def clustering_with_4_methods(df, n_clusters=5):
    """
    Perform clustering using four different methods and visualize the results in 2D.

    Parameters:
    n_clusters (int): Number of clusters to create. Default is 5.
    df (pd.DataFrame): DataFrame containing feature data for the clustering.
    Returns:
    clustering_methods (dict): Dictionary containing the clustering methods used.
    df (pd.DataFrame): DataFrame containing features with additional columns for each clustering method.
    """
    X = df.iloc[:, 1:]
    pca = PCA(n_components=2)
    X_reduced = pca.fit_transform(X)
    clustering_methods = {
        'KMeans': KMeans(n_clusters=n_clusters),
        'AgglomerativeClustering': AgglomerativeClustering(n_clusters=n_clusters),
        'DBSCAN': DBSCAN(eps=0.5, min_samples=n_clusters),
        'SpectralClustering': SpectralClustering(n_clusters=n_clusters)
    }
    results = {}
    for method_name, method in clustering_methods.items():
        clusters = method.fit_predict(X)
        results[method_name] = clusters
        df[method_name] = clusters

    fig, axes = plt.subplots(2, 2, figsize=(20, 15))
    axes = axes.flatten()
    for i, method_name in enumerate(clustering_methods.keys()):
        ax = axes[i]
        scatter = ax.scatter(X_reduced[:, 0], X_reduced[:, 1], c=df[method_name], cmap='viridis', s=100)   
        ax.set_title(f'{method_name} Clustering', fontsize=12)
        ax.set_xlabel('PCA 1', fontsize=10)
        ax.set_ylabel('PCA 2', fontsize=10)
        plt.colorbar(scatter, ax=ax, label='Cluster ID')

    plt.tight_layout()
    plt.show()
    return clustering_methods, df

def display_results_in_console(clustering_methods, df, df_with_drinktype_and_glass):
    """
    Display the clustering results in a colorful table for each method, including ingredient types and glass info.
    
    Parameters:
    clustering_methods (dict): Dictionary containing the clustering methods used.
    df (pd.DataFrame): DataFrame containing features with additional columns for each clustering method.
    df_with_drinktype_and_glass (pd.DataFrame): DataFrame containing ingredient types and glass information.
    Returns:
    None.
    """
    console = Console()
    cluster_colors = {
        0: "green",
        1: "red",
        2: "blue",
        3: "yellow",
        4: "magenta",
        -1: "white"  
    }

    # Zakładam, że 1. kolumna w df to nazwa, a 3. kolumna w df_with_drinktype_and_glass to nazwa
    merge_columns = [df_with_drinktype_and_glass.columns[1]] + \
                    [f'ingredient{i}_type' for i in range(1, 7)] + \
                    ['glass']
    merged_df = pd.merge(
        df,
        df_with_drinktype_and_glass[merge_columns],
        left_on=df.columns[0], 
        right_on=df_with_drinktype_and_glass.columns[1],
        how='left'
    )

    cocktail_names = merged_df.iloc[:, 0] 
    for method_name in clustering_methods.keys():
        console.print(f"\nWyniki klasteryzacji dla metody: [bold green]{method_name}[/bold green]")
        cluster_table = pd.DataFrame({
            'Cocktail Name': cocktail_names,
            'Cluster': merged_df[method_name],
            'Ingredient1 Type': merged_df['ingredient1_type'],
            'Ingredient2 Type': merged_df['ingredient2_type'],
            'Ingredient3 Type': merged_df['ingredient3_type'],
            'Ingredient4 Type': merged_df['ingredient4_type'],
            'Ingredient5 Type': merged_df['ingredient5_type'],
            'Ingredient6 Type': merged_df['ingredient6_type'],
            'Glass': merged_df['glass']
        })
        
        table = Table(title=f"{method_name} Clustering Results", show_header=True, header_style="bold magenta")
        table.add_column("Cocktail Name", style="cyan", justify="left")
        table.add_column("Cluster", justify="center")
        for i in range(1, 7):
            table.add_column(f"Ingredient{i} Type", style="yellow", justify="left")
        table.add_column("Glass", style="green", justify="left")
        
        for _, row in cluster_table.iterrows():
            cluster_id = row['Cluster']
            color = cluster_colors.get(cluster_id, "white")
            table.add_row(
                str(row['Cocktail Name']),
                str(cluster_id),
                str(row['Ingredient1 Type']),
                str(row['Ingredient2 Type']),
                str(row['Ingredient3 Type']),
                str(row['Ingredient4 Type']),
                str(row['Ingredient5 Type']),
                str(row['Ingredient6 Type']),
                str(row['Glass']),
                style=color
            )
        
        console.print(table)
        console.print(f"Liczba klastrów: [bold blue]{len(set(merged_df[method_name]))}[/bold blue]")

def find_optimal_clusters(df, max_clusters = 20):
    wcss = []
    silhouette_scores = []
    cluster_range = range(2, max_clusters + 1)
    X = df.iloc[:, 1:]
    for k in cluster_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X)
        wcss.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(X, cluster_labels))
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(cluster_range, wcss, marker='o')
    plt.xlabel('Liczba klastrów')
    plt.ylabel('WCSS')
    plt.title('Metoda łokcia')
    
    plt.subplot(1, 2, 2)
    plt.plot(cluster_range, silhouette_scores, marker='o', color='red')
    plt.xlabel('Liczba klastrów')
    plt.ylabel('Silhouette Score')
    plt.title('Współczynnik Silhouette')
    
    plt.show()
    
    optimal_clusters = cluster_range[np.argmax(silhouette_scores)]
    print(f'Optymalna liczba klastrów: {optimal_clusters}')
    return optimal_clusters