import pandas as pd
import holidays
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import numpy as np
pd.set_option('display.max_rows', None)

def run_clustering(data: pd.DataFrame) -> pd.DataFrame:
    """
    il dataset è filtrato dalle domeniche e festività (cluster1) e dai sabati (cluster2). viene creato poi un algoritmo
    di clustering gerarchico con metodo ward (minimizzare la varianza all'interno del cluster) sul dataset rimanente. il
    numero di cluster (n) è quello che massimizza l'indice di silhouette in un range compreso tra 3 e 6. il numero
    finale di cluster è quindi (n + 2), ricavato grazie a divisioni domain based e algoritmo di clustering k-means.
    ! MODIFICARE IL CLUSTER1 CON I GIORNI DI CHIUSURA DEL POLITECNICO NON PRESENTI IN ITALIAN_HOLIDAYS !
    """

    data['timestamp'] = data.index
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    data['date'] = data['timestamp'].dt.date
    data['time'] = data['timestamp'].dt.time

    #CLUSTER1
    italian_holidays = holidays.IT()
    holiday_sunday_dates = [
        date for date in data['date'].unique()
        if pd.Timestamp(date).weekday() == 6 or date in italian_holidays
    ]
    Cluster1 = pd.DataFrame({'date': holiday_sunday_dates})
    #CLUSTER2
    saturdays = [
        date for date in data['date'].unique()
        if pd.Timestamp(date).weekday() == 5
    ]
    Cluster2 = pd.DataFrame({'date': saturdays})

    #CLUSTERING GERARCHICO
    df_working_days = data[~data['date'].isin(set(Cluster1['date']).union(set(Cluster2['date'])))][['value', 'date', 'time']]
    wd_daily_matrix = df_working_days.pivot(index='date', columns='time', values='value')
    linkage_matrix = linkage(wd_daily_matrix, method='ward')
    range_clusters = range(3, 7)
    silhouette_scores = []
    for n_clusters in range_clusters:
        clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
        cluster_labels = clustering.fit_predict(wd_daily_matrix)
        score = silhouette_score(wd_daily_matrix, cluster_labels)
        silhouette_scores.append(score)
    optimal_clusters = range_clusters[silhouette_scores.index(max(silhouette_scores))]
    final_clustering = AgglomerativeClustering(n_clusters=optimal_clusters, linkage='ward')
    final_labels = final_clustering.fit_predict(wd_daily_matrix) + 3
    wd_daily_matrix['Cluster'] = final_labels
    plt.figure(figsize=(10, 6))
    dendrogram(linkage_matrix)
    plt.title("Dendrogramma - Metodo Ward")
    plt.xlabel("Profili di carico giornaliero")
    plt.ylabel("Distanza")
    # plt.show()

    #CSV FINALE
    group_cluster = pd.DataFrame({'timestamp': pd.to_datetime(data['date'].unique())})
    group_cluster['Cluster_1'] = group_cluster['timestamp'].dt.date.isin(Cluster1['date'])
    group_cluster['Cluster_2'] = group_cluster['timestamp'].dt.date.isin(Cluster2['date'])
    for i in range(3, optimal_clusters + 3):
        col_name = f'Cluster_{i}'
        group_cluster[col_name] = group_cluster['timestamp'].dt.date.isin(
            wd_daily_matrix[wd_daily_matrix['Cluster'] == i].index
        )
    # print(group_cluster)
    print(f"Number of cluster: {optimal_clusters + 2}") # +2 uno per i sabati e uno per domenica e festivi
    print(f"Silouette score: {round(max(silhouette_scores),2)}")
    group_cluster.to_csv('data/group_cluster_new.csv', index=False)
    return group_cluster

if __name__ == '__main__':
    df = pd.read_csv("data/data.csv")
    run_clustering(df)
