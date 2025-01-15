from typing import Union

import logging
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s](%(name)s) %(message)s')


def run_clustering(data: pd.DataFrame, df_holidays: Union[None, pd.DataFrame]) -> pd.DataFrame:
    """
    il dataset è filtrato dalle domeniche e festività (cluster1) e dai sabati (cluster2). viene creato poi un algoritmo
    di clustering gerarchico con metodo ward (minimizzare la varianza all'interno del cluster) sul dataset rimanente. il
    numero di cluster (n) è quello che massimizza l'indice di silhouette in un range compreso tra 3 e 6. il numero
    finale di cluster è quindi (n + 2), ricavato grazie a divisioni domain based e algoritmo di clustering k-means.
    ! MODIFICARE IL CLUSTER1 CON I GIORNI DI CHIUSURA DEL POLITECNICO NON PRESENTI IN ITALIAN_HOLIDAYS !
    """

    logging.info("🌲 Running Hierarchical clustering algorithm with ward linkage method.")

    data['date'] = data.index.date
    data['time'] = data.index.time

    if df_holidays is not None:
        # of sundays and holidays
        sunday_dates = [
            date for date in data['date'].unique()
            if pd.Timestamp(date).weekday() == 6 or date in df_holidays.index
        ]
    else:
        # Only sundays
        sunday_dates = [date for date in data['date'].unique() if pd.Timestamp(date).weekday() == 6]
    Cluster1 = pd.DataFrame({'date': sunday_dates})

    # Cluster of saturdays
    saturdays = [
        date for date in data['date'].unique()
        if pd.Timestamp(date).weekday() == 5
    ]
    Cluster2 = pd.DataFrame({'date': saturdays})

    # Hierarchical clustering
    df_working_days = data[~data['date'].isin(set(Cluster1['date']).union(set(Cluster2['date'])))][['value', 'date', 'time']]
    wd_daily_matrix = df_working_days.pivot(index='date', columns='time', values='value')
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

    # Grouping clusters
    group_cluster = pd.DataFrame({'timestamp': pd.to_datetime(data['date'].unique())})
    group_cluster['Cluster_1'] = group_cluster['timestamp'].dt.date.isin(Cluster1['date'])
    group_cluster['Cluster_2'] = group_cluster['timestamp'].dt.date.isin(Cluster2['date'])
    for i in range(3, optimal_clusters + 3):
        group_cluster[f'Cluster_{i}'] = group_cluster['timestamp'].dt.date.isin(
            wd_daily_matrix[wd_daily_matrix['Cluster'] == i].index
        )

    logging.info(f"📊 Clustering algorithm completed successfully. Final number of cluster: {optimal_clusters + 2}")

    return group_cluster
