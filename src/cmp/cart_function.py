import pandas as pd
import holidays
from sklearn.tree import DecisionTreeRegressor, plot_tree, _tree
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import numpy as np

pd.set_option('display.max_rows', None)

def run_cart(data: pd.DataFrame)-> pd.DataFrame:
    """
    è usato il dataset sotto dorma di timeserie (e non come matrice MxN con profili di carico giornalieri) filtrato
    da sabati, domeniche e festività.
    Viene trasformata l'ora in numero per facilitare la costruzione del CART. La funzione extract_intervals prende in
    input il modello del CART ed estrapola gli intervalli orari provocati dagli split e i nomi dei nodi del CART. Nel
    ciclo while True è creato il CART utilizzando l'orario come variabile di split, la potenza quartioraria come
    variabile target e numero di campioni nei nodi foglia come criterio di arresto (pari a 2000 inizialmente).
    è utilizzata la cross-validation per la creazione del cart con 4 fold. è richiamata la funzione
    extract_intervals, vengono estratti gli intervalli e gli estremi vengono approssimati al quarto d'ora più vicino
    (prima numericamente e poi trasformati in formato orari). Dopo l'arrotondamento viene calcolata la durata di ogni
    intervallo e se ognuno di essi è maggiore o uguale a 2,5h allora il CART è quello definitivo, viene plottato e viene
    creato il csv "time_window_corrected" con tutte le info. Se almeno un intervallo è minore di 2,5h allora è aumentato
    il numero minimo di campioni nei nodi fogli di 500 unità.
    """
    min_samples_leaf = 2000
    min_interval_length = 2.5

    data['timestamp'] = data.index
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    data['date'] = data['timestamp'].dt.date
    data['time'] = data['timestamp'].dt.time
    holidays_it = holidays.IT()
    working_days_df = data[~data['date'].apply(lambda x: x in holidays_it)]
    working_days_df = working_days_df[~working_days_df['timestamp'].dt.weekday.isin([5, 6])]
    working_days_df = working_days_df.drop(columns=['temp', 'timestamp'])

    working_days_df['time_numeric'] = working_days_df['time'].apply(lambda x: x.hour + x.minute / 60)
    X = working_days_df['time_numeric']
    y = working_days_df['value']

    def extract_intervals(tree): #fa una lista degli intervalli orari dagli split dell'albero
        cart = tree.tree_
        intervals = []
        nodes = []
        node_counter = [1]

        def recurse(node, lower=0, upper=24):
            current_node = f"Node {node_counter[0]}"
            node_counter[0] += 1
            if cart.feature[node] != _tree.TREE_UNDEFINED:
                threshold = cart.threshold[node]
                left_child = cart.children_left[node]
                right_child = cart.children_right[node]
                recurse(left_child, lower, min(upper, threshold))
                recurse(right_child, max(lower, threshold), upper)
            else:
                intervals.append((lower, upper))
                nodes.append(current_node)
        recurse(0)
        return intervals, nodes

    n_iterations = 0
    while True:
        n_iterations += 1
        tree = DecisionTreeRegressor(min_samples_leaf=min_samples_leaf, random_state=42)
        scores = cross_val_score(tree, X.values.reshape(-1, 1), y, cv=4, scoring='neg_mean_squared_error')
        # print(f"Cross-validation scores: {scores}")
        # print(f"Mean of cross-validation scores: {-scores.mean()}")
        tree.fit(X.values.reshape(-1, 1), y)

        intervals, nodes = extract_intervals(tree)
        intervals = [(float(start), float(end)) for start, end in intervals]
        rounded_intervals = np.round(np.array(intervals) * 4) / 4
        intervals_duration = [(end - start) for start, end in rounded_intervals]

        if all(duration >= min_interval_length for duration in intervals_duration):
            # plt.figure(figsize=(20, 10))
            # plot_tree(tree, feature_names=['time_numeric'], filled=True, fontsize=10)
            # plt.show()
            time_window_corrected = pd.DataFrame({
                'id': range(1, len(rounded_intervals) + 1),
                'description': [
                    f"[{int(start):02d}:{int((start % 1) * 60):02d} - {int(end):02d}:{int((end % 1) * 60):02d})"
                    for start, end in rounded_intervals
                ],
                'observations': [
                    int((end - start) * 4)
                    for start, end in rounded_intervals
                ],
                'from': [
                    f"{int(start):02d}:{int((start % 1) * 60):02d}"
                    for start, end in rounded_intervals
                ],
                'to': [
                    f"{int(end):02d}:{int((end % 1) * 60):02d}"
                    for start, end in rounded_intervals
                ],
                'duration': [
                    f"{int((end - start) * 3600)}s (~{(end - start):.2f} hours)"
                    for start, end in rounded_intervals
                ],
                'node': nodes
            })
            time_window_corrected.to_csv('data/time_window_corrected_new.csv', index=False)
            # print(time_window_corrected)
            print(f"Minimum time window of 2.5h achieved after {n_iterations} iterations.")
            print(f"Minimum number of samples in leaf nodes of CART: {min_samples_leaf}")
            break
        else:
            min_samples_leaf += 500

    return time_window_corrected

if __name__ == '__main__':
    df = pd.read_csv("data/data.csv")
    run_cart(df)
