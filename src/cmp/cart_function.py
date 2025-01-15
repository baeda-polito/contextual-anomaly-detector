import pandas as pd
from sklearn.tree import DecisionTreeRegressor, _tree
import numpy as np
import logging

MIN_INTERVAL_LENGTH = 2.5

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s](%(name)s) %(message)s')


def run_cart(data: pd.DataFrame) -> pd.DataFrame:
    """
    Fits a Decision Tree Regressor (CART) model on time-series data (excluding weekends) to identify time intervals
    based on power consumption. The model iteratively adjusts the minimum number of samples per leaf to ensure that all
    intervals have a duration of at least 2.5 hours. The final time intervals and related information are returned in a
    DataFrame.

    :param data: time-series data with a datetime index and a 'value' column
    :return: DataFrame with time intervals and related information. Columns include 'id', 'description', 'observations',
    'from', 'to', 'duration', and 'node'.
    """

    min_samples_leaf = int(len(data) * 0.05)

    # Data preparation
    data['date'] = data.index.date
    data['time'] = data.index.time
    working_days_df = data[~data.index.weekday.isin([5, 6])]
    working_days_df = working_days_df.drop(columns=['temp'])
    working_days_df['time_numeric'] = working_days_df['time'].apply(lambda x: x.hour + x.minute / 60)

    # Defining X and y for the CART
    X = working_days_df['time_numeric'].to_numpy().reshape(-1, 1)
    y = working_days_df['value'].to_numpy().reshape(-1, 1)

    def extract_intervals(tree):
        """Extract the time intervals from the CART model

        :param tree: CART model

        :return: intervals, nodes
        """
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
    iter_max = 10
    while n_iterations < iter_max:
        tree = DecisionTreeRegressor(min_samples_leaf=min_samples_leaf, max_depth=4, random_state=42)
        tree.fit(X, y)

        intervals, nodes = extract_intervals(tree)
        intervals = [(float(start), float(end)) for start, end in intervals]
        rounded_intervals = np.round(np.array(intervals) * 4) / 4
        intervals_duration = [(end - start) for start, end in rounded_intervals]

        if all(duration >= MIN_INTERVAL_LENGTH for duration in intervals_duration):
            time_windows = pd.DataFrame({
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
            break
        else:
            min_samples_leaf += 500
    n_iterations += 1

    return time_windows
