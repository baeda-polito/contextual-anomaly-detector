#  Copyright © Roberto Chiosa 2024.
#  Email: roberto.chiosa@polito.it
#  Last edited: 13/8/2024

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
from kneed import KneeLocator

from src.cmp.utils import hour_to_dec


def boxplot_fun(group, vector_ad):
    """ Implements outlier detection through IQR rule and box-plot

    :param group: is an 365 length array defining group (values 0 and 1)
    :type group: np.ndarray

    :param vector_ad:  is the cmp matrix median by group (i.e., cluster) by column
    :type vector_ad: np.ndarray

    :return column: array of same length of group specifying if outlier or not (values 0 and 1)
    :rtype column: np.ndarray

    :return fig: figure of the method
    :rtype fig: plot
    """

    # plot
    fig, ax = plt.subplots()
    bp = ax.boxplot(vector_ad, notch=True, patch_artist=True)
    ax.set_ylabel('Distribution of the columns median')
    ax.axes.get_xaxis().set_visible(False)  # remove x-axis
    ax.set_title('Notched box plot')
    # close
    plt.close(fig)

    # get the outliers
    outliers_both_whisker = [flier.get_ydata() for flier in bp["fliers"]]
    outliers_both_whisker = np.array(outliers_both_whisker)
    outliers = outliers_both_whisker[outliers_both_whisker > np.median(vector_ad)]

    # create an array of medians according cluster on yearly period
    median_of_day = np.zeros(group.size)
    j = 0
    for i in range(0, group.size):
        if group[i] == 1:
            median_of_day[i] = vector_ad[j]
            j += 1

    # take the outliers
    try:
        threshold = np.min(outliers)
        column = (median_of_day >= threshold)
    except ValueError:  # raised if outliers is empty.
        column = np.zeros(group.size)
    except Exception as e:
        print("EXCEPTION in boxplot_fun")
        print(e)
        column = np.zeros(group.size)

    # return a vector of 0/1 int
    column = column.astype(int)
    return column, fig


def zscore_fun(group, vector_ad, upper_bound=2):
    """ Implements outlier detection through z-score standardization

    :param group: is an 365 length array defining group (values 0 and 1)
    :type group: np.ndarray

    :param vector_ad:  is the cmp matrix median by group (i.e., cluster) by column
    :type vector_ad: np.ndarray

    :param upper_bound:  upper bound for zscore outlier
    :type upper_bound: float

    :return column: array of same length of group specifying if outlier or not (values 0 and 1)
    :rtype column: np.ndarray

    :return fig: figure of the method
    :rtype fig: plot
    """

    zscore = stats.zscore(vector_ad)

    fig, ax = plt.subplots()
    sns.kdeplot(data=zscore)
    plt.axvline(x=upper_bound, linestyle='dashed', color='gray')
    plt.close(fig)
    # create an array of medians according cluster on yearly period
    outliers = np.zeros(group.size)
    j = 0

    # define vector outliers as same dimensions of group (365)
    for i in range(group.size):
        if group[i] == 1:
            outliers[i] = zscore[j]
            j += 1

    # identify outliers depending on the upper bound
    for k in range(group.size):
        if outliers[k] > upper_bound:
            outliers[k] = 1
        else:
            outliers[k] = 0

    # return a vector of 0/1 int
    column = outliers.astype(int)
    return column, fig


def elbow_fun(group, vector_ad):
    """ Implements outlier detection through elbow method, the values on the sx of the elbow are labelled as anomaly

    :param group: is an 365 length array defining group (values 0 and 1)
    :type group: np.ndarray

    :param vector_ad:  is the cmp matrix median by group (i.e., cluster) by column
    :type vector_ad: np.ndarray

    :return column: array of same length of group specifying if outlier or not (values 0 and 1)
    :rtype column: np.ndarray

    :return fig: figure of the method
    :rtype fig: plot
    """

    xx = np.array(range(0, vector_ad.size))
    yy = np.sort(vector_ad)[::-1]  # decreasing
    kn = KneeLocator(xx, yy, curve='convex', direction='decreasing')
    num_anomalies_to_show = kn.knee

    fig, ax = plt.subplots(figsize=(6, 5))
    plt.plot(yy)
    plt.ylabel("Anomaly Score")
    plt.title("Sorted Anomaly Scores")
    plt.axvline(num_anomalies_to_show, ls=":", c="gray")
    anomaly_ticks = list(range(0, vector_ad.size, int(vector_ad.size / 5)))
    anomaly_ticks.append(num_anomalies_to_show)
    plt.xticks(anomaly_ticks)

    # create an array of medians according cluster on yearly period
    outliers = np.zeros(group.size)

    j = 0
    for i in range(group.size):
        if group[i] == 1:
            outliers[i] = vector_ad[j]
            j += 1

    # take the outliers
    anomaly_day = yy[0:num_anomalies_to_show]

    # take the outliers
    try:
        threshold = min(anomaly_day)
        column = (outliers >= threshold)
    except ValueError:  # raised if anomaly_day is empty.
        column = np.zeros(group.size)
    except Exception as e:
        print("EXCEPTION in elbow_fun")
        print(e)
        column = np.zeros(group.size)

    # return a vector of 0/1 int
    column = column.astype(int)
    return column, fig


def gesd_esd_test(input_series, max_outliers, alpha=0.05):
    """ GESD methods function

    :param input_series: the uni variate series of observations to test
    :type input_series: np.ndarray

    :param max_outliers: the number of expected outliers
    :type max_outliers: int

    :param alpha: level of confidence, default to 0.05
    :type alpha: float

    :return n_outliers: number of outliers found
    :rtype n_outliers: int
    """

    stats_1 = []
    n_outliers = 0
    critical_values = []

    for i in range(1, max_outliers + 1):

        # Calculates the statistical test Ri = max_i*(x_i-x_bar)/sts_dv(x_i)
        max_ind = np.argmax(abs(input_series - np.mean(input_series)))
        r_i = max(abs(input_series - np.mean(input_series))) / np.std(input_series)

        # print('Test {}'.format(iteration))
        # print("Test Statistics Value(R{}) : {}".format(iteration, r_i))

        # compute the critical values
        # 1- alpha/(2*(A+1))  A=n-i  B=tp,n-i-1  i=1....r
        size = len(input_series)
        t_dist = stats.t.ppf(1 - alpha / (2 * size), size - 2)  # 1- alpha/(2*(A+1))   A=n-i
        numerator = (size - 1) * np.sqrt(np.square(t_dist))  # A*B B=tp,n-i-1  i=1....r
        denominator = np.sqrt(size) * np.sqrt(size - 2 + np.square(t_dist))
        lambda_i = numerator / denominator
        # print("Critical Value(λ{}): {}".format(iteration, critical_value))

        # # check values from function
        # if r_i > lambda_i:  # R > C:
        #     print('{} is an outlier. R{} > λ{}: {:.4f} > {:.4f} \n'.format(input_series[max_ind], i,
        #                                                                    i, r_i, lambda_i))
        # else:
        #     print(
        #         '{} is not an outlier. R{}> λ{}: {:.4f} > {:.4f} \n'.format(input_series[max_ind], i,
        #                                                                     i, r_i, lambda_i))

        input_series = np.delete(input_series, max_ind)
        critical_values.append(lambda_i)
        stats_1.append(r_i)
        # The number of outliers is determined by finding the largest i such that Ri > lambda_i.
        if r_i > lambda_i:
            n_outliers = i

    # print('H0:  there are no outliers in the data')
    # print('Ha:  there are up to 10 outliers in the data')
    # print('')
    # print('Significance level:  α = {}'.format(alpha))
    # print('Critical region:  Reject H0 if Ri > critical value')
    # print('Ri: Test statistic')
    # print('lambda_i: Critical Value')
    # print(' ')
    # df = pd.DataFrame({'i': range(1, max_outliers + 1), 'Ri': stats_1, 'lambda_i': critical_values})
    # df.index = df.index + 1

    # print('Number of outliers {}'.format(n_outliers))

    return n_outliers


def gesd_fun(group, vector_ad):
    """ Implements outlier detection through GESD-test

    :param group: is an 365 length array defining group (values 0 and 1)
    :type group: np.ndarray

    :param vector_ad:  is the cmp matrix median by group (i.e., cluster) by column
    :type vector_ad: np.ndarray

    :return column: array of same length of group specifying if outlier or not (values 0 and 1)
    :rtype column: np.ndarray

    :return fig: figure of the method
    :rtype fig: plot
    """

    fig, ax = plt.subplots()
    stats.probplot(vector_ad, plot=plt)
    plt.close(fig)

    n_outliers = gesd_esd_test(input_series=vector_ad, alpha=0.05, max_outliers=10)

    # create an array of medians according cluster on yearly period
    medians = np.zeros(group.size)

    j = 0
    for i in range(group.size):
        if group[i] == 1:
            medians[i] = vector_ad[j]
            j += 1

    # get the n_outliers higher values and store in a vector
    outliers = vector_ad[np.argpartition(vector_ad, -n_outliers)[-n_outliers:]]

    try:
        threshold = min(outliers)
        column = (medians >= threshold) * 1
    except Exception as e:
        print("EXCEPTION in gesd_fun")
        print(e)
        column = np.zeros(group.size)

    return column, fig


def extract_vector_ad_energy(group, data_full, tw, tw_id):
    """
    :param group: is an 365 length array defining group (values 0 and 1)
    :type group: np.ndarray

    :param data_full: is the full dataset
    :type data_full: pd.dataframe

    :param tw: is the time window dataframe
    :type tw: pd.dataframe

    :param tw_id: is the time window index
    :type tw_id: int

    :return vector_ad_energy: array of same length of group specifying the energy absorbed in the time window
    :rtype vector_ad_energy: np.ndarray

    """

    data_tmp = pd.DataFrame(columns=['Date', 'energy', 'mindec'])
    data_tmp['Date'] = data_full.index.date
    # units conversion
    # x * kW * 15min * 1h / (4 * 15min) = x * kW * 1h / 4  = x/4 * kWh
    data_tmp['energy'] = data_full['value'].array / 4  # kWh
    data_tmp['mindec'] = data_full.index.strftime("%H:%M")
    # convert to decimal
    data_tmp = data_tmp.assign(
        mindec=lambda dataframe: dataframe['mindec'].map(
            lambda x: hour_to_dec(x)
        )
    )

    data_tmp['time_window'] = np.where(
        np.logical_and(data_tmp['mindec'] >= hour_to_dec(tw.iloc[tw_id]['from']), data_tmp['mindec'] <= hour_to_dec(
            tw.iloc[tw_id]['to'])), True, False)

    # filter
    data_tmp = data_tmp[data_tmp.time_window == True]
    # calculate energy
    data_tmp_summary = data_tmp.groupby(['Date']).energy.agg('sum')
    # keep only stuff related to the group
    data_tmp_summary = data_tmp_summary[group]
    # convert to array for further processing
    vector_ad_energy = np.asarray(data_tmp_summary)

    return vector_ad_energy


def extract_vector_ad_temperature(group, data_full, tw, tw_id):
    """
    :param group: is an 365 length array defining group (values 0 and 1)
    :type group: np.ndarray

    :param data_full: is the full dataset
    :type data_full: pd.dataframe

     :param tw: is the time window dataframe
    :type tw: pd.dataframe

    :param tw_id: is the time window index
    :type tw_id: int


    :return temperature_score: array of same length of group specifying the energy absorbed in the time window
    :rtype temperature_score: np.ndarray

    """

    data_tmp = pd.DataFrame(columns=['Date', 'temperature', 'mindec'])
    data_tmp['Date'] = data_full.index.date
    data_tmp['temperature'] = data_full['temp'].array
    data_tmp['mindec'] = data_full.index.strftime("%H:%M")
    # convert to decimal
    data_tmp = data_tmp.assign(
        mindec=lambda dataframe: dataframe['mindec'].map(
            lambda x: hour_to_dec(x)
        )
    )

    data_tmp['time_window'] = np.where(
        np.logical_and(data_tmp['mindec'] >= hour_to_dec(tw.iloc[tw_id]['from']), data_tmp['mindec'] <= hour_to_dec(
            tw.iloc[tw_id]['to'])), True, False)

    # filter
    data_tmp = data_tmp[data_tmp.time_window == True]
    # calculate energy
    data_tmp_summary = data_tmp.groupby(['Date']).temperature.mean()
    # keep only stuff related to the group
    data_tmp_summary = data_tmp_summary[group]
    # convert to array for further processing
    vector_ad_temperature = np.asarray(data_tmp_summary)

    return vector_ad_temperature


def extract_vector_ad_cmp(group_cmp):
    """

    :param group_cmp:
    :type group_cmp:

    :return:
    :rtype:
    """
    vector_ad_cmp = np.nanmedian(group_cmp, axis=0)

    return vector_ad_cmp


def anomaly_detection(group, vector_ad):
    """ This function implements the anomaly detection methods

    :param group: is an 365 length array defining group (values 0 and 1)
    :type group: np.ndarray

    :param vector_ad: the vector on which perform the anomaly detection
    :type vector_ad: np.ndarray

    :return ad_score: array of same length of group specifying the severity
    :rtype ad_score: np.ndarray

    """

    column_1, plot_1 = boxplot_fun(group=group, vector_ad=vector_ad)
    column_2, plot_2 = zscore_fun(group=group, vector_ad=vector_ad, upper_bound=2.5)
    column_3, plot_3 = elbow_fun(group=group, vector_ad=vector_ad)
    column_4, plot_4 = gesd_fun(group=group, vector_ad=vector_ad)

    df = pd.DataFrame()
    df['box-plot'] = pd.Series(column_1.astype(int))
    df['z-score'] = pd.Series(column_2.astype(int))
    df['elbow'] = pd.Series(column_3.astype(int))
    df['gesd'] = pd.Series(column_4.astype(int))

    # majority voting
    df['severity'] = df.sum(axis=1, numeric_only=True)

    # return the severity as ndarray to integrate on main code
    ad_score = np.asarray(df['severity'])

    return ad_score
