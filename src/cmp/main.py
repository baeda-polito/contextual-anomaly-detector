#  Copyright © Roberto Chiosa 2024.
#  Email: roberto.chiosa@polito.it
#  Last edited: 17/9/2024

import argparse
import datetime  # data
from statistics import mean

import plotly.express as px
from scipy.stats import zscore
from src.cmp.anomaly_detection_functions import anomaly_detection, extract_vector_ad_temperature, \
    extract_vector_ad_energy, extract_vector_ad_cmp
from src.cmp.utils import *
from src.distancematrix.calculator import AnytimeCalculator
from src.distancematrix.consumer.contextmanager import GeneralStaticManager
from src.distancematrix.consumer.contextual_matrix_profile import ContextualMatrixProfile
from src.distancematrix.generator.euclidean import Euclidean

if __name__ == '__main__':

    # setup logging
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s](%(name)s) %(message)s')

    # setup argument parser
    parser = argparse.ArgumentParser(
        prog='Matrix profile CLI',
        description='Matrix profile')
    parser.add_argument('input_file', help='Path to file', type=str)
    parser.add_argument('variable_name', help='Variable name', type=str)
    parser.add_argument('output_file', help='Path to the output file', type=str)
    args = parser.parse_args()

    ########################################################################################
    # define a begin time to evaluate execution time & performance of algorithm
    begin_time = datetime.datetime.now()

    # The context is a dict defining parameters for report generation
    report_content = {
        'title': 'Anomaly detection report',
        'subtitle': f'Generated on {begin_time.strftime("%Y-%m-%d %H:%M:%S")}',
        'footer_text': '© 2024 Roberto Chiosa',
        'contexts': []
    }

    logger.info(f"Arguments: {args}")

    # automatically identify the number of time windows
    # time window equal bin oppure con cart
    # set mcontext
    # k = 4 per i clusters

    raw_data = download_data(args.input_file)
    data, obs_per_day, obs_per_hour = process_data(raw_data, args.variable_name)

    # print dataset main characteristics
    summary = f''' \n*********************\n
              DATASET: Electrical Load dataset from {args.variable_name}\n
              - From\t{data.index[0]}\n
              - To\t{data.index[len(data) - 1]}\n
              - {len(data.index[::obs_per_day])}\tdays\n
              - 1 \tobservations every 15 min\n
              - {obs_per_day}\tobservations per day\n
              - {obs_per_hour}\tobservations per hour\n
              - {len(data)}observations
              '''

    # Visualise the data  with plotly line plot
    fig = px.line(data['value'])
    fig.update_layout(xaxis_title=None, yaxis_title="Electrical Load [kW]", showlegend=False,
                      paper_bgcolor='rgba(0,0,0,0)')

    report_content['summary'] = {
        "title": "Dataset Summary",
        "content": summary,
        "plot": fig.to_html(full_html=False)
    }

    ########################################################################################
    # Define configuration for the Contextual Matrix Profile calculation.

    # The number of time window has been selected from CART on total electrical power,
    # results are contained in 'time_window.csv' file
    # todo perform cart and create dataframe accordingly
    df_time_window = pd.read_csv(os.path.join(path_to_data, "time_window_corrected.csv"))

    # The context is defined as 1 hour before time window, to be consistent with other analysis,
    # results are loaded from 'm_context.csv' file
    m_context = 1

    # Define output files as dataframe
    # - df_anomaly_results -> in this file the anomaly results will be saved
    # - df_contexts -> the name and descriptions of contexts
    df_anomaly_results = pd.DataFrame()
    df_contexts = pd.DataFrame(
        columns=["from", "to", "context_string", "context_string_small", "duration", "observations"])

    # begin for loop on the number of time windows
    for id_tw in range(len(df_time_window)):

        ########################################################################################
        # Data Driven Context Definition
        if id_tw == 0:
            # manually define context if it is the beginning
            context_start = 0  # [hours] i.e., 00:00
            context_end = context_start + m_context  # [hours] i.e., 01:00
            # [observations] = ([hour]-[hour])*[observations/hour]
            # m = int((hour_to_dec(df_time_window["to"][id_tw]) - 0.25 - m_context) * obs_per_hour)
            m = 23
        else:
            m = df_time_window["observations"][id_tw]  # [observations]
            context_end = hour_to_dec(df_time_window["from"][id_tw]) + 0.25  # [hours]
            context_start = context_end - m_context  # [hours]

        # print string to explain the created context in an intelligible way
        context_string = (f'Subsequences of {dec_to_hour(m / obs_per_hour)} h (m = {m}) that '
                          f'start in [{dec_to_hour(context_start)},{dec_to_hour(context_end)})')

        # contracted context string for names
        context_string_small = (f'ctx_from{dec_to_hour(context_start)}_'
                                f'to{dec_to_hour(context_end)}_m{dec_to_hour(m / obs_per_hour)}'
                                ).replace(":", "_")

        # update context dataframe
        df_contexts.loc[id_tw] = [
            dec_to_hour(context_start),  # "from"
            dec_to_hour(context_end),  # "to"
            context_string,  # "context_string"
            context_string_small,  # "context_string_small"
            str(m_context) + " h",  # "duration"
            m_context * obs_per_hour  # "observations"
        ]

        print(f'\n*********************\nCONTEXT {str(id_tw + 1)} : {context_string} ({context_string_small})')

        # if figures directory doesnt exists create and save into it
        ensure_dir(os.path.join(path_to_figures, context_string_small))

        # Context Definition:
        contexts = GeneralStaticManager([
            range(
                # FROM  [observations]  = x * 96 [observations] + 0 [hour] * 4 [observation/hour]
                ((x * obs_per_day) + dec_to_obs(context_start, obs_per_hour)),
                # TO    [observations]  = x * 96 [observations] + (0 [hour] + 2 [hour]) * 4 [observation/hour]
                ((x * obs_per_day) + dec_to_obs(context_end, obs_per_hour)))
            for x in range(len(data) // obs_per_day)
        ])

        ########################################################################################
        # Calculate Contextual Matrix Profile
        calc = AnytimeCalculator(m, data['value'].values)

        # Add generator Not Normalized Euclidean Distance
        distance_string = 'Not Normalized Euclidean Distance'
        calc.add_generator(0, Euclidean())

        # We want to calculate CMP initialize element
        cmp = calc.add_consumer([0], ContextualMatrixProfile(contexts))

        # Calculate Contextual Matrix Profile (CMP)
        calc.calculate_columns(print_progress=True)
        print("\n")

        # if data directory doesnt exists create and save into it
        ensure_dir(os.path.join(path_to_data, context_string_small))

        # Save CMP for R plot (use to_csv)
        np.savetxt(os.path.join(path_to_data, context_string_small, 'plot_cmp_full.csv'),
                   nan_diag(cmp.distance_matrix),
                   delimiter=",")

        # Save CMP for R plot (use to_csv)
        np.savetxt(os.path.join(path_to_data, context_string_small, 'match_index_query.csv'),
                   cmp.match_index_query,
                   delimiter=",")

        # Save CMP for R plot (use to_csv)
        np.savetxt(os.path.join(path_to_data, context_string_small, 'match_index_series.csv'),
                   cmp.match_index_series,
                   delimiter=",")

        # calculate the date labels to define the extent of figure
        date_labels = data.index[::obs_per_day].strftime('%Y-%m-%d')

        # USe colorscale consistent in eaxh context
        val_min = 0
        # Find the maximum value among the finite values
        val_max = np.nanmax(cmp.distance_matrix[np.isfinite(cmp.distance_matrix)])

        fig = px.imshow(cmp.distance_matrix, zmin=val_min, zmax=round(val_max),
                        labels=dict(color="Distance"),
                        x=date_labels, y=date_labels)
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')

        report_content['contexts'].append({
            "title": f"Context {str(id_tw + 1)}",
            "subtitle": f"{context_string}({context_string_small}",
            "content": f"On the right the CMP result is reported for the "
                       f"similarity-join search of subsequences that start in context {str(id_tw + 1)}. Each value of "
                       f"the matrix shows the Euclidean distance between the best matching subsequences. The lower "
                       f"the distance the better the match.",
            "plot": fig.to_html(full_html=False),
            "clusters": []
        })

        ########################################################################################
        # todo perform cluster analysis
        # Load Cluster results as boolean dataframe: each column represents a group
        group_df = pd.read_csv(os.path.join(path_to_data, "group_cluster.csv"), index_col='timestamp', parse_dates=True)
        # initialize dataframe of results for context to be appended to the overall result
        df_anomaly_context = group_df.astype(int)

        # set labels
        day_labels = data.index[::obs_per_day]
        # get number of groups
        n_group = group_df.shape[1]

        # perform analysis of context on groups (clusters)
        for id_cluster in range(n_group):
            # create this dataframe where dates cluster and anomalies scores will be saved
            df_result_context_cluster = pd.DataFrame()
            # time when computation starts
            begin_time_group = datetime.datetime.now()

            # get group name from dataframe
            group_name = group_df.columns[id_cluster]

            # add column of context of group in df_output
            df_anomaly_context[f'{group_name}.{context_string_small}'] = [0 for id_cluster in
                                                                          range(len(df_anomaly_context))]

            # create empty group vector
            group = np.array(group_df.T)[id_cluster]
            # get cmp from previously computed cmp
            group_cmp = cmp.distance_matrix[:, group][group, :]
            # substitute inf with zeros
            group_cmp[group_cmp == np.inf] = 0
            # get dates
            group_dates = data.index[::obs_per_day].values[group]

            # save group CMP for R plot
            np.savetxt(os.path.join(path_to_data, context_string_small, f'plot_cmp_{group_name}.csv'),
                       nan_diag(group_cmp), delimiter=",")

            # Save CMP for R plot (use to_csv)
            np.savetxt(os.path.join(path_to_data, context_string_small, f'match_index_query_{group_name}.csv'),
                       cmp.match_index_query[:, group][group, :], delimiter=",")

            # Save CMP for R plot (use to_csv)
            np.savetxt(os.path.join(path_to_data, context_string_small, f'match_index_series_{group_name}.csv'),
                       cmp.match_index_series[:, group][group, :], delimiter=",")

            # plot cluster matrix
            fig = px.imshow(group_cmp, zmin=val_min, zmax=round(val_max),
                            labels=dict(color="Distance"),
                            x=group_dates, y=group_dates)
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')

            #######################################

            # add date to df_result_context_cluster
            df_result_context_cluster["Date"] = group_df.index
            df_result_context_cluster["cluster"] = group

            # calculate the vector to be used for the anomaly detection
            vector_ad_cmp = extract_vector_ad_cmp(group_cmp=group_cmp)

            vector_ad_energy = extract_vector_ad_energy(
                group=group,
                data_full=data,
                tw=df_time_window,
                tw_id=id_tw)

            vector_ad_temperature = extract_vector_ad_temperature(
                group=group,
                data_full=data,
                tw=df_time_window,
                tw_id=id_tw)

            # calculate anomaly score though majority voting
            cmp_ad_score = anomaly_detection(
                group=group,
                vector_ad=vector_ad_cmp)

            energy_ad_score = anomaly_detection(
                group=group,
                vector_ad=vector_ad_energy)

            temperature_ad_score = anomaly_detection(
                group=group,
                vector_ad=vector_ad_temperature)

            # temperature_ad_score = stats.zscore(vector_ad_temperature)

            # add anomaly score to df_result_context_cluster
            df_result_context_cluster["cmp_score"] = cmp_ad_score
            df_result_context_cluster["energy_score"] = energy_ad_score
            df_result_context_cluster["temperature_score"] = temperature_ad_score

            # add categorization depending on some criteria
            # set to nan if severity 0/1/2 (no anomaly or not severe)
            # cmp_ad_score = np.where(cmp_ad_score == 0, np.nan, cmp_ad_score)
            # override definition of cmp_ad_score with this new definition
            # todo: when everything works fine change the variable cmp_ad_score to avoid misunderstandings
            cmp_ad_score = np.array(df_result_context_cluster["cmp_score"] + df_result_context_cluster["energy_score"])

            cmp_ad_score = np.where(cmp_ad_score < 6, np.nan, cmp_ad_score)
            # get date to plot
            cmp_ad_score_index = np.where(~np.isnan(cmp_ad_score))[0].tolist()
            cmp_ad_score_dates = date_labels[cmp_ad_score_index]

            anomalies_table = pd.DataFrame()
            anomalies_table["Date"] = cmp_ad_score_dates
            anomalies_table["Anomaly Score"] = cmp_ad_score[cmp_ad_score_index]
            anomalies_table["Rank"] = anomalies_table.index + 1

            # the number of anomalies is the number of non nan elements, count
            num_anomalies_to_show = np.count_nonzero(~np.isnan(cmp_ad_score))

            report_content['contexts'][id_tw]["clusters"].append({
                "title": f"Cluster {id_cluster + 1}",
                "content": f"The current cluster contains {len(group_dates)} days and "
                           f"{num_anomalies_to_show} anomalies "
                           f"identified in the context defines as {context_string.lower()}. The plot referring to the "
                           f"cluster and relative anomaly (if any) are reported in the line-plot below. The red line "
                           f"refers to the anomalous day, while the light orange box refers to the time "
                           f"window {id_tw + 1} and the dark orange the context under analysis.",
                "plot": fig.to_html(full_html=False),
                "table": anomalies_table.to_html(index=False,
                                                 classes='table table-striped table-hover',
                                                 border=0, ),
                "plot_anomalies": None,
            })

            # only visualize if some anomaly are shown
            if num_anomalies_to_show > 0:

                # limit the number of anomalies
                if num_anomalies_to_show > 10:
                    num_anomalies_to_show = 10

                # plot lineplot with daily load profiles using plotly
                data_plot = data['value'].values.reshape((-1, obs_per_day))[group].T
                # rename columns with group_date
                data_plot = pd.DataFrame(data_plot, columns=group_dates)
                # plot lines
                fig = px.line(data_plot, line_shape="spline")

                # Update traces to set all lines to gray
                fig.update_traces(line=dict(color='rgba(128, 128, 128, 0.2)'))

                # Highlight only anomalous days
                for date in cmp_ad_score_dates:
                    # get column that matches the date
                    index_anom_plot = data_plot.columns.get_loc(date)
                    fig.data[index_anom_plot].update(line=dict(color='red', width=2))

                # add rectangle area defining the context
                fig.add_vrect(x0=context_start * obs_per_hour,
                              x1=context_end * obs_per_hour,
                              fillcolor="lightsalmon", opacity=0.8, layer="below", line_width=0)
                # add rectangle defining time window
                fig.add_vrect(
                    x0=hour_to_dec(df_time_window["from"][id_tw]) * obs_per_hour,
                    x1=hour_to_dec(df_time_window["to"][id_tw]) * obs_per_hour,
                    fillcolor="lightsalmon", opacity=0.5, layer="below", line_width=0)

                fig.update(layout=dict(
                    xaxis_title=None,
                    yaxis_title="Power [kW]",
                    paper_bgcolor='rgba(0,0,0,0)',
                    showlegend=False
                ))

                # add plot to be plotted in report
                report_content['contexts'][id_tw]["clusters"][id_cluster]["plot_anomalies"] = fig.to_html(
                    full_html=False)

                # print the execution time
                time_interval_group = datetime.datetime.now() - begin_time_group
                hours, remainder = divmod(time_interval_group.total_seconds(), 3600)
                minutes, seconds = divmod(remainder, 60)
                string_anomaly_print = '- %s (%.3f s) \t-> %.d anomalies' % (
                    group_name.replace('_', ' '), seconds, num_anomalies_to_show)
                print(string_anomaly_print)

            # if no anomaly to show not visualize
            else:
                string_anomaly_print = "- " + group_name.replace('_', ' ') + ' (-) \t\t-> no anomalies'
                print(string_anomaly_print, "green")

            # save intermediate results

            # filter only where cluster
            df_result_context_cluster = df_result_context_cluster[df_result_context_cluster.cluster == True]

            # drop cluster column
            df_result_context_cluster = df_result_context_cluster.drop(['cluster'], axis=1)

            df_result_context_cluster["vector_ad_cmp"] = vector_ad_cmp
            df_result_context_cluster["vector_ad_energy"] = vector_ad_energy

            mean_energy = mean(df_result_context_cluster["vector_ad_energy"])

            df_result_context_cluster["vector_ad_energy_absolute"] = df_result_context_cluster[
                                                                         "vector_ad_energy"] - mean_energy
            df_result_context_cluster["vector_ad_energy_relative"] = (df_result_context_cluster[
                                                                          "vector_ad_energy"] / mean_energy - 1) * 100
            df_result_context_cluster["vector_ad_temperature"] = zscore(vector_ad_temperature)

            df_result_context_cluster.to_csv(
                os.path.join(path_to_data, context_string_small, f'anomaly_results_{group_name}.csv'), index=False)

        # at the end of loop on groups save dataframe corresponding to given context or append to existing one
        if df_anomaly_results.empty:
            df_anomaly_results = df_anomaly_context
        else:
            # concatenate dataframes by column
            df_anomaly_results = pd.concat([df_anomaly_results, df_anomaly_context], axis=1)
            # remove redundant columns
            df_anomaly_results = df_anomaly_results.loc[:, ~df_anomaly_results.columns.duplicated()]

    # at the end of loop on context save dataframe of results
    df_anomaly_results.to_csv(os.path.join(path_to_data, "anomaly_results.csv"))
    df_contexts.to_csv(os.path.join(path_to_data, "contexts.csv"), index=False)

    # print the execution time
    total_time = datetime.datetime.now() - begin_time
    hours, remainder = divmod(total_time.total_seconds(), 3600)
    minutes, seconds = divmod(remainder, 60)
    print('\n*********************\n')
    logger.info(f"TOTAL {str(int(minutes))} min {str(int(seconds))} s")

    save_report(report_content, args.output_file)
