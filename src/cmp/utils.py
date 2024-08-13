#  Copyright © Roberto Chiosa 2024.
#  Email: roberto.chiosa@polito.it
#  Last edited: 13/8/2024
import logging
import math
import os

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import requests
from jinja2 import Environment, FileSystemLoader

# Path to folders
path_to_data = os.path.join(os.path.dirname(__file__), 'data')
path_to_figures = os.path.join(os.path.dirname(__file__), 'results', 'figures')
path_to_reports = os.path.join(os.path.dirname(__file__), 'results', 'reports')
path_to_templates = os.path.join(os.path.dirname(__file__), 'templates')

color_palette = 'viridis'
dpi_resolution = 300
fontsize = 10
line_style_context = '-'
line_style_other = ':'
line_color_context = '#D83C3B'
line_color_other = '#D5D5E0'
line_size = 1

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s](%(name)s) %(message)s')


def calculate_time_windows():
    """
    Calculate the time windows for the contextual matrix profile
    :return:
    """
    df = pd.read_csv(os.path.join(path_to_data, "time_window_corrected.csv"))
    return df


def save_report(context, filepath: str) -> None:
    """Save the report to a file

    :param context: context of the report
    :param filepath: path to save the report

    """
    # Set up the Jinja2 environment for report
    env = Environment(loader=FileSystemLoader(path_to_templates))
    template = env.get_template('cmp.html')

    # Render the template with the data
    html_content = template.render(context)

    # Save the rendered HTML to a file (optional, for inspection)
    with open(filepath, 'w') as file:
        file.write(html_content)
        logger.info(f'🎉 Report generated successfully on {filepath}')


def download_data(filepath: str) -> pd.DataFrame:
    """
    Download data from a user specified path

    The input dataset via an HTTP URL. The tool should then download the dataset from that URL;
    since it's a pre-signed URL, the tool would not need to deal with authentication—it can just download
    the dataset directly.

    :param filepath:
    :return: data
    """
    # if filepath is an url request otherwise read from file
    if filepath.startswith('http'):
        logger.info(f"⬇️ Downloading file from online url <{filepath}>")
        res = requests.get(filepath)
        data = pd.read_csv(res.text)
    else:
        logger.info(f"⬇️ Reading local file from <{filepath}> path")
        data = pd.read_csv(filepath)
    return data


def process_data(data_raw: pd.DataFrame, variable: str) -> tuple:
    """Load data from a file
    the data should always have timestamp ant temperature and n columns with electrical loads names in a custom way.
    This function transforms the dataset into a 3 column format renaming the desired column with "value" name
    :param data_raw:
    :param variable:
    :return data: data from the file
    :return obs_per_day: number of observations per day
    :return obs_per_hour: number of observations per hour
    """
    try:
        # subset the dataset into 3 columns
        data_raw = data_raw[['timestamp', variable, 'temp']]
        # rename columns
        data_raw = data_raw.rename(columns={variable: "value"})
        # convert timestamp to datetime
        data_raw['timestamp'] = pd.to_datetime(data_raw['timestamp'])
        data_raw = data_raw.set_index('timestamp')
        # a little preprocessing if necessary
        data_process = data_raw.copy()
        data_process['value'] = data_process['value'].interpolate(method='linear')
        data_process['temp'] = data_process['temp'].interpolate(method='linear')
        # calculate observation per day
        obs_per_hour = int(np.median(data_process.resample('1h').count()))  # [observation/hour]
        obs_per_day = int(np.median(data_process.resample('1d').count()))  # [observation/day]

        logger.info('📊 Data processed successfully')
        return data_process, obs_per_day, obs_per_hour
    except Exception as e:
        logger.error(f"🔴 Error processing data: {e}")
        raise


def ensure_dir(dir_path: str) -> None:
    """Ensures that the directory exists

    :param dir_path: path to the file

    :example:
    >>> ensure_dir('data/processed')
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def hour_to_dec(hour_str: str) -> float:
    """ Transforms float hours from HH:MM string format to float with decimal places

    :param hour_str: hour in format HH:MM
    :type hour_str: str

    :return hour_dec: hour in numerical format
    :rtype hour_dec: float

    :example:
    >>> hour_to_dec('02:00')
    2.0
    """

    (H, M) = hour_str.split(':')
    hour_dec = int(H) + int(M) / 60
    return hour_dec


def dec_to_hour(hour_dec: float) -> str:
    """ Transforms float hours with decimal places into HH:MM string format

    :param hour_dec: hour in numerical format
    :type hour_dec: float

    :return hour_str: hour in format HH:MM
    :rtype hour_str: str

    :example:
    >>> dec_to_hour(2.5)
    '02:30'
    """

    (H, M) = divmod(hour_dec * 60, 60)
    hour_str = "%02d:%02d" % (H, M)
    return hour_str


def dec_to_obs(hour_dec, obs_per_hour):
    """  transforms float hours with decimal places into HH:MM string format

    :param hour_dec: hour interval in numerical format
    :type hour_dec: float

    :param obs_per_hour: number of observations per hour
    :type obs_per_hour: int

    :return observations: number of observations
    :rtype observations: int

    :example:
    >>> # 6.30 -> H = 6, M = 30
    >>> #6[hours]*4[observations/hour] + 30[minutes]*1/15[observations/minutes] = 25 [observations]
    >>> dec_to_obs(6.30 , 4)
    25
    """

    (H, M) = divmod(hour_dec * 60, 60)
    observations = int(H * obs_per_hour + M / 15)
    return observations


def roundup(x, digit=1):
    """  rounds number too upper decimal

    :param x: number
    :type x: float

    :param digit: number of digit to round
    :type digit: int

    :return rounded: rounded number
    :rtype rounded: int

    :example:
    >>> roundup(733, digit=10)
    740
    """
    rounded = int(math.ceil(x / digit)) * digit
    return rounded


def nan_diag(matrix):
    """ Transforms a square matrix into a matrix with na on main diagonal

    :param matrix:a matrix of numbers
    :type matrix: np.matrix

    :return matrix_nan: matrix of numbers
    :rtype matrix_nan: np.matrix
    """

    (x, y) = matrix.shape

    if x != y:
        raise RuntimeError("Matrix is not square")

    matrix_nan = matrix.copy()
    matrix_nan[range(x), range(y)] = np.nan
    return matrix_nan


def cmp_plot(contextual_mp,
             palette="viridis",
             title=None,
             xlabel=None,
             legend_label=None,
             extent=None,
             date_ticks=14,
             index_ticks=5
             ):
    """ utils function used to plot the contextual matrix profile

    :param contextual_mp:
    :param palette:
    :param title:
    :param xlabel:
    :param legend_label:
    :param extent:
    :param date_ticks:
    :param index_ticks:
    :return:
    """

    figure = plt.figure()
    axis = plt.axes()

    if extent is not None:
        # no extent dates given
        im = plt.imshow(nan_diag(contextual_mp),
                        cmap=palette,
                        origin="lower",
                        extent=extent
                        )
        # Label layout
        axis.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        axis.yaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        axis.xaxis.set_major_locator(mticker.MultipleLocator(date_ticks))
        axis.yaxis.set_major_locator(mticker.MultipleLocator(date_ticks))
        plt.gcf().autofmt_xdate()

    else:
        # index as
        im = plt.imshow(nan_diag(contextual_mp),
                        cmap=palette,
                        origin="lower",
                        vmin=np.min(contextual_mp),
                        vmax=np.max(contextual_mp)
                        )
        plt.xlabel(xlabel)
        ticks = list(range(0, len(contextual_mp), int(len(contextual_mp) / index_ticks)))
        plt.xticks(ticks)
        plt.yticks(ticks)

    # Create an axes for colorbar. The position of the axes is calculated based on the position of ax.
    # You can change 0.01 to adjust the distance between the main image and the colorbar.
    # You can change 0.02 to adjust the width of the colorbar.
    # This practice is universal for both subplots and GeoAxes.
    plt.title(title)
    cax = figure.add_axes([axis.get_position().x1 + 0.01, axis.get_position().y0, 0.02, axis.get_position().height])
    cbar = plt.colorbar(im, cax=cax)  # Similar to fig.colorbar(im, cax = cax)
    cbar.set_label(legend_label)
