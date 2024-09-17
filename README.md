# Contextual Matrix Profile Calculation Tool

Matrix Profile is an algorithm capable to discover motifs and discords in time series data. It is a powerful tool that
by calculating the (z-normalized) Euclidean distance between any subsequence within a time series and its nearest
neighbor it is able to provide insights on potential anomalies and/or repetitive patterns. In the field of building
energy management it can be employed to detect anomalies in electrical load timeseries.

This tool is a Python implementation of the Matrix Profile algorithm that employs contextual information (such as
external air temperature) to identify abnormal pattens in electrical load subsequences that start in predefined sub
daily time windows, as shown in the following figure.

![](./docs/example.png)

**Table of Contents**

* [Usage](#usage)
    * [Data format](#data-format)
    * [Run locally](#run-locally)
    * [Run with Docker](#run-with-docker)
* [Additional Information](#additional-information)
* [Cite](#cite)
* [Contributors](#contributors)
* [License](#license)

## Usage

The tool comes with a CLI that helps you to execute the script with the desired commands

```console 
$ python -m src.cmp.main -h

Matrix profile

positional arguments:
  input_file     Path to file
  variable_name  Variable name
  output_file    Path to the output file

options:
  -h, --help     show this help message and exit
```

The arguments to pass to the script are the following:

* `input_file`: The input dataset via an HTTP URL. The tool should then download the dataset from that URL; since it's a
  pre-signed URL, the tool would not need to deal with authentication—it can just download the dataset directly.
* `variable_name`: The variable name to be used for the analysis (i.e., the column of the csv that contains the
  electrical load under analysis).
* `output_file`: The local path to the output HTML report. The platform would then get that HTML report and upload it to
  the object
  storage service for the user to review later.

You can run the main script through the console using either local files or download data from an external url. This
repository comes with a sample dataset ([data.csv](.src/cmp/data/data.csv)) that you can use to generate a report and
you can pass the local path
as `input_file` argument as follows:

### Data format

The tool requires the user to provide a csv file as input that contains electrical power timeseries for a specific
building, meter or energy system (e.g., whole building electrical power timeseries). The `csv` is a wide table format as
follows:

```csv
timestamp,column_1,temp
2019-01-01 00:00:00,116.4,-0.6
2019-01-01 00:15:00,125.6,-0.9
2019-01-01 00:30:00,119.2,-1.2
```

The csv must have the following columns:

- `timestamp` [case sensitive]: The timestamp of the observation in the format `YYYY-MM-DD HH:MM:SS`. This column is
  supposed to be in
  UTC timezone string format. It will be internally transformed by the tool into the index of the dataframe.
- `temp` [case sensitive]: Contains the external air temperature in Celsius degrees. This column is required to perform
  thermal sensitive
  analysis on the electrical load.
- `column_1`: Then the dataframe may have `N` arbitrary columns that refers to electrical load time series. The user has
  to specify the column name that refers to the electrical load time series in the `variable_name` argument.

### Run locally

Create virtual environment and activate it and install dependencies:

- Makefile
  ```bash
  make setup
  ```

- Linux:
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  pip install poetry
  poetry install
  ```
- Windows:
  ```bash
  python -m venv venv
  venv\Scripts\activate
  pip install poetry
  poetry install
  ```

Now you can run the script from the console by passing the desired arguments. In the following we pass the sample
dataset [`data.csv`](src/cmp/data/data.csv) as input file and the variable `Total_Power` as the variable name to be used
for the analysis. The output file will be saved in the [`results`](src/cmp/results) folder.

```console
$ python -m src.cmp.main src/cmp/data/data.csv Total_Power src/cmp/results/reports/report.html

2024-08-13 12:45:42,821 [INFO](src.cmp.utils) ⬇️ Downloading file from <src/cmp/data/data.csv>
2024-08-13 12:45:43,070 [INFO](src.cmp.utils) 📊 Data processed successfully

*********************
CONTEXT 1 : Subsequences of 05:45 h (m = 23) that start in [00:00,01:00) (ctx_from00_00_to01_00_m05_45)
99.997%        0.0 sec

- Cluster 1 (1.660 s)   -> 1 anomalies
- Cluster 2 (0.372 s)   -> 3 anomalies
- Cluster 3 (0.389 s)   -> 4 anomalies
- Cluster 4 (0.593 s)   -> 5 anomalies
- Cluster 5 (-)         -> no anomalies green

[...]

2024-08-13 12:46:27,187 [INFO](__main__) TOTAL 0 min 44 s
2024-08-13 12:46:32,349 [INFO](src.cmp.utils) 🎉 Report generated successfully on src/cmp/results/reports/report.html

```

At the end of the execution you can find the report in the path specified by the `output_file` argument, in this case
you will find it in the [`results`](src/cmp/results) folder.

### Run with Docker

Build the docker image.

- Makefile
  ```bash
  make docker-build
  ```
- Linux:
  ```bash
  docker build -t cmp .
  ```

Run the docker image with the same arguments as before

- Makefile
  ```bash
  make docker-run
  ```
- Linux:
  ```bash
  docker run cmp data/data.csv Total_Power results/reports/report.html
  ```

At the end of the execution you can find the results in the [`results`](src/cmp/results) folder inside the docker
container.

## Cite

You can cite this work by using the following reference or either though [this Bibtex file](./docs/ref.bib) or the
following plain text citation

> Chiosa, Roberto, et al. "Towards a self-tuned data analytics-based process for an automatic context-aware detection
> and
> diagnosis of anomalies in building energy consumption timeseries." Energy and Buildings 270 (2022): 112302.

## Contributors

- Author [Roberto Chiosa](https://github.com/RobertoChiosa)

## References

- Series Distance Matrix repository (https://github.com/predict-idlab/seriesdistancematrix)
- Stumpy Package (https://stumpy.readthedocs.io/en/latest/)

## License

This code is licensed under the MIT License - see the [LICENSE](LICENSE.md) file for details.
