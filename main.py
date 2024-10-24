import pandas as pd
import dask.dataframe as dd
import dask.array as da
import dask.bag as db
import numpy as np
import dask
from datetime import time
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from matplotlib.ticker import ScalarFormatter
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from dask_ml.model_selection import train_test_split
import joblib
from dask.distributed import Client


def get_data():
    # pd_full = pd.read_csv("Trips_Full Data.csv")
    # pd_trips = pd.read_csv("Trips_by_Distance.csv")
    pd.options.display.float_format = '{:.2f}'.format  # disables scientific notation
    dd_full = dd.read_csv("Trips_Full Data.csv", assume_missing=True).fillna(0)
    dd_full.compute()
    dd_trips = dd.read_csv("Trips_by_Distance.csv", dtype={'County Name': 'object',
                                                           'Number of Trips': 'float64',
                                                           'Number of Trips 1-3': 'float64',
                                                           'Number of Trips 10-25': 'float64',
                                                           'Number of Trips 100-250': 'float64',
                                                           'Number of Trips 25-50': 'float64',
                                                           'Number of Trips 250-500': 'float64',
                                                           'Number of Trips 3-5': 'float64',
                                                           'Number of Trips 5-10': 'float64',
                                                           'Number of Trips 50-100': 'float64',
                                                           'Number of Trips <1': 'float64',
                                                           'Number of Trips >=500': 'float64',
                                                           'Population Not Staying at Home': 'float64',
                                                           'Population Staying at Home': 'float64',
                                                           'State Postal Code': 'object'}).fillna(0)
    # todo check everything works by using dd_trips.compute()
    dd_trips["Date"] = dd.to_datetime(dd_trips["Date"])
    dd_full["Date"] = dd.to_datetime(dd_full["Date"])
    # dd_full = dd.from_pandas(pd_full)
    # dd_trips = dd.from_pandas(pd_trips)
    return dd_full, dd_trips


# a) DASK: average number of people staying home
def avg_num_staying_home(dd_trips):
    dd_numpeopleathome_byweek = dd_trips.groupby("Week")['Population Staying at Home'].mean()
    dd_numpeopleathome_byweek.compute().plot(kind='bar', ylabel='Population (million)',
                                             title='Population staying Home by Week')
    plt.show()


# DASK: how far are people travelling when they don't stay home?
def distance_people_travelling(dd_full):
    df_distances = dd_full[['Trips 1-25 Miles',
                            'Trips 1-3 Miles', 'Trips 10-25 Miles', 'Trips 100-250 Miles',
                            'Trips 100+ Miles', 'Trips 25-100 Miles', 'Trips 25-50 Miles',
                            'Trips 250-500 Miles', 'Trips 3-5 Miles', 'Trips 5-10 Miles',
                            'Trips 50-100 Miles', 'Trips 500+ Miles', 'Week of Date']].groupby(
        'Week of Date').mean()
    df_distances.compute().plot(kind='bar', ylabel='Number of trips')
    plt.show()


# b) Identify the dates that > 10 000 000 people conducted 10-25 Number of Trips and
# compare them to them that the same number of people (> 10 000 000) conducted 50-
# 100 Number of trips.
def dates_10_25(dd_trips):
    df_trips_10_25 = dd_trips[dd_trips["Number of Trips 10-25"] > 10e6]
    df_trips_10_25.loc[:, 'Date'] = pd.to_datetime(df_trips_10_25['Date'])
    ax = df_trips_10_25.plot(kind='scatter', x='Date', y='Number of Trips 10-25',
                             ylabel='Number of Trips', title='Number of People making 10-25 Trips')
    ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
    plt.show()


def dates_50_100(dd_trips):
    df_trips_50_100 = dd_trips[dd_trips["Number of Trips 50-100"] > 10e6]
    df_trips_50_100.loc[:, 'Date'] = pd.to_datetime(df_trips_50_100['Date'])
    ax = df_trips_50_100.plot(kind='scatter', x='Date', y='Number of Trips 50-100',
                              ylabel='Number of Trips',
                              title='Number of People making 50-100 Trips')
    ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
    plt.show()


# 10/20 processors

def parallel_proc(dd_trips, dd_full):
    # define number of processors
    n_processors = [10, 20]
    n_processors_time = {}  # define n_processors_time dictionary
    for processor in n_processors:
        start_time = time.time()
        # code for question a):
        dd_numpeopleathome_byweek = dd_trips.groupby("Week")['Population Staying at Home'].mean()
        dd_numpeopleathome_byweek.compute().plot(kind='bar', ylabel='Population (million)',
                                                 title='Population staying Home by Week')
        plt.show()

        dd_full['Week'].nunique()
        df_distances = dd_full[['Trips 1-25 Miles',
                                'Trips 1-3 Miles', 'Trips 10-25 Miles', 'Trips 100-250 Miles',
                                'Trips 100+ Miles', 'Trips 25-100 Miles', 'Trips 25-50 Miles',
                                'Trips 250-500 Miles', 'Trips 3-5 Miles', 'Trips 5-10 Miles',
                                'Trips 50-100 Miles', 'Trips 500+ Miles', 'Week of Date']].groupby(
            'Week of Date').mean()
        df_distances.plot(kind='bar', ylabel='Number of trips')
        plt.show()

        # code for question b):
        df_trips_10_25 = dd_trips[dd_trips["Number of Trips 10-25"] > 10e6]
        df_trips_10_25.loc[:, 'Date'] = pd.to_datetime(df_trips_10_25['Date'])
        ax = df_trips_10_25.plot(kind='scatter', x='Date', y='Number of Trips 10-25',
                                 ylabel='Number of Trips',
                                 title='Number of People making 10-25 Trips')
        ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
        plt.show()

        df_trips_50_100 = dd_trips[dd_trips["Number of Trips 50-100"] > 10e6]
        df_trips_50_100.loc[:, 'Date'] = pd.to_datetime(df_trips_50_100['Date'])
        ax = df_trips_50_100.plot(kind='scatter', x='Date', y='Number of Trips 50-100',
                                  ylabel='Number of Trips',
                                  title='Number of People making 50-100 Trips')
        ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
        plt.show()

        dask_time = time.time() - start_time
        n_processors_time[processor] = dask_time
        print(n_processors_time)


pd.options.display.expand_frame_repr = False


def get_x_y(dd_trips: dd.DataFrame, dd_full: dd.DataFrame):
    # Careful that WEEK needs to be equal to 31 to have the matching week in X because it's 0 indexed
    y: dd.DataFrame = \
        dd_trips[(dd_trips["Week"] == 31) & (dd_trips["Date"].dt.year == 2019)].groupby("Date")[
            ["Number of Trips 5-10", "Number of Trips 10-25"]].sum()
    x: dd.DataFrame = dd_full[['Trips 1-25 Miles']]
    x = x.set_index(dd_full["Date"])
    # this step below is important to clean the computational graph and start fresh
    x = dd.from_pandas(x.compute())
    y = dd.from_pandas(y.compute())
    assert y.iloc[:, 0].compute() is not None
    assert x.iloc[:, 0].compute() is not None
    return x, y


def select_column_by_index(df, idx):
    return df.iloc[:, idx]


def linear_regression(x, y):
    with joblib.parallel_backend('dask'):
        # Linear regression
        for i in range(y.shape[1]):
            model = LinearRegression()
            selected_column = y.map_partitions(select_column_by_index, idx=i)
            model.fit(x.to_frame(), selected_column)
            y_hat = pd.DataFrame(model.predict(x.to_frame()))
            selected_column_pd = selected_column.compute()
            y_hat.set_index(selected_column_pd.index, inplace=True)
            error = pd.DataFrame(y_hat.iloc[:, 0] - selected_column_pd)
            column = y.columns[i]
            # print(f'{column}:  {error.iloc[:,0]}')
            predictions_df = pd.concat(
                [selected_column_pd, y_hat, error.div(selected_column_pd, axis=0) * 100.0], axis=1)
            predictions_df = predictions_df.set_axis([column, "y_hat", "Error %"], axis=1)
            predictions_df.iloc[:, [0, 1]].plot()
            plt.show()


def linear_regression_polyfeatures(x, y):
    with joblib.parallel_backend('dask'):
        # Linear regression with PolynomialFeatures
        for i in range(y.shape[1]):
            transformer = PolynomialFeatures(degree=2, include_bias=False)
            transformer.fit(x.to_frame())
            new_x = transformer.transform(x.to_frame())
            model = LinearRegression()
            selected_column = y.map_partitions(select_column_by_index, idx=i)
            model.fit(new_x, selected_column)
            y_hat = pd.DataFrame(model.predict(new_x))
            selected_column_pd = selected_column.compute()
            y_hat.set_index(selected_column_pd.index, inplace=True)
            error = pd.DataFrame(y_hat.iloc[:, 0] - selected_column_pd)
            column = y.columns[i]
            # print(f'{column}:  {error}')
            predictions_df = pd.concat(
                [selected_column_pd, y_hat, error.div(selected_column_pd, axis=0) * 100.0], axis=1)
            predictions_df = predictions_df.set_axis([column, "y_hat", "Error %"], axis=1)
            predictions_df.iloc[:, [0, 1]].plot()
            plt.show()


def multi_linear_regression(X_train:dd.DataFrame, y_train:dd.DataFrame, X_test:dd.DataFrame, y_test:dd.DataFrame, col_index):
    # Multiple Linear regression

    model = LinearRegression()
    selected_column_train = y_train.iloc[:, col_index]
    model.fit(X_train, selected_column_train)

    selected_column_test = y_test.iloc[:, col_index]
    selected_column_pd_test = selected_column_test.compute()
    y_hat = pd.DataFrame(model.predict(X_test))
    y_hat.set_index(selected_column_pd_test.index, inplace=True)
    error = pd.DataFrame(y_hat.iloc[:, 0] - selected_column_pd_test)
    column = y_test.columns[col_index]
    # print(f'{column}:  {error}')
    predictions_df = pd.concat([selected_column_pd_test, y_hat,
                                error.div(selected_column_pd_test, axis=0) * 100.0], axis=1)
    predictions_df = predictions_df.set_axis([column, "y_hat", "Error %"], axis=1)
    predictions_df.iloc[:, [0, 1]].plot()
    # plt.show()
    # error.plot()
    # plt.title(f"Error for {column}")
    # plt.show()
    return error.pow(2).mean()


# ML with Dask
def train(x: dd.DataFrame, y: dd.DataFrame):
    with joblib.parallel_backend('dask'):
        epochs = 5
        for col_index in range(len(y.columns)):
            loss_cumulative = 0
            for i in range(epochs):
                j = 0
                while True:
                    X_train, X_test, y_train, y_test = train_test_split(x,
                                                                        y,
                                                                        test_size=0.2, random_state=i+j)
                    if len(y_test)!=0:
                        break
                    else:
                        j += 1
                loss = multi_linear_regression(X_train, y_train, X_test, y_test, col_index)
                loss_cumulative += loss
            avg_loss = loss_cumulative / epochs
            print(f"Average loss{y.columns[col_index]}: {avg_loss}")


if __name__ == '__main__':
    client = Client()
    dd_full, dd_trips = get_data()
    x, y = get_x_y(dd_trips, dd_full)
    train(x, y)

# pd_trips["Number of Trips 10-25"] = number of people who did 10-25 trips for that date
# pd_full['Trips 10-25 Miles'] = number of people who travelled between 10-25 miles


#     # train the model using X_train, y_train and test on X_test and y_test
