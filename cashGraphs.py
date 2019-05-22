import matplotlib.pyplot as plt
import numpy as np
from alpha_vantage.timeseries import TimeSeries
from sklearn.neighbors import KNeighborsRegressor


def generate_graph(return_series, stamp):
    fig = plt.plot(return_series, color='k')
    plt.axis('off')
    plt.savefig('graphs/return_chart_' + str(stamp) + '.png')

def print_return_matrix(return_matrix):
    for i in range(return_matrix.shape[0]):
        generate_graph(return_matrix[i,:], i)

#Returns numpy array instead of this pandas nonsense
#Most recent date is at bottom of array
def API_daily_close_time_series(ticker):
    ts = TimeSeries(key='15BJH4XND0N7L3KL', output_format='pandas')
    data, meta_data = ts.get_daily(symbol=ticker, outputsize='full')
    return data['4. close'].values, meta_data

def calculate_30_days_series(daily_returns_matrix):
    number_intervals = int(daily_returns_matrix.shape[0]/30)
    series_30_days_trailing = np.empty([number_intervals, 30])
    #All intervals start at 1
    series_30_days_trailing[:,0] = 1
    #Fill the out the 30 day timeseries -->Needs to be trailing
    interval = 0;
    day = 0;

    for i in range(number_intervals * 30):
        if(day == 30):
            day = 0
            interval += 1
        series_30_days_trailing[interval][day] = daily_returns_matrix[i]
        day += 1
    return series_30_days_trailing

def normalise_time_series(time_series_matrix):
    number_intervals = time_series_matrix.shape[0]
    series_length = time_series_matrix.shape[1]
    return_matrix = np.array(time_series_matrix, copy=True)

    for i in range(number_intervals):
        base = return_matrix[i][0]
        for j in range(series_length):
            return_matrix[i][j] = return_matrix[i][j] / base
    return return_matrix

def labelled_time_series(return_series):
    intervals = return_series.shape[0]
    days = return_series.shape[1]
    #Need to slice of the last period and add a column for the label
    labelled_return_series = np.empty([intervals - 1, days + 1])
    labelled_return_series[:,0:days] = return_series[0:intervals-1,:]

    #Add the subsequents month return as label
    for i in range(intervals-1):
        labelled_return_series[i][days] = return_series[i+1][days-1]

    return labelled_return_series

def k_nearest_neighbour_regression(regressor, training_set, labels, neighbors):
    neigh = KNeighborsRegressor(n_neighbors=neighbors)
    neigh.fit(training_set, labels)
    return neigh.predict([regressor])

def test_accuracy(test_set, test_labels, training_set, training_labels, neighbors):

    for i in range(test_set.shape[0]):
        temp_result = k_nearest_neighbour_regression(test_set[i,:],training_set, training_labels, neighbors)
        print(str(temp_result*100-100) + ' ' + str(test_labels[i]*100-100))

#Main
data, meta_data = API_daily_close_time_series('DAX')
trailing_30_days = calculate_30_days_series(data)
normalised_30_days_trailing = normalise_time_series(trailing_30_days)
normalised_30_days_trailing_labeled = labelled_time_series(normalised_30_days_trailing)
test_values = normalised_30_days_trailing_labeled[0:20,]
training_values = normalised_30_days_trailing_labeled[20:161,]

test_accuracy(test_values[:,0:30], test_values[:,30], training_values[:,0:30], training_values[:,30], 5)
