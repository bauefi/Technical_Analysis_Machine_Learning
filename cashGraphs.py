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

#Generate the maximum number of sample intervals from a given TimeSeries
def calculate_intervals(timeseries, interval_length):
    number_intervals = timeseries.shape[0] - interval_length
    intervals = np.empty([number_intervals, interval_length])

    interval = 0
    iteration = 0
    i = 0;

    while(i < timeseries.shape[0]-1):
        if(iteration == interval_length):
            iteration = 0
            interval += 1
            i -= interval_length-1
        intervals[interval][iteration] = timeseries[i]
        iteration += 1
        i += 1
    return intervals

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
    same = 0
    for i in range(test_set.shape[0]):
        prediction = k_nearest_neighbour_regression(test_set[i,:],training_set, training_labels, neighbors)
        prediction = prediction*100-100
        actual = test_labels[i]*100-100

        if(np.sign(prediction) == np.sign(actual)):
            same +=1
        print(str(prediction) + ' ' + str(actual))
    print("Correct sign %: " + str(float(same/len(test_set))))

#Main
data, meta_data = API_daily_close_time_series('SPX')
intervals = calculate_intervals(data, 30)
intervals_normalized = normalise_time_series(intervals)
intervals_normalized_labelled = labelled_time_series(intervals_normalized)
# print(intervals_normalized_labelled.shape)
#
# print(intervals_normalized_labelled[:,4840:])
test_values = intervals_normalized_labelled[0:50,]
training_values = intervals_normalized_labelled[50:4848,]
#
test_accuracy(test_values[:,0:30], test_values[:,30], training_values[:,0:30], training_values[:,30], 10)
