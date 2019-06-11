import matplotlib.pyplot as plt
import numpy as np
from alpha_vantage.timeseries import TimeSeries
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import NearestNeighbors

# Arguments: Underlying, range, period
#
class PriceData:
    def __init__(self, ticker, range, period):
        #For now period is one day everytime
        #Input values
        self._ticker = ticker
        self._period = period
        self._range = range

        #Calculating the needed price data
        self._prices, meta_data = self.get_price_data_daily(ticker)
        intervals = self.calculate_intervals(self._prices, range)
        self._normalized_intervals = self.normalise_time_series(intervals)
        self._future_returns = self.calculate_future_returns_daily(self._prices, range)

    def get_price_data_daily(self, ticker):
        ts = TimeSeries(key='15BJH4XND0N7L3KL', output_format='pandas')
        data, meta_data = ts.get_daily(symbol=ticker, outputsize='full')
        return data['4. close'].values, meta_data

    def calculate_intervals(self, prices, interval_length):
        number_intervals = prices.shape[0] - interval_length
        intervals = np.empty([number_intervals, interval_length])

        interval = 0
        iteration = 0
        i = 0;

        while(i < prices.shape[0]-1):
            if(iteration == interval_length):
                iteration = 0
                interval += 1
                i -= interval_length-1
            intervals[interval][iteration] = prices[i]
            iteration += 1
            i += 1
        return intervals

    def normalise_time_series(self, time_series_matrix):
        number_intervals = time_series_matrix.shape[0]
        series_length = time_series_matrix.shape[1]
        return_matrix = np.array(time_series_matrix, copy=True)

        for i in range(number_intervals):
            base = return_matrix[i][0]
            for j in range(series_length):
                return_matrix[i][j] = return_matrix[i][j] / base
        return return_matrix

    #Price Data is daily data
    def calculate_future_returns_daily(self, prices, rangE):
        number_intervals = prices.shape[0]
        full_histories = number_intervals - 30
        #Loop through the thing
        future_returns = np.empty([full_histories, 4])

        #Start the loop at the beginning of the second interval
        i = 0

        for j in range(rangE, full_histories, rangE):
            #1, 2, 7, 30
            future_returns[i][0] = prices[j+1] / prices[j]
            future_returns[i][1] = prices[j+2] / prices[j]
            future_returns[i][2] = prices[j+7] / prices[j]
            future_returns[i][3] = prices[j+30] / prices[j]
            i = i + 1

        return future_returns

class Neighbours:
    def __init__(self, price_data, neighbours):
        self._price_data = price_data
        self._neighbors = neighbours
        self._nbrs = NearestNeighbors(n_neighbors=neighbours, algorithm='ball_tree')
        self._nbrs.fit(price_data._normalized_intervals);

    def find_nearest(self, datapoint):
        distances, indices = self._nbrs.kneighbors(datapoint.reshape(1,-1))
        neighbors_intervals = np.empty([self._neighbors, self._price_data._range])
        neighbors_future_returns = np.empty([self._neighbors, 4])

        for i in range(self._neighbors):
            neighbors_intervals[i] = self._price_data._normalized_intervals[indices[0][i]]
            neighbors_future_returns[i] = self._price_data._future_returns[indices[0][i]]

        return neighbors_intervals, neighbors_future_returns


def generate_graph(return_series, stamp):
    fig = plt.plot(return_series, color='k')
    plt.axis('off')
    plt.savefig('graphs/return_chart_' + str(stamp) + '.png')
    plt.close()


#Main
prices = PriceData('SPX', 30, 10);
neighbors = Neighbours(prices, 5)
results, returns = neighbors.find_nearest(prices._normalized_intervals[100]);
print(results)
print(returns)
