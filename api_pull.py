

import pandas as pd
import numpy as np
import scipy
import statsmodels
import sklearn
import yfinance as yf
import requests
import csv
import alpha_vantage
from alpha_vantage.timeseries import TimeSeries

from Vault import get_api_key


""""
This file is designed to be a self contained data pull from Alpha Vantage for financial data 
After instantiating the class, the user should first use the 'save_data' method 
which will create a csv file with Intraday data of the users chosen tickers

"""


class alpha_vantage_data:
    def __init__(self, interval, timeframe, api_key):
        self.interval = interval
        self.timeframe = timeframe
        self.api_key = api_key

    def user_prompt(self):
        """
        First, we want to ask the user to input a list of the tickers that they want to pull
        Next, we want to sk them over what timeframe we should pull the data (intraday vs. daily)
            
        :return: list of tickers, timeframe 
        """

        # First Question
        user_input = input("Please enter the ticker(s) you would like to pull data for (comma-separated): ")

        user_list = user_input.split(',')
        user_list = [item.strip() for item in user_list]

        # Second Question
        user_timeframe = input("Please enter the timeframe you would like to receive data over ('intraday' or 'daily'):")

        user_timeframe = user_timeframe.strip()

        return user_list, user_timeframe

    def get_data(self, symbol, timeframe):

        """ Using the Alpha Vantage API to pull in Intraday stock data for the SP500

        symbol: the ticker which you want to pull your data for
        interval: the time interval on which we will be pulling data. Acceptable inputs are as follows:
        1min, 5min, 15min, 30min, 60min
        api_key: Your specific API key for Alpha Vantage

        For further details please see the documentation: https://www.alphavantage.co/documentation/
        """

        # Initialize Alpha Vantage TimeSeries object
        ts = TimeSeries(key=self.api_key, output_format='pandas')

        # Get data depending on what the user selected
        if timeframe == 'intraday':
            data, meta_data = ts.get_intraday(symbol=symbol, interval=self.interval, outputsize='full')
        elif timeframe == 'daily':
            data, meta_data = ts.get_daily(symbol=symbol, outputsize='full')
        else:
            print("Error understanding data type. Please specify 'intraday' or 'daily'")

        # Rename columns for clarity
        data.columns = [f"{symbol}_{col}" for col in data.columns]

        return data

    def save_data(self):
        """
        We want to save down the data from the get_data method into a CSV file so that we can load this for future use
        We want to limit the number of api calls as the free version is capped at 25 calls per day

        :return: csv file with the relevant data
        """
        ticker_list, time_frame = self.user_prompt()

        # Create an empty DataFrame with the MultiIndex columns
        df = pd.DataFrame()

        # Loop through the list of stock tickers and make an api request for each ticker and append this to the
        # empty dataframe
        for ticker in ticker_list:
            raw_data = self.get_data(ticker, time_frame)
            df = pd.concat([df, raw_data], axis=1)
            print(f"{ticker} data pulled successfully")

        # Create a MultiIndex for columns
        multiindex = pd.MultiIndex.from_tuples(tuple(df.columns.str.split('_').values),
                                               names=['Stock', 'Attributes'])
        df.columns = multiindex

        csv_filename = f"stock_data_{self.timeframe}.csv"
        df.to_csv(csv_filename)
