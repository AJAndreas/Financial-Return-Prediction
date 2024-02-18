
import pandas as pd
import numpy as np
import scipy
import statsmodels.api as sm
import sklearn
from sklearn import metrics
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
import yfinance as yf
import requests
import csv
import talib
import plotly
import plotly.express as px
import catboost as cb
import ipywidgets
from catboost_search import ts_tuning


from Vault import get_api_key
from api_pull import alpha_vantage_data

import feature_engineering
from feature_engineering import feature_engineering

from talib import BBANDS, RSI

# Selecting the interval for the data pull
# Once this has been run and the csv file created, this code can be blocked out to prevent duplicate api calls
# when this file is run

api_key = get_api_key()
time_frame = 'daily'
data_interval = '5min'

# Copy and paste the ticker_list below when prompted
# ticker_list = HWDN.L, AGR, LAND, JLL, UTG

"""
*** Section: Importing and Manipulating the Data ***

Importing the csv files we saved down previously and then doing some manipulations on the data including dropping 
some columns and rows, creating a multiindex so that we can slice the dataframe by stock and variable type

"""

# Commented out the below api_pull code post retrieving and saving down data into a csv
#api_pull = alpha_vantage_data(data_interval, time_frame, api_key)
#api_pull.save_data()

# Read csv file containing the OHLCV stock data
stock_df = pd.read_csv("stock_data_daily.csv")

# Creating a multiindex with Level 1: Stock Name and Level 2: The Variable
columns = tuple(zip(stock_df.columns.values, stock_df.iloc[0].values))

multiindex = pd.MultiIndex.from_tuples(columns, names=['Stocks', 'Attributes'])
stock_df.columns = multiindex

# Dropping some unnecessary rows
stock_df.drop(labels=[0, 1], axis="index", inplace=True)

# Set the index to a datetime object
stock_df.index = pd.to_datetime(stock_df['Stock', 'Attributes'].values)

# Drop irrelevant columns
stock_df.drop(columns=['Stock'], level=0, axis="columns", inplace=True)

# Casting the DataFrame data types as floats rather than strings. Because we import the file with the first row being
# the attributes (.e.g., 'Open', 'High', etc) I think this results in the rest of the data being interpreted as a string
stock_df = stock_df.astype(dtype=float, copy=False)

old_column_names = stock_df.columns
new_column_names = []

for idx, label in enumerate(old_column_names):
    ticker = old_column_names[idx][0].split('.')[0]
    feature = old_column_names[idx][1].split('.')[1].strip().title()
    column_labels = (ticker, feature)
    new_column_names.append(column_labels)

new_multiindex = pd.MultiIndex.from_tuples(tuple(new_column_names),
                                           names=['Stocks', 'Attributes'])
stock_df.columns = new_multiindex

"""
*** Section: Pre-Processing and Feature Engineering *** 

The next section of this script is going to focus on i) cleaning the data and ii) feature creation, from which we are 
going to consult Stefan Jensen's Machine Learning for Algorithmic Trading book to create some predictive signals 

"""

# Defining the lag periods we will use for the data
lag_period = [1, 2, 3, 6, 9, 12, 28]

# Retrieving the unique tickers from the dataframe
unique_tickers = stock_df.columns.get_level_values('Stocks').unique()

# Defining a list of the tickers and 'Close' which can be used to index the dataframe
target_columns = list((x, 'Close') for x in unique_tickers)
target_df = stock_df[target_columns].pct_change(1)

# Rename the column labels from 'Close' to 'One Day Return'. These will be our target variables
rename_dict = {'Close': 'One Day Return'}
target_df.rename(mapper=rename_dict, axis=1, inplace=True)

# Need to shift the dataframe to prevent data leakage - we want to use data at time 't' to predict the  at time 't+1'
stock_df = stock_df.shift(1)

# Instantiate the 'feature_engineering' class and compute the returns and momentum factors over the lag periods
feature_eng = feature_engineering()
return_df = feature_eng.pct_change(stock_df, target_columns, lag_period)

return_df = feature_eng.lagged_return_momentum(return_df, target_columns, lag_period)

# Let's join the two dataframes together now so we have the dependent and independent variables in the same place
# This makes specifying the training, validation and test splits easier
return_df = return_df.join(target_df, how='left')

"""
We are going to try and predict the sign of the stock return so we need to create a classifier output and to do this we 
will mark positive daily returns as a 1 and negative returns as a 0

The intuition would be that we want to buy a stock if the enxt day's return is positive and our model accurately 
predicts this

"""

for labels in target_columns:
    return_df[f'{labels[0]}', 'flag'] = np.where(return_df[f'{labels[0]}', 'One Day Return'] > 0, 1, 0)

"""
Section: Feature Selection 

We want to select which features are most predictive and we will test if there is a linear relationship 
To do this we are going to assume a linear regression model and run a t-test across the features 
The Null Hypothesis, H_0, would be that the slope is zero at a specified confidence interval. So if we fail to reject 
the Null, then we will incorporate this feature into the model

To be further investigated at a later stage

"""

"""
Section: Model Training 

We are going to use the CatBoost model as it has been shown to perform robustly on tabular data 
We will likely test other models and methodologies at a later point  

"""

# Creating training, validation and test datasets
# Time series data are ordered data meaning their sequence provides structure and importance, so we cannot blindly use
# Cross Validation or assume that the data is i.i.d
training_data = return_df.loc[:'2020']
#validation_data = return_df.loc['2020':'2021']
test_data = return_df['2021':"2023"]

# We need to separate our datasets into the dependent and independent variables
# We are going to do this just for a single ticker at the moment

ticker = unique_tickers[3]
print(ticker)
seed = 42
splits = 4

y_train = training_data[ticker, 'flag']
x_train = training_data[ticker].drop(columns=['flag', 'One Day Return'])

"""y_val = validation_data[ticker, 'flag']
x_val = validation_data[ticker].drop(columns=['flag', 'One Day Return'])"""

y_test = test_data[ticker, 'flag']
x_test = test_data[ticker].drop(columns=['flag', 'One Day Return'])


# Here we specify the CatBoostRegressor as our model
model = cb.CatBoostClassifier(custom_loss="Accuracy",
                              random_seed=seed,
                              logging_level='Silent',
                              iterations=25)


"""# Commented out after having run once - saved model is called optimized_model

# We want to find the optimal parameters for our model using the grid search methodology, but first we specify a
# set of parameters that we will search over
param_grid = {'iterations': [50, 100, 150],
        'learning_rate': [0.01, 0.1],
        'depth': [2, 4, 6, 8],
        'l2_leaf_reg': [0.2, 0.5, 1, 3]}

# Setting a random seed to create reproducible results
seed = 42
splits = 5

ts_tuning = ts_tuning(params=param_grid, seed=seed, splits=splits)
fitted_model = ts_tuning.search(x_train,
                                y_train,
                                model)"""


optimized_model = cb.CatBoostClassifier()
optimized_model.load_model('optimized_model')

y_pred = optimized_model.predict(x_test)
prediction_accuracy = np.sum(np.where(y_pred==y_test, 1, 0))/np.size(y_test)
print(prediction_accuracy)

















# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/

