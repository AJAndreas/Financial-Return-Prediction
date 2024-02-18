
import numpy as np
import pandas as pd


class feature_engineering:

    def resample_data_monthly(self, data, column_name):
        """
        :param data: (pd.Dataframe) The dataframe which contains the raw price data which we will be resampling.
        :param column_name: (String) This is the column name or index which we would perform the resampling on.
                            Note that this will be the 'Close Price' in this example

        :return:
            pd.DataFrame of the resampled data

        Note that this would return a new dataframe on the desired resampled timeframe
        """

        monthly_df = pd.DataFrame(index=column_name)

        monthly_prices = data[column_name].resample('M').last()

        monthly_df[column_name] = monthly_prices

        return monthly_df

    def pct_change(self, data, column_names, lags):
        """

        :param: data (pd.DataFrame) The raw price data which we will be using to compute the returns
        :param: column_name (List) A list containing the multi-level indexing of the column names.
                The first level is the stock ticker and the second level the market data ('Open', 'High', etc.)
                Note that this will be the 'Close' in this example
        :param: (List) A list of integers which specifies the periods over which we compute the percentage
        price change

        :return: pd.DataFrame of the computed returns

        """

        for labels in column_names:
            for lag in lags:
                data[f'{labels[0]}', f'{lag}_return'] = data[labels].pct_change(lag).add(1).pow(1/lag).sub(1)

        return data

    def lagged_return_momentum(self, data, column_name, lags):

        """

        :param: data (pd.DataFrame) The raw price data which we will be using to compute the returns
        :param: column_name (List) A list containing the multi-level indexing of the column names.
                The first level is the stock ticker and the second level the market data ('Open', 'High', etc.)
                Note that this will be the 'Close' in this example
        :param: (List) A list of integers which specifies the periods over which we compute the percentage
        price change

        :return: pd.DataFrame of the computed returns

        Objective: Compute and append momentum factors for the given data into the existing dataframe
        """

        # Computes the returns over a specified lag period and returns a new pandas dataframe
        return_df = self.pct_change(data, column_name, lags)

        # We exclude the case where lag = 1 otherwise this would return 1 which is uninformative
        for labels in column_name:
            for lag in lags:
                if lag != 1:
                    return_df[f'{labels[0]}', f'{lag}_momentum'] = return_df[f'{labels[0]}', f'{lag}_return'].sub(return_df[f'{labels[0]}', '1_return'])
                elif lag == 1:
                    pass

        return return_df





