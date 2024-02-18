
import pandas as pd
import numpy as np
import sklearn
from sklearn import metrics
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
import catboost as cb
import ipywidgets

class ts_tuning:
    def __init__(self, params, seed, splits):
        self.param_grid = params
        self.seed = seed
        self.splits = splits

    def search(self, x, y, model):
        model.grid_search(self.param_grid,
                                    x,
                                    y,
                                    cv=TimeSeriesSplit(n_splits=self.splits),
                                    search_by_train_test_split=False,
                                    partition_random_seed=self.seed,
                                    shuffle=False,
                                    train_size=1,
                                    verbose=False,
                                    plot=True)

        model.save_model('optimized_model',
                      format='cbm')

        return model

