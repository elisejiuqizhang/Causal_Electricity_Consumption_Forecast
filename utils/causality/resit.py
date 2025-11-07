import numpy as np
import pandas as pd

import lingam
import graphviz
from lingam.utils import print_causal_directions, print_dagc

import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR



def create_resit(regressor_name='random_forest', random_state=None, alpha=0.01):

    assert regressor_name in ['random_forest', 'linear', 'svr'], "regressor_name must be one of ['random_forest', 'linear', 'svr']"

    if regressor_name == 'random_forest':
        regressor = RandomForestRegressor(random_state=random_state, max_depth=5)
    elif regressor_name == 'linear':
        regressor = LinearRegression()
    elif regressor_name == 'svr':
        regressor = SVR()

    resit=lingam.RESIT(regressor=regressor, alpha=alpha, random_state=random_state)
    
    return resit