# using GRU to forecast per city consumption, following a rolling time-series cross validation setup
# available data from 20180101 to 20240310, each split is a one year history as train-val, and 2 months period for testing; keep rolling forward
# exp/train/train_forecast.py
import os, sys, json, math, argparse, random
ROOT= os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(ROOT)


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler, MinMaxScaler

from utils.data_utils.info_cities import list_cities, dict_regions, list_vars, list_era5_vars, list_ieso_vars
from utils.data_utils.datetime_utils import time_features
from utils.data_utils.info_features import list_datetime, list_F1, list_F2, list_F3
from utils.data_utils.processing import TargetScaler, split_by_time, build_windows

from utils.forecast.rnn import GRUForecaster

# Paths
OUTPUT_DIR = os.path.join(ROOT, 'outputs', 'forecast', 'per_region', 'gru_per_city')
os.makedirs(OUTPUT_DIR, exist_ok=True)
DATA_DIR = os.path.join(ROOT, 'data', 'ieso_era5')
DATA_FILE_PREFIX='combined_ieso_era5_avg_'