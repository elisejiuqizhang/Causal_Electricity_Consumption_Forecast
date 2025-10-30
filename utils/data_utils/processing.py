import numpy as np
import pandas as pd

# min max scaling
def min_max_scale(series):
    """
    returns scaled series, min, max
    """
    min_val = series.min()
    max_val = series.max()
    scaled_series = (series - min_val) / (max_val - min_val)
    return scaled_series, min_val, max_val