# for a given dataframe with a datetime column, add new columns for hour of day, day of week, day of month, day of year, quarter, month, is_weekend, is_holiday (here consider Ontario holidays)

import numpy as np
import pandas as pd
from typing import Iterable, Optional, Union, Mapping, Callable

def add_hourly_calendar_features(
    df: pd.DataFrame,
    time_col: str,
    *,
    tz: Optional[str] = None,
    holidays: Optional[Union[Iterable, Mapping, Callable]] = None,
    drop_original: bool = False,
) -> pd.DataFrame:
    """
    Adds: hour, day_of_week, day_of_month, day_of_year, quarter, month,
    is_weekend, is_holiday (optional).

    Parameters
    ----------
    df : DataFrame
    time_col : str
        Name of the timestamp column.
    tz : str, optional
        If provided, convert timestamps to this timezone before feature extraction.
    holidays : iterable/mapping/callable, optional
        - Iterable of date-likes (YYYY-MM-DD, datetime.date, Timestamp.date)
        - Mapping (keys are dates; values ignored)
        - Callable(date) -> bool
    drop_original : bool, default False
        If True, drop the original time column.

    Returns
    -------
    DataFrame with new columns appended.
    """
    out = df.copy()
    ts = pd.to_datetime(out[time_col], utc=True, errors="coerce")
    if tz:
        ts = ts.dt.tz_convert(tz)
    # use naive for feature ops
    ts = ts.dt.tz_localize(None)

    out["hour"]          = ts.dt.hour
    out["day_of_week"]   = ts.dt.dayofweek        # Mon=0, Sun=6
    out["day_of_month"]  = ts.dt.day
    out["day_of_year"]   = ts.dt.day_of_year
    out["quarter"]       = ts.dt.quarter
    out["month"]         = ts.dt.month
    out["is_weekend"]    = out["day_of_week"].isin([5, 6]).astype("int8")

    if holidays is not None:
        dates = ts.dt.date
        if callable(holidays):
            out["is_holiday"] = dates.map(lambda d: bool(holidays(d))).astype("int8")
        else:
            try:
                holi_set = set(getattr(holidays, "keys", lambda: holidays)())
            except TypeError:
                holi_set = set(holidays)
            out["is_holiday"] = dates.isin(holi_set).astype("int8")

    if drop_original:
        out = out.drop(columns=[time_col])
    return out


# test
# file_path='/home/automation/elisejzh/Desktop/elisejzh/Projects/Mine/Causal_Electricity_Consumption_Forecast/data/era5/toronto_era5_timeseries.csv.gz'
# df=pd.read_csv(file_path, parse_dates=['time'])
# df_features = add_hourly_calendar_features(df, time_col='time', tz='America/Toronto', drop_original=False)
# print(df_features.head())
# print(df_features.keys()) # 'time', 'tcw', 'tcc', 't2m_degC', 'u10_ms', 'v10_ms', 'd2m_degC', 'ssr','net_sw_Wm2', 'tp_mm', 'e_mm', 'skt', 'lw_down_Wm2', 'lw_up_Wm2','net_lw_Wm2', 'net_radiation_Wm2', 'city', 'hour', 'day_of_week','day_of_month', 'day_of_year', 'quarter', 'month', 'is_weekend'

# file_path='/home/automation/elisejzh/Desktop/elisejzh/Projects/Mine/Causal_Electricity_Consumption_Forecast/data/ieso_hourly/ieso_residential_Ottawa.csv.gz'
# df=pd.read_csv(file_path, parse_dates=['TIMESTAMP'])
# df_features = add_hourly_calendar_features(df, time_col='TIMESTAMP', tz='America/Toronto', drop_original=False)
# print(df_features.head())
# print(df_features.keys()) # 'TIMESTAMP', 'TOTAL_CONSUMPTION', 'PREMISE_COUNT', 'hour','day_of_week', 'day_of_month', 'day_of_year', 'quarter', 'month','is_weekend'