# for a given dataframe with a datetime column, add new columns for hour of day, day of week, day of month, day of year, quarter, month, is_weekend, is_holiday (here consider Ontario holidays)

import numpy as np
import pandas as pd
from typing import Iterable, Optional, Union, Mapping, Callable
import holidays


def time_features(
        df: pd.DataFrame, 
        time_col: str ="time",
        tz: Optional[str] = None,
        holidays_info: Optional[Union[Iterable, Mapping, Callable]] = None,
        drop_original: bool = False
        ) -> pd.DataFrame:
    """Add cyclical datetime features. Assumes time is parseable to pandas datetime."""
    if not np.issubdtype(df[time_col].dtype, np.datetime64):
        df[time_col] = pd.to_datetime(df[time_col])

    if tz:
        df[time_col] = df[time_col].dt.tz_convert(tz)
        df[time_col] = df[time_col].dt.tz_localize(None)

    dt = df[time_col].dt
    df["hour"] = dt.hour
    df["dow"]  = dt.dayofweek
    df["doy"]  = dt.dayofyear
    df["month"]= dt.month

    # cyclical encodings
    df["hour_sin"] = np.sin(2*np.pi*df["hour"]/24.0)
    df["hour_cos"] = np.cos(2*np.pi*df["hour"]/24.0)
    df["dow_sin"]  = np.sin(2*np.pi*df["dow"]/7.0)
    df["dow_cos"]  = np.cos(2*np.pi*df["dow"]/7.0)
    df["doy_sin"] = np.sin(2*np.pi*df["doy"]/365.0)
    df["doy_cos"] = np.cos(2*np.pi*df["doy"]/365.0)
    df["month_sin"]= np.sin(2*np.pi*df["month"]/12.0)
    df["month_cos"]= np.cos(2*np.pi*df["month"]/12.0)
    
    # is_holiday (Ontario and Canada federal holidays)
    if holidays_info is not None:
        if callable(holidays_info):
            df["is_holiday"] = df[time_col].dt.date.map(lambda d: bool(holidays_info(d))).astype(int)
        else:
            try:
                holi_set = set(getattr(holidays_info, "keys", lambda: holidays_info)())
            except TypeError:
                holi_set = set(holidays)
            df["is_holiday"] = df[time_col].dt.date.isin(holi_set).astype(int)
    else:
        ca_on_holidays = holidays.country_holidays('CA', subdiv='ON') 
        # df["is_holiday"] = df[time_col].dt.date.isin(ca_on_holidays).astype(int)
        df["is_holiday"] = df[time_col].dt.date.map(lambda d: int(d in ca_on_holidays))

    if drop_original:
        df = df.drop(columns=[time_col])

    return df


# def add_time_features(
#     df: pd.DataFrame,
#     time_col: str,
#     *,
#     tz: Optional[str] = None,
#     holidays: Optional[Union[Iterable, Mapping, Callable]] = None,
#     drop_original: bool = False,
# ) -> pd.DataFrame:
#     """
#     Adds: hour, day_of_week, day_of_month, day_of_year, quarter, month,
#     is_weekend, is_holiday (optional) and cyclical features for hour, day of week, and month.

#     Parameters
#     ----------
#     df : DataFrame
#     time_col : str
#         Name of the timestamp column.
#     tz : str, optional
#         If provided, convert timestamps to this timezone before feature extraction.
#     holidays : iterable/mapping/callable, optional
#         - Iterable of date-likes (YYYY-MM-DD, datetime.date, Timestamp.date)
#         - Mapping (keys are dates; values ignored)
#         - Callable(date) -> bool
#     drop_original : bool, default False
#         If True, drop the original time column.

#     Returns
#     -------
#     DataFrame with new columns appended.
#     """
#     out = df.copy()
#     ts = pd.to_datetime(out[time_col], utc=True, errors="coerce")
#     if tz:
#         ts = ts.dt.tz_convert(tz)
#     ts = ts.dt.tz_localize(None)

#     out["hour"]          = ts.dt.hour
#     out["day_of_week"]   = ts.dt.dayofweek        # Mon=0, Sun=6
#     out["day_of_month"]  = ts.dt.day
#     out["day_of_year"]   = ts.dt.day_of_year
#     out["quarter"]       = ts.dt.quarter
#     out["month"]         = ts.dt.month
#     out["is_weekend"]    = out["day_of_week"].isin([5, 6]).astype("int8")

#     if holidays is not None:
#         dates = ts.dt.date
#         if callable(holidays):
#             out["is_holiday"] = dates.map(lambda d: bool(holidays(d))).astype("int8")
#         else:
#             try:
#                 holi_set = set(getattr(holidays, "keys", lambda: holidays)())
#             except TypeError:
#                 holi_set = set(holidays)
#             out["is_holiday"] = dates.isin(holi_set).astype("int8")

#     # Add cyclical encodings
#     out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24.0)
#     out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24.0)
#     out["dow_sin"]  = np.sin(2 * np.pi * out["day_of_week"] / 7.0)
#     out["dow_cos"]  = np.cos(2 * np.pi * out["day_of_week"] / 7.0)
#     out["month_sin"] = np.sin(2 * np.pi * out["month"] / 12.0)
#     out["month_cos"] = np.cos(2 * np.pi * out["month"] / 12.0)

#     if drop_original:
#         out = out.drop(columns=[time_col])
#     return out
