import os, sys
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(ROOT_DIR)

import utils.data_utils.info_cities as info_cities

list_datetime=['is_holiday', 'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'doy_sin', 'doy_cos', 'month_sin', 'month_cos']

list_iseo_vars = ['TOTAL_CONSUMPTION', 'PREMISE_COUNT']

list_F0=list_datetime + list_iseo_vars
list_F1=list_F0+info_cities.list_era5_vars
list_F2=list_F0+['t2m_degC', 'tp_mm', 'tcw', 'avg_snswrf']
list_F3=list_F0+['t2m_degC', 'tp_mm', 'skt']