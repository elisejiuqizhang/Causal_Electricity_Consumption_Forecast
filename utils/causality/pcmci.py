import tigramite
from tigramite import data_processing as pp
from tigramite import plotting as tp
from tigramite.pcmci import PCMCI
from tigramite.lpcmci import LPCMCI

from tigramite.independence_tests.parcorr import ParCorr
from tigramite.independence_tests.robust_parcorr import RobustParCorr
from tigramite.independence_tests.parcorr_wls import ParCorrWLS   

def create_pcmci(dataframe, var_names=None, robust=False, wls=False):
    """
    Create a PCMCI or LPCMCI object from a pandas DataFrame.

    Parameters
    ----------
    dataframe : pd.DataFrame
        The input data with time series variables as columns.
    var_names : list of str, optional
        List of variable names corresponding to the columns of the dataframe.
        If None, column names from the dataframe will be used.
    time_lag_max : int, default 1
        The maximum time lag to consider for causal relationships.
    robust : bool, default False
        If True, use RobustParCorr as the independence test.
    wls : bool, default False
        If True, use ParCorrWLS as the independence test.

    Returns
    -------
    pcmci : PCMCI or LPCMCI object
        The initialized PCMCI or LPCMCI object.
    """
    if var_names is None:
        var_names = list(dataframe.columns)

    data_array = dataframe.values

    data = pp.DataFrame(data_array, var_names=var_names)

    if robust:
        indep_test = RobustParCorr(significance='analytic')
    elif wls:
        indep_test = ParCorrWLS(significance='analytic')
    else:
        indep_test = ParCorr(significance='analytic')

    # if time_lag_max > 1:
    #     pcmci = LPCMCI(dataframe=data, cond_ind_test=indep_test, verbosity=1)
    # else:
    #     pcmci = PCMCI(dataframe=data, cond_ind_test=indep_test, verbosity=1)

    pcmci = PCMCI(dataframe=data, cond_ind_test=indep_test, verbosity=1)

    return pcmci




