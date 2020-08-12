#######################################################################################################################
#  Stock Prediction helper consists of helper funcitons to calculcate the Moving Average, Exponential Moving Average, #
#  Moving Average Convergence Divergence for the stock market data. Also predetermined number of days data can be     #
#  obtained to proceed ML Algorithms. Tecnical terms and phrases are taken from investopedia.com.                     #
#######################################################################################################################


import numpy as np


def calc_mova(arr, period_of_time):
    """
    Returns Moving Average of passed array in terms of period of time.

    A moving average (MA) is an indicator that smooths out price trends by filtering
    out the “noise” from random short-term price fluctuations.

    Parameters
    ----------
    arr : numpy array or list of float/int
        Array to be calculated moving average on.

    days : int
        Period of time to calculcate Moving Average.
    """
    mova_list = [np.NaN for i in range(period_of_time-1)]
    for i in range(len(arr)-period_of_time+1):
        mova_list.append(arr[i:i+period_of_time].mean())
    return mova_list


def calc_ema(arr, period_of_time):
    """
    Returns Exponential Moving Average of passed array in terms of period of time.

    An exponential moving average (EMA) is a type of moving average (MA) that places 
    a greater weight and significance on the most recent data points.

    Parameters
    ----------
    arr : numpy array or list of float/int
        Array to be calculated Exponential Moving Average on.

    days : int
        Period of time to calculcate Exponential Moving Average.
    """
    multiplier = (2/(period_of_time+1))
    ema_list = [np.NaN for i in range(period_of_time-1)]

    ema_list.append(calc_mova(arr[:period_of_time], period_of_time)[-1])

    for i in range(period_of_time, len(arr)):

        ema = arr[i]*multiplier + (1-multiplier)*ema_list[i-1]
        ema_list.append(ema)

    return ema_list


def calc_macd_redline(arr, ema_day=26, macd_day=9):
    """
    Returns MACD Red Line of passed MACD Blueline in terms of period of ema_day and macd_day.

    Moving Average Convergence Divergence (MACD) is a trend-following 
    momentum indicator that shows the relationship between two moving 
    averages of a security’s price. A nine-day EMA of the MACD called 
    the MACD Redline.

    Parameters
    ----------
    arr : numpy array or list of float/int (MACD Blueline should be passed)
        Array to be calculated MACD Redline on.

    ema_day : 
    """

    macd_red = [np.NaN for i in range(ema_day)]
    macd_red = macd_red + (calc_ema(arr[ema_day:], macd_day))

    return macd_red


def calc_macd_blueline(arr, ema1=12, ema2=26):
    """
    Returns MACD Blueline of passed array in terms of 2 Exponential Moving Average values.

    The MACD Line is calculated by subtracting the 26-period 
    Exponential Moving Average (EMA) from the 12-period EMA.

    Parameters
    ----------
    arr : numpy array or list of float/int
        Array to be calculated MACD Line on.

    ema1 : int
        Firts  value to calculate MACD => EMA(ema1) - EMA(ema2)

    ema2 : int
        Second value to calcucalte MACD => EMA(ema1) - EMA(ema2)  
    """

    diff = np.array(calc_ema(arr, ema1)) - \
        np.array(calc_ema(arr, ema2))
    return diff


def transpose_data_for_train(df, label, period_of_time, droplist):
    """
    Returns (n_row-period_of_time+1, n_row*(m_column-1)+1) shaped Dataframe
    of (n_row, m_column) shaped Dataframe.

    Parameters
    ----------
    df : Dataframe
        Dataframe to transpose its shape to create larger dataset for ML Models.

    label : str
        Column name of the label in the Dataframe

    period_of_time: int
        Number of the rows to transpose.

    droplist : List(str) 
        List of column names to drop from Dataframe.


    """

    if label not in droplist:
        droplist.append(label)

    target = df[label].values
    target = target[period_of_time-1:]

    df = df.drop(droplist, axis=1).values
    arr = df[0:period_of_time].ravel()

    for i in range(1, len(df)-period_of_time+1):
        arr = np.vstack((arr, df[i:i+period_of_time].ravel()))

    target = target.reshape((len(target), 1))
    arr = np.hstack((arr, target))
    return arr


def transpose_data_for_test(df, period_of_time, droplist):
    """
    Returns (n_row-period_of_time+1, n_row*m_column) shaped Dataframe
    of (n_row, m_column) shaped Dataframe.

    Parameters
    ----------
    df : Dataframe
        Dataframe to transpose its shape to create larger dataset for ML Models.

    period_of_time: int
        Number of the rows to transpose.

    droplist : List(str) 
        List of column names to drop from Dataframe.
    """
    df = df.drop(droplist, axis=1).values
    arr = df[0:period_of_time].ravel()

    for i in range(1, len(df)-period_of_time+1):
        arr = np.vstack((arr, df[i:i+period_of_time].ravel()))

    return arr


def is_profitable(prices, period_of_time, profit=1.05):
    """
    Returns array of ones and zeros that indicates profitability of the passed array.

    Calculate and returns whether the prices is higher than the profit value
    in consecutive period of time or not.  

    Parameters
    ----------
    prices : numpy array or list of float/int
        Array to calculate on the profitablitiy .

    period_of_time : int
        Period of time onward to calculate the profitability.

    profit : float
        Profit margin, default is %5.
    """
    prices = np.array(prices)
    lst = []

    for i in range(len(prices)-period_of_time):
        a = prices[i+1:i+1+period_of_time]

        lst.append(np.any(np.where(a < prices[i]*profit, 0, 1))*1)

    lst = lst + [np.NaN for i in range(period_of_time)]
    return lst


def change_punc(df, cols_to_change):
    """
    Returns Dataframe with float data types.

    Parameters
    ----------
    df : Dataframe
        Dataframe to change data types of 'object' to float.

    cols_to_change : list(str)
        List of column names of which data types is 'object'.
    """
    for col in cols_to_change:
        if df[col].dtype == "object":
            df[col] = df[col].apply(lambda x: float(x.replace(",", "")))
    return df


def change_volume(col):
    """
    Changes 'M' to 10^6 and 'K' to 10^3.
    """
    if col[-1] == "M":
        return float(col[:-1].replace(",", "."))*1000000
    elif col[-1] == "K":
        return float(col[:-1].replace(",", "."))*1000
