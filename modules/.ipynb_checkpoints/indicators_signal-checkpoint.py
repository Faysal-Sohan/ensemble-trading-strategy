import pandas as pd 
import numpy as np 

def calc_rsi(df:pd.DataFrame, window=14):
    gains = (df['close'] - df['open']).apply(lambda x: x if x > 0 else 0)
    loss = (df['close'] - df['open']).apply(lambda x: -x if x < 0 else 0)

    ema_gains = gains.ewm(span=window, min_periods=window).mean()
    ema_losses = loss.ewm(span=window, min_periods=window).mean()

    rs = ema_gains / ema_losses
    df['rsi_14'] = 100 - (100 / (rs + 1))
    return df