import pandas as pd 
import numpy as np

def EMA (df: pd.DataFrame, short_period = 5, long_period = 10):
    short_ema = df['close'].ewm(span=short_period, adjust=False).mean() 
    long_ema = df['close'].ewm(span=long_period, adjust=False).mean() 
    df['short_ema'] = short_ema
    df['long_ema'] = long_ema

    df['Signal'] = np.where(long_ema<short_ema,1,0)
    df['Signal'] = np.where(long_ema>short_ema,-1,df['Signal'])
    return df

def rsi_signals(df:pd.DataFrame, window=14):
    gains = (df['close'] - df['open']).apply(lambda x: x if x > 0 else 0)
    loss = (df['close'] - df['open']).apply(lambda x: -x if x < 0 else 0)

    ema_gains = gains.ewm(span=window, min_periods=window).mean()
    ema_losses = loss.ewm(span=window, min_periods=window).mean()

    rs = ema_gains / ema_losses
    df['rsi_14'] = 100 - (100 / (rs + 1))
    df['rsi_signal'] = 0
    df.loc[df['rsi_14'] >= 70, 'rsi_signal'] = -1
    df.loc[df['rsi_14'] <= 30, 'rsi_signal'] = 1
    return df

def macd_signals(df, fast_period=12, slow_period=26, signal_period=9):
    ema_fast = df['close'].ewm(span=fast_period, adjust=False).mean()
    ema_slow = df['close'].ewm(span=slow_period, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal = macd.ewm(span=signal_period, adjust=False).mean()
    # Generate trading signals based on the MACD and signal line
    df['MACD_signal'] = 0
    df['MACD_signal'] = np.where((macd > signal) & (macd > 0),1, 0)
    df['MACD_signal'] = np.where((macd < signal) & (macd < 0), -1, df['MACD_signal'])
    return df

def bb_signals(df:pd.DataFrame, n = 20, m = 2):
    sma = df['close'].rolling(n).mean()
    rolling_std = df['close'].rolling(n).std()
    df['bb_ub'] = sma + m * rolling_std
    df['bb_lb'] = sma - m * rolling_std
    df['bb_signal'] = 0
    df.loc[df['close'] >= df['bb_ub'],'bb_signal'] = -1
    df.loc[df['close'] <= df['bb_lb'],'bb_signal'] = 1
    return df

def keltner_channels(df: pd.DataFrame, n: int = 10, m: int = 1) -> pd.DataFrame:
    middle_line = df['close'].ewm(span=n).mean()
    tr1 = df['high'] - df['low']
    tr2 = abs(df['high'] - df['close'].shift())
    tr3 = abs(df['low'] - df['close'].shift())
    true_range = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
    upper_band = middle_line + m * true_range.ewm(span=n).mean()
    lower_band = middle_line - m * true_range.ewm(span=n).mean()
    # Generate signals
    df['Keltner_channels'] = 0
    df.loc[(df['close'] > upper_band), 'Keltner_channels'] = -1  # Sell signal
    df.loc[(df['close'] < lower_band), 'Keltner_channels'] = 1  # Buy signal
    return df
    
def stochastic_oscillator_strategy(df:pd.DataFrame, period=14, oversold=20, overbought=80):
    highs = df['high']
    lows = df['low']
    closes = df['close']
    stochastic_oscillator = 100 * ((closes - lows.rolling(period).min()) / (highs.rolling(period).max() - lows.rolling(period).min()))
    df['Stochastic_oscillator'] = 0
    df['Stochastic_oscillator'] = np.where(stochastic_oscillator < oversold, 1, 0)
    df['Stochastic_oscillator'] = np.where(stochastic_oscillator > overbought, -1, df['Stochastic_oscillator'])
    return df
    