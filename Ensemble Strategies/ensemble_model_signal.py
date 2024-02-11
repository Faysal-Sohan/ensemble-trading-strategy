import pandas as pd
import indicators as ind
def group3_ensemble_model_signals(df:pd.DataFrame):
    df = ind.EMA(df)
    df = ind.macd_signals(df)
    df = ind.rsi_signals(df)
    df = ind.bb_signals(df)
    df = ind.keltner_channels(df)
    df = ind.stochastic_oscillator_strategy(df)

    