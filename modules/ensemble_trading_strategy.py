import numpy as np
import pandas as pd
import indicators as ind
import matplotlib.pyplot as plt
import pandas_ta as ta

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

def add_features(df:pd.DataFrame):
    df['sma_20'] = ta.sma(df.close, length=20)
    df['sma_90'] = ta.sma(df.close, length=90)
    df['ema_20'] = ta.ema(df.close, length=20)
    df['ema_90'] = ta.ema(df.close, length=90)
    a = ta.macd(df.close)
    df.join(a)
    a = ta.adx(df.high,df.low,df.close)
    df.join(a)
    df['rsi_16'] = ta.rsi(df.close, length=16)
    a = ta.bbands(df.close)
    df.join(a)
    df["cci_16"] = ta.cci(df.high, df.low, df.close, length=16)
    df["atr"] = ta.atr(df.high, df.low, df.close, length=16)
    a = ta.stoch(df.high, df.low, df.close)
    df = df.join(a)
    a = ta.stochrsi(df.close, length=16)
    df = df.join(a)
    df["wpr"] = ta.willr(df.high, df.low, df.close, length=16)
    df.dropna(inplace=True)
    # df.index = pd.to_datetime(df['date'])
    if 'date' in df.columns:
        df.drop('date',axis=1, inplace=True)
    return df

def get_target_signal_rets(close: pd.Series, period = 2):
    signals = pd.Series()
    rets = close.pct_change(periods=2).mul(100)
    signals = np.where((rets > 2), 1, 0)
    signals = np.where((rets < -2), -1, signals)
    return signals

def get_target_signal_next_price(df: pd.DataFrame, period = -1):
    df['next_prices'] = df.close.shift(period)
    df['signals'] = np.where((df.close - df['next_prices']) < -1, 1, 0)
    df['signals'] = np.where((df.close - df['next_prices']) > 1, -1, df['signals'])
    df.dropna(inplace=True)
    signals = df['signals']
    df.drop(['next_prices', 'signals'], axis=1, inplace=True)
    return signals

def get_target_signal_prev_fwd_price(df: pd.DataFrame, period = -1):
    df['prev_3d_price'] = df.close.shift(3)
    df['fwd_3d_price'] = df.close.shift(-3)

    df.dropna(inplace=True)

    df['target_signal'] = np.where((df['close'] < df['prev_3d_price']) & (df['close'] < df['fwd_3d_price']), 1, 0)
    df['target_signal'] = np.where((df['close'] > df['prev_3d_price']) & (df['close'] > df['fwd_3d_price']), -1, df['target_signal'])

    signals = df['target_signal']

    df.drop(['prev_3d_price', 'fwd_3d_price','target_signal'], axis=1, inplace=True)
    return signals

def data_preprocess(X:pd.DataFrame):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled

def train_rfc_model(X: np.array, y: np.array):
    
    # splitting the dataset into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=0)
    
    # train with RFC model
    rf_model = RandomForestClassifier(n_estimators=100, random_state=0)
    rf_model.fit(X_train, y_train)
    
    # making predictions with test data
    predictions = rf_model.predict(X_test)
    
    # saving the model
    joblib.dump(rf_model, 'random_forrest_classifier_on_return_signal.pkl')
    
    # accuracy calculation
    accuracy = accuracy_score(y_test, predictions)
    print(f'Accuracy of RFC model: {accuracy}')
    

def train_bag_model_with_lr(X: np.array, y: np.array):
    
    # splitting the dataset into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=10)
    
    # train with Bagging Classifier
    bag_model = BaggingClassifier(base_estimator=LogisticRegression()
                             ,n_estimators=100
                             ,max_samples=0.8
                             ,oob_score=True
                             ,random_state=10)
    bag_model.fit(X_train, y_train)
    
    # making predictions with test data
    predictions = bag_model.predict(X_test)
    
    # saving the model
    joblib.dump(bag_model, 'bagging_logistic_regression_next_price.pkl')
    
    # accuracy calculation
    accuracy = accuracy_score(y_test, predictions)
    print(f'Accuracy of Bagging with LR: {accuracy}')
    
def group3_ensemble_model_signals(df: pd.DataFrame):
    # adding feutures to the data  
    X = add_features(df)
    
    # scaling the dataset
    X_scaled = data_preprocess(X)
    
    # loading the trained model
    loaded_model = joblib.load('/home/sohan/Desktop/Final_Assesment/modules/bagging_logistic_regression.pkl')

    # Now you can use the loaded model to make predictions
    predictions = loaded_model.predict(X_scaled)
    
    return predictions[-1]
    
    

