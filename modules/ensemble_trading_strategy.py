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
    df['sma_16'] = ta.sma(df.close, length=16)
    df['ema_16'] = ta.ema(df.close, length=16)
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
    df.index = pd.to_datetime(df['date'])
    return df.drop('date',axis=1)

def get_target_signal(close: pd.Series, signal_type = 'return', period = 2):
    signals = pd.Series()
    rets = close.pct_change(periods=2).mul(100)
    signals = np.where((rets > 2), -1, 0)
    signals = np.where((rets < -2), 1, signals)
    return signals

def data_preprocess(df:pd.DataFrame):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df)
    return X_scaled

def train_rfc_model(df:pd.DataFrame):
    # adding feutures to the data  
    X = add_feauters(df)
    y = get_target_signal(df.close)
    
    # scaling the dataset
    X_scaled = data_preprocess(X)
    
    # splitting the dataset into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, stratify=y, random_state=10)
    
    # train with RFC model
    rf_model = RandomForestClassifier(n_estimators=100, random_state=10)
    rf_model.fit(X_train, y_train)
    
    # making predictions with test data
    predictions = rf_model.predict(X_test)
    
    # saving the model
    joblib.dump(rf_model, 'random_forrest_classifier_on_return_signal.pkl')
    
    # accuracy calculation
    accuracy = accuracy_score(y_test, predictions)
    print(f'Accuracy of RFC model: {accuracy}')
    

def train_bag_model_with_lr(df:pd.DataFrame):
    # adding feutures to the data  
    X = add_feauters(df)
    y = get_target_signal(df.close)
    
    # scaling the dataset
    X_scaled = data_preprocess(X)
    
    # splitting the dataset into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, stratify=y, random_state=10)
    
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
    joblib.dump(rf_model, 'bagging_logistic_regression.pkl')
    
    # accuracy calculation
    accuracy = accuracy_score(y_test, predictions)
    print(f'Accuracy of Bagging with LR: {accuracy}')
    
def group3_ensemble_model_signals(df: pd.DataFrame):
    # adding feutures to the data  
    X = add_feauters(df)
    
    # scaling the dataset
    X_scaled = data_preprocess(X)
    
    # loading the trained model
    loaded_model = joblib.load('bagging_logistic_regression.pkl')

    # Now you can use the loaded model to make predictions
    predictions = loaded_model.predict(X_scaled)
    
    return predictions[-1]
    
    

