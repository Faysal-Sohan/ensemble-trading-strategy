import os
import sys
root_dir = os.getcwd()
sys.path.insert(1, root_dir)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_ta as ta
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib

def add_features(df:pd.DataFrame):
    """
    this function takes a dataframe as an input and
    add different technical indicators value as features
    """   
    df['sma_5'] = ta.sma(df.close, length=5)
    df['sma_10'] = ta.sma(df.close, length=10)
    df['ema_5'] = ta.ema(df.close, length=5)
    df['ema_10'] = ta.ema(df.close, length=10)
    a = ta.macd(df.close)
    df = df.join(a)
    a = ta.adx(df.high,df.low,df.close)
    df = df.join(a)
    df['rsi_14'] = ta.rsi(df.close, length=14)
    a = ta.bbands(df.close)
    df = df.join(a)
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
        df.index = pd.to_datetime(df['date'])
        df.drop('date',axis=1, inplace=True)
    return df

def get_target_signal_rets(df: pd.DataFrame, period = 2):
    """
    this function generates trading signal based on return type
    if return is greater than 1% then we go for long position
    if return is less than -1% then we go for short position 
    """
    df['rets'] = df.close.pct_change(periods=period).mul(100)
    df['signals'] = np.where((df['rets'] > 1.5), 1, 0) #long_postion
    df['signals'] = np.where((df['rets'] < -1.5), -1, df['signals']) #short_position
    df.dropna(inplace=True)
    signals = df['signals']
    df.drop(['rets','signals'], axis=1, inplace=True)
    return signals

def get_target_signal_next_price(df: pd.DataFrame, period = -1):
    """
    this function generates trading signal based on next day price
    if next day's price is increased by 1 then we go for long/buy position
    if next day's price is decreased by 1 then we go for short/sell position
    """
    df['next_prices'] = df.close.shift(period)
    df['signals'] = np.where((df.close - df['next_prices']) < -1, 1, 0) # long/buy position
    df['signals'] = np.where((df.close - df['next_prices']) > 1, -1, df['signals']) # short/sell position
    df.dropna(inplace=True)
    signals = df['signals']
    df.drop(['next_prices', 'signals'], axis=1, inplace=True)
    return signals

def get_target_signal_prev_fwd_price(df: pd.DataFrame, period = -1):
    """
    this function generates trading signal based on future price & past price
    if current price is higher than both past and future price then it indicates to sell
    and if current price is lower than both past and future price then it indicates to buy
    """                                         
    df['prev_3d_price'] = df.close.shift(3)
    df['fwd_3d_price'] = df.close.shift(-3)

    df.dropna(inplace=True)

    df['target_signal'] = np.where((df['close'] < df['prev_3d_price']) & (df['close'] < df['fwd_3d_price']), 1, 0) # long/buy position
    df['target_signal'] = np.where((df['close'] > df['prev_3d_price']) & (df['close'] > df['fwd_3d_price']), -1, df['target_signal']) # short/sell position

    signals = df['target_signal']

    df.drop(['prev_3d_price', 'fwd_3d_price','target_signal'], axis=1, inplace=True)
    return signals

def data_preprocess(X:pd.DataFrame):
    """
    this function scaled the featured dataframe for train and test set split
    """                                
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled

def plot_cm(cm):
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.xlabel("Predicted labels")
    plt.ylabel("True labels")
    plt.title("Confusion Matrix")
    plt.show()

def train_rfc_model(X: np.array, y: np.array, signal_type='returns'):
    
    # splitting the dataset into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=0)
    
    # train with RFC model
    rf_model = RandomForestClassifier(n_estimators=100, random_state=0)
    rf_model.fit(X_train, y_train)
    
    # making predictions with test data
    predictions = rf_model.predict(X_test)
    
    # saving the model
    joblib.dump(rf_model, f'random_forrest_classifier_on_{signal_type}.pkl')
    
    # model metrics calculation
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, average='weighted')
    recall = recall_score(y_test, predictions, average='weighted')
    f1 = f1_score(y_test, predictions, average='weighted')
    cm = confusion_matrix(y_test, predictions)
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    plot_cm(cm)
    

def train_bag_model_with_lr(X: np.array, y: np.array, signal_type='returns'):
    
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
    joblib.dump(bag_model, f'bagging_logistic_regression_on_{signal_type}.pkl')
    
    # model metrics calculation
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, average='weighted')
    recall = recall_score(y_test, predictions, average='weighted')
    f1 = f1_score(y_test, predictions, average='weighted')
    cm = confusion_matrix(y_test, predictions)
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    plot_cm(cm)
                                     
def train_bag_model_with_dtc(X: np.array, y: np.array, signal_type='returns'):
    
    # splitting the dataset into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=10)
    
    # train with Bagging Classifier
    bag_model = BaggingClassifier(base_estimator=DecisionTreeClassifier()
                             ,n_estimators=100
                             ,max_samples=0.8
                             ,oob_score=True
                             ,random_state=10)
    bag_model.fit(X_train, y_train)
    
    # making predictions with test data
    predictions = bag_model.predict(X_test)
    
    # saving the model
    joblib.dump(bag_model, f'bagging_decision_tree_classifier_on_{signal_type}.pkl')
    
    # model metrics calculation
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, average='weighted')
    recall = recall_score(y_test, predictions, average='weighted')
    f1 = f1_score(y_test, predictions, average='weighted')
    cm = confusion_matrix(y_test, predictions)
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    plot_cm(cm)   

def get_indicator_signal(df: pd.DataFrame):
    """
    This function receives featured dataframe and returns a dataframe with 
    indicator trading signal
    """
    signals = pd.DataFrame()
    
    signals['sma_signal'] = np.where((df['sma_5'] > df['sma_10']), 1, 0)
    signals['sma_signal'] = np.where((df['sma_5'] < df['sma_10']), -1, signals['sma_signal'])  

    signals['ema_signal'] = np.where((df['ema_5'] > df['ema_10']), 1, 0)
    signals['ema_signal'] = np.where((df['ema_5'] < df['ema_10']), -1, signals['ema_signal'])  

    signals['macd_signal'] = np.where((df['MACD_12_26_9'] > df['MACDs_12_26_9']) & (df['MACD_12_26_9'] > 0),1,0)
    signals['macd_signal'] = np.where((df['MACD_12_26_9'] > df['MACDs_12_26_9']) & (df['MACD_12_26_9'] > 0),1,signals['macd_signal'])

    signals['rsi_signal'] = np.where((df['rsi_14'] > 70), -1, 0)
    signals['rsi_signal'] = np.where((df['rsi_14'] < 30), 1, signals['rsi_signal'])

    signals['bb_signal'] = np.where((df['BBU_5_2.0'] < df['close']), -1, 0)
    signals['bb_signal'] = np.where((df['BBL_5_2.0'] > df['close']), 1, 0)

    return signals

    
def group3_ensemble_model_signals(df: pd.DataFrame):
    # adding feutures to the data  
    X = add_features(df)

    # get indicator_signal
    signals = get_indicator_signal(X)
    
    # scaling the dataset
    X_scaled = data_preprocess(X)
    
    # loading the trained model
    loaded_model = joblib.load('../modules/bagging_logistic_regression_on_returns.pkl')

    # Using the loaded model to make predictions
    predictions = loaded_model.predict(X_scaled)
    signal = np.where((predictions == 1) & (signals['macd_signal'] == 1) & (signals['ema_signal'] == 1), 1, 0)
    signal = np.where((predictions == -1) & (signals['macd_signal'] == -1) & (signals['ema_signal'] == -1), -1, signal)
    
    return signal[-1]
    
    

