## Ensemble Trading Strategy

This repository contains an ensemble trading strategy for 50 assets of the S&P 500 index.

### How to Use

1. **Copy Files**: Copy all the files from the [`modules`](https://github.com/Faysal-Sohan/ensemble-trading-strategy/tree/main/modules) folder of this repository and paste them into your local project's module folder.

2. **Import Module**: To use the `ensemble_trading_strategy.py` module, import it into your code or notebook as demonstrated in the provided demo notebook.
   ```python
   import sys
   import os
   root_dir = os.getcwd()
   sys.path.insert(1, root_dir)
   sys.path.append('../modules/')
   import ensemble_trading_strategy as ets
   ```
3. **Getting Signal**: To get the trading signal, run the below code
   ```python
     signal = ets.group3_ensemble_model_signals(df)
     # here df is a dataframe that contains columns [date, open, high, low, close, volumes]
     # df must contains minimum 35 rows of historical data
   ```

### Demo Notebook

Check out the demo notebook [here](https://github.com/Faysal-Sohan/ensemble-trading-strategy/blob/main/example/demo.ipynb) to see how to use the ensemble trading strategy.

### Dependencies

To calculate technical indicator values, this strategy utilizes the `pandas-ta` library. Install this library by executing the following command in your Python environment:

```bash
$ pip install pandas-ta
```

## S & P 50 Assets

### Industrials
- "MMM"
- "AOS"
- "BA"
- "AXON"
- "CAT"

### Health Care
- "ABT"
- "BAX"
- "BDX"
- "TECH"
- "ALGN"

### Information Technology
- "ADBE"
- "AMD"
- "AAPL"
- "CDNS"
- "NVDA"

### Financials
- "AFL"
- "BAC"
- "GS"
- "BX"
- "COF"

### Materials
- "FMC"
- "IFF"
- "KLAC"
- "APD"
- "CE"

### Consumer Staples
- "BG"
- "MO"
- "CPB"
- "STZ"
- "WMT"

### Energy
- "TRGP"
- "VLO"
- "WMB"
- "APA"
- "BKR"

### Communication Services
- "DIS"
- "WBD"
- "GOOGLE"
- "CHTR"
- "EA"

### Utilities
- "AES"
- "LNT"
- "AEP"
- "AWK"
- "CMS"

### Real Estate
- "ARE"
- "BXP"
- "CPT"
- "AMT"
- "CCI"

## Backtesting Notebook
- Check out the backtest notebook [here](https://github.com/Faysal-Sohan/ensemble-trading-strategy/blob/main/Notebooks/backtest.ipynb)
