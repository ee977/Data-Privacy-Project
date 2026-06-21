import os
import numpy as np
import pandas as pd
import yfinance as yf
import warnings

warnings.filterwarnings("ignore")

def get_raw_data(ticker_symbol: str, start_date: str, end_date: str, input_dir: str) -> pd.DataFrame:
    safe_name = ticker_symbol.replace('.', '_').replace('=', '_')
    file_name = f"{safe_name}_raw.csv"
    file_path = os.path.join(input_dir, file_name)
    
    if os.path.exists(file_path):
        df = pd.read_csv(file_path, parse_dates=['Date'], index_col='Date')
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        return df
    else:
        df = yf.download(ticker_symbol, start=start_date, end=end_date)
        if df.empty:
            raise ValueError(f"No data retrieved for {ticker_symbol}")
        
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
            
        df.to_csv(file_path)
        return df

def calculate_indicators(df_raw: pd.DataFrame, prefix: str) -> pd.DataFrame:
    df = df_raw.copy()
    close_series = df['Close']
    indicator_df = pd.DataFrame(index=df.index)
    
    indicator_df[f"{prefix}Close"] = close_series
    close_smooth = close_series.ewm(span=3, adjust=False).mean()
    indicator_df[f"{prefix}Close_Smooth"] = close_smooth
    
    resistance = close_series.rolling(window=20).max()
    support = close_series.rolling(window=20).min()
    indicator_df[f"{prefix}Resistance_20D"] = resistance
    indicator_df[f"{prefix}Support_20D"] = support
    
    indicator_df[f"{prefix}Open_Rel"] = (df['Open'] - close_series) / (close_series + 1e-10)
    indicator_df[f"{prefix}High_Rel"] = (df['High'] - close_series) / (close_series + 1e-10)
    indicator_df[f"{prefix}Low_Rel"] = (df['Low'] - close_series) / (close_series + 1e-10)
    indicator_df[f"{prefix}Volume_Change"] = df['Volume'].pct_change()
    
    for n in [1, 5, 10, 15]:
        raw_return = close_series.pct_change(periods=n)
        bound = 0.10 * np.sqrt(n)
        indicator_df[f"{prefix}Return_{n}D"] = raw_return.clip(lower=-bound, upper=bound)
        
    indicator_df[f"{prefix}Target_Return"] = close_smooth.pct_change(periods=1).clip(lower=-0.10, upper=0.10)
        
    for n in [5, 10, 15]:
        ma = close_series.rolling(window=n).mean()
        indicator_df[f"{prefix}MA_Rel_{n}D"] = (ma - close_series) / (close_series + 1e-10)
        
        rolling_std = close_series.rolling(window=n).std()
        vol = rolling_std * np.sqrt(n)
        indicator_df[f"{prefix}Volatility_Rel_{n}D"] = vol / (close_series + 1e-10)
        
    delta = close_series.diff()
    for n in [5, 10, 15]:
        gains = delta.clip(lower=0)
        losses = -delta.clip(upper=0)
        avg_gain = gains.rolling(window=n).mean()
        avg_loss = losses.rolling(window=n).mean()
        rs = avg_gain / (avg_loss + 1e-10)
        indicator_df[f"{prefix}RSI_{n}D"] = 100 - (100 / (1 + rs))
        
    for n in [1, 5, 10, 15]:
        shifted_close = close_series.shift(n)
        raw_mom = ((close_series - shifted_close) / (shifted_close + 1e-10)) * 100
        bound = 10.0 * np.sqrt(n)
        indicator_df[f"{prefix}Momentum_{n}D"] = raw_mom.clip(lower=-bound, upper=bound)
        
    bb_middle = close_series.rolling(window=20).mean()
    bb_std = close_series.rolling(window=20).std()
    indicator_df[f"{prefix}BB_Upper_Rel"] = ((bb_middle + 2.0 * bb_std) - close_series) / (close_series + 1e-10)
    indicator_df[f"{prefix}BB_Lower_Rel"] = ((bb_middle - 2.0 * bb_std) - close_series) / (close_series + 1e-10)
    
    ema_12 = close_series.ewm(span=12, adjust=False).mean()
    ema_26 = close_series.ewm(span=26, adjust=False).mean()
    macd_line = ema_12 - ema_26
    indicator_df[f"{prefix}MACD_Rel"] = macd_line / (close_series + 1e-10)
        
    return indicator_df

def generate_inputs(target_ticker: str, input_dir: str, start_date: str = "2010-01-04", end_date: str = "2026-05-24") -> pd.DataFrame:
    target_raw = get_raw_data(target_ticker, start_date, end_date, input_dir)
    bist_raw = get_raw_data("XU100.IS", start_date, end_date, input_dir)
    usdtry_raw = get_raw_data("USDTRY=X", start_date, end_date, input_dir)
    
    target_features = calculate_indicators(target_raw, "TARGET_")
    bist_features = calculate_indicators(bist_raw, "BIST_")
    
    usdtry_features = pd.DataFrame(index=usdtry_raw.index)
    usdtry_features["USDTRY_Close"] = usdtry_raw["Close"] 
    usdtry_features["USDTRY_Return_1D"] = usdtry_raw["Close"].pct_change().clip(lower=-0.10, upper=0.10)
    usdtry_features["USDTRY_High_Rel"] = (usdtry_raw["High"] - usdtry_raw["Close"]) / (usdtry_raw["Close"] + 1e-10)
    
    consolidated_df = pd.concat([target_features, bist_features, usdtry_features], axis=1, join='inner')
    
    consolidated_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    consolidated_df.ffill(inplace=True)
    consolidated_df.bfill(inplace=True)
    consolidated_df.dropna(inplace=True)
    consolidated_df = consolidated_df.clip(lower=-1e4, upper=1e4)
    
    safe_name = target_ticker.replace('.', '_')
    consolidated_df.to_csv(os.path.join(input_dir, f"{safe_name}_features.csv"))
    
    return consolidated_df