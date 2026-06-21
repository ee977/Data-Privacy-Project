import os
import pandas as pd
import matplotlib.pyplot as plt

def generate_plots(df: pd.DataFrame, figures_dir: str):
    inputs_dir = os.path.join(figures_dir, 'inputs')
    os.makedirs(inputs_dir, exist_ok=True)
    
    raw_types = ['Open', 'High', 'Low', 'Close', 'Volume']
    for val_type in raw_types:
        fig_name = f"{val_type}_values.png"
        fig_path = os.path.join(inputs_dir, fig_name)
        if os.path.exists(fig_path): continue
            
        fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(14, 10), sharex=True)
        axes[0].plot(df.index, df[f"TARGET_{val_type}"], color='blue', label=f"TARGET {val_type}")
        axes[0].legend(loc="upper left")
        axes[0].grid(True, linestyle='--', alpha=0.5)
        
        axes[1].plot(df.index, df[f"BIST_{val_type}"], color='red', label=f"BIST100 {val_type}")
        axes[1].legend(loc="upper left")
        
        axes[2].plot(df.index, df[f"USDTRY_{val_type}"], color='green', label=f"USDTRY {val_type}")
        axes[2].legend(loc="upper left")
        
        plt.tight_layout()
        plt.savefig(fig_path, dpi=150)
        plt.close()

def generate_plots(df: pd.DataFrame, figures_dir: str):
    inputs_dir = os.path.join(figures_dir, 'inputs')
    os.makedirs(inputs_dir, exist_ok=True)
    
    raw_types = ['Open', 'High', 'Low', 'Close', 'Volume']
    for val_type in raw_types:
        fig_name = f"{val_type}_values.png"
        fig_path = os.path.join(inputs_dir, fig_name)
        
        if os.path.exists(fig_path):
            continue
            
        fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(14, 10), sharex=True)
        
        axes[0].plot(df.index, df[f"SISE_{val_type}"], color='blue', label=f"SISE {val_type}")
        axes[0].set_ylabel("SISE Price (TRY)")
        axes[0].legend(loc="upper left")
        axes[0].grid(True, linestyle='--', alpha=0.5)
        axes[0].set_title(f"SISE - BIST100 - USDTRY {val_type} Values Over Time", fontsize=12, fontweight='bold')
        
        axes[1].plot(df.index, df[f"BIST_{val_type}"], color='red', label=f"BIST100 {val_type}")
        axes[1].set_ylabel("BIST100 Index")
        axes[1].legend(loc="upper left")
        axes[1].grid(True, linestyle='--', alpha=0.5)
        
        axes[2].plot(df.index, df[f"USDTRY_{val_type}"], color='green', label=f"USDTRY {val_type}")
        axes[2].set_ylabel("USD/TRY Exchange Rate")
        axes[2].legend(loc="upper left")
        axes[2].grid(True, linestyle='--', alpha=0.5)
        
        plt.xlabel("Date")
        plt.tight_layout()
        plt.savefig(fig_path, dpi=150)
        plt.close()
        print(f"file created as {fig_name}")
        
    returns_fig = "Returns_comparison.png"
    returns_path = os.path.join(inputs_dir, returns_fig)
    if not os.path.exists(returns_path):
        plt.figure(figsize=(14, 8))
        for d in [1, 5, 10, 15]:
            plt.plot(df.index, df[f"SISE_Return_{d}D"], label=f"SISE Return {d}D", alpha=0.7)
        for d in [1, 5, 10, 15]:
            plt.plot(df.index, df[f"BIST_Return_{d}D"], label=f"BIST Return {d}D", alpha=0.7, linestyle='--')
            
        plt.title("SISE & BIST100 Returns Comparison (1D, 5D, 10D, 15D)", fontsize=14, fontweight='bold')
        plt.xlabel("Date")
        plt.ylabel("Return (Percentage Change)")
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
        plt.tight_layout()
        plt.savefig(returns_path, dpi=150)
        plt.close()
        print(f"file created as {returns_fig}")

    ma_fig = "MA_comparison.png"
    ma_path = os.path.join(inputs_dir, ma_fig)
    if not os.path.exists(ma_path):
        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(14, 10), sharex=True)
        axes[0].plot(df.index, df["SISE_Close"], label="SISE Close Price", color='black', alpha=0.5)
        for d in [5, 10, 15]:
            axes[0].plot(df.index, df[f"SISE_MA_{d}D"], label=f"SISE MA {d}D")
        axes[0].set_ylabel("SISE Price")
        axes[0].legend(loc="upper left")
        axes[0].grid(True, linestyle='--', alpha=0.5)
        axes[0].set_title("Moving Average (MA) Comparison", fontsize=14, fontweight='bold')
        
        axes[1].plot(df.index, df["BIST_Close"], label="BIST Close Index", color='black', alpha=0.5)
        for d in [5, 10, 15]:
            axes[1].plot(df.index, df[f"BIST_MA_{d}D"], label=f"BIST MA {d}D", linestyle='--')
        axes[1].set_ylabel("BIST Index")
        axes[1].legend(loc="upper left")
        axes[1].grid(True, linestyle='--', alpha=0.5)
        
        plt.xlabel("Date")
        plt.tight_layout()
        plt.savefig(ma_path, dpi=150)
        plt.close()
        print(f"file created as {ma_fig}")

    vol_fig = "Volatility_comparison.png"
    vol_path = os.path.join(inputs_dir, vol_fig)
    if not os.path.exists(vol_path):
        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(14, 10), sharex=True)
        for d in [5, 10, 15]:
            axes[0].plot(df.index, df[f"SISE_Volatility_{d}D"], label=f"SISE Volatility {d}D")
        axes[0].set_ylabel("SISE Volatility")
        axes[0].legend(loc="upper left")
        axes[0].grid(True, linestyle='--', alpha=0.5)
        axes[0].set_title("Historical Volatility Comparison (5D, 10D, 15D)", fontsize=14, fontweight='bold')
        
        for d in [5, 10, 15]:
            axes[1].plot(df.index, df[f"BIST_Volatility_{d}D"], label=f"BIST Volatility {d}D", linestyle='--')
        axes[1].set_ylabel("BIST Volatility")
        axes[1].legend(loc="upper left")
        axes[1].grid(True, linestyle='--', alpha=0.5)
        
        plt.xlabel("Date")
        plt.tight_layout()
        plt.savefig(vol_path, dpi=150)
        plt.close()
        print(f"file created as {vol_fig}")

    rsi_fig = "RSI_comparison.png"
    rsi_path = os.path.join(inputs_dir, rsi_fig)
    if not os.path.exists(rsi_path):
        plt.figure(figsize=(14, 8))
        for d in [5, 10, 15]:
            plt.plot(df.index, df[f"SISE_RSI_{d}D"], label=f"SISE RSI {d}D")
        for d in [5, 10, 15]:
            plt.plot(df.index, df[f"BIST_RSI_{d}D"], label=f"BIST RSI {d}D", linestyle='--')
            
        plt.title("RSI Comparison (5D, 10D, 15D)", fontsize=14, fontweight='bold')
        plt.axhline(70, color='red', linestyle=':', alpha=0.5, label='Overbought (70)')
        plt.axhline(30, color='blue', linestyle=':', alpha=0.5, label='Oversold (30)')
        plt.xlabel("Date")
        plt.ylabel("RSI Value (0 - 100)")
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
        plt.tight_layout()
        plt.savefig(rsi_path, dpi=150)
        plt.close()
        print(f"file created as {rsi_fig}")

    mom_fig = "Momentum_comparison.png"
    mom_path = os.path.join(inputs_dir, mom_fig)
    if not os.path.exists(mom_path):
        plt.figure(figsize=(14, 8))
        for d in [1, 5, 10, 15]:
            plt.plot(df.index, df[f"SISE_Momentum_{d}D"], label=f"SISE Momentum {d}D", alpha=0.7)
        for d in [1, 5, 10, 15]:
            plt.plot(df.index, df[f"BIST_Momentum_{d}D"], label=f"BIST Momentum {d}D", alpha=0.7, linestyle='--')
            
        plt.title("Momentum Comparison (1D, 5D, 10D, 15D)", fontsize=14, fontweight='bold')
        plt.xlabel("Date")
        plt.ylabel("Momentum Value (Relative %)")
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
        plt.tight_layout()
        plt.savefig(mom_path, dpi=150)
        plt.close()
        print(f"file created as {mom_fig}")

    # 7. Bollinger Bands visualizer
    bb_fig = "BB_comparison.png"
    bb_path = os.path.join(inputs_dir, bb_fig)
    if not os.path.exists(bb_path):
        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(14, 10), sharex=True)
        axes[0].plot(df.index, df["SISE_Close"], color='black', label="SISE Close")
        axes[0].plot(df.index, df["SISE_BB_Upper"], color='red', linestyle='--', label="BB Upper")
        axes[0].plot(df.index, df["SISE_BB_Lower"], color='green', linestyle='--', label="BB Lower")
        axes[0].set_ylabel("Price")
        axes[0].legend(loc="upper left")
        axes[0].grid(True, linestyle='--', alpha=0.5)
        axes[0].set_title("Bollinger Bands Boundary Exploration", fontsize=14, fontweight='bold')
        
        axes[1].plot(df.index, df["BIST_Close"], color='black', label="BIST Close")
        axes[1].plot(df.index, df["BIST_BB_Upper"], color='red', linestyle='--', label="BB Upper")
        axes[1].plot(df.index, df["BIST_BB_Lower"], color='green', linestyle='--', label="BB Lower")
        axes[1].set_ylabel("Index Value")
        axes[1].legend(loc="upper left")
        axes[1].grid(True, linestyle='--', alpha=0.5)
        
        plt.xlabel("Date")
        plt.tight_layout()
        plt.savefig(bb_path, dpi=150)
        plt.close()
        print(f"file created as {bb_fig}")

    # 8. MACD Indicator Visualizer
    macd_fig = "MACD_comparison.png"
    macd_path = os.path.join(inputs_dir, macd_fig)
    if not os.path.exists(macd_path):
        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(14, 10), sharex=True)
        axes[0].plot(df.index, df["SISE_MACD"], color='blue', label="MACD Line")
        axes[0].plot(df.index, df["SISE_MACD_Signal"], color='red', label="Signal Line")
        axes[0].bar(df.index, df["SISE_MACD_Hist"], color='gray', alpha=0.5, label="Histogram")
        axes[0].set_ylabel("Value")
        axes[0].legend(loc="upper left")
        axes[0].grid(True, linestyle='--', alpha=0.5)
        axes[0].set_title("MACD Momentum Oscillation", fontsize=14, fontweight='bold')
        
        axes[1].plot(df.index, df["BIST_MACD"], color='blue', label="MACD Line")
        axes[1].plot(df.index, df["BIST_MACD_Signal"], color='red', label="Signal Line")
        axes[1].bar(df.index, df["BIST_MACD_Hist"], color='gray', alpha=0.5, label="Histogram")
        axes[1].set_ylabel("Value")
        axes[1].legend(loc="upper left")
        axes[1].grid(True, linestyle='--', alpha=0.5)
        
        plt.xlabel("Date")
        plt.tight_layout()
        plt.savefig(macd_path, dpi=150)
        plt.close()
        print(f"file created as {macd_fig}")

    sr_fig = "SR_comparison.png"
    sr_path = os.path.join(inputs_dir, sr_fig)
    if not os.path.exists(sr_path):
        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(14, 10), sharex=True)
        axes[0].plot(df.index, df["SISE_Close"], color='black', label="SISE Close")
        axes[0].plot(df.index, df["SISE_Resistance_20D"], color='red', linestyle=':', label="Resistance")
        axes[0].plot(df.index, df["SISE_Support_20D"], color='green', linestyle=':', label="Support")
        axes[0].set_ylabel("Price")
        axes[0].legend(loc="upper left")
        axes[0].grid(True, linestyle='--', alpha=0.5)
        axes[0].set_title("Support & Resistance Bounds", fontsize=14, fontweight='bold')
        
        axes[1].plot(df.index, df["BIST_Close"], color='black', label="BIST Close")
        axes[1].plot(df.index, df["BIST_Resistance_20D"], color='red', linestyle=':', label="Resistance")
        axes[1].plot(df.index, df["BIST_Support_20D"], color='green', linestyle=':', label="Support")
        axes[1].set_ylabel("Index Value")
        axes[1].legend(loc="upper left")
        axes[1].grid(True, linestyle='--', alpha=0.5)
        
        plt.xlabel("Date")
        plt.tight_layout()
        plt.savefig(sr_path, dpi=150)
        plt.close()
        print(f"file created as {sr_fig}")