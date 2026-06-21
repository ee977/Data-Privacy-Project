import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, f1_score

def plot_test_predictions(stock_name, model_name, dates, actual, predicted, metrics, figures_dir, is_30_days=False):
    results_dir = os.path.join(figures_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)
    clean_model_name = model_name.split('_', 1)[-1] if '_' in model_name else model_name

    if is_30_days:
        dates = dates[-30:]
        actual = actual[-30:]
        predicted = predicted[-30:]
        plot_name = f"{clean_model_name}_{stock_name}_LAST_30_DAYS.png"
        title = f"{clean_model_name}: {stock_name} Last 30 Days Forecast"
    else:
        plot_name = f"{clean_model_name}_{stock_name}_prediction.png"
        title = f"{clean_model_name}: {stock_name} Next-Day Prediction (Test Set: 2025-2026)"

    plot_path = os.path.join(results_dir, plot_name)
    plt.figure(figsize=(15, 8))
    plt.plot(dates, actual, label=f"Actual {stock_name} Price", color='blue', alpha=0.85, linewidth=1.5)
    plt.plot(dates, predicted, label=f"Predicted {stock_name} Price", color='orange', alpha=0.85, linewidth=1.5, linestyle='--')

    if not is_30_days and len(dates) > 250:
        highlight_start_date = dates[-250]
        plt.axvspan(highlight_start_date, dates[-1], color='gray', alpha=0.1, label='Focus: Last Year')

    textstr = '\n'.join((
        f"RMSE: {metrics['RMSE']:.4f}",
        f"Accuracy (MDA): {metrics['MDA']:.2f}%",
        f"Directional F1: {metrics['F1']:.2f}%",
        f"MAPE: {metrics['MAPE']:.4f}%",
        f"R2 Score: {metrics['R2']:.4f}"
    ))
    props = dict(boxstyle='round', facecolor='white', alpha=0.85, edgecolor='gray')
    plt.gca().text(0.02, 0.88, textstr, transform=plt.gca().transAxes, fontsize=12, verticalalignment='top', bbox=props)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel("Date", fontsize=11)
    plt.ylabel(f"{stock_name} Price", fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()

def _get_predictions(model, batch_x, device):
    is_svm = hasattr(model, 'model')
    if is_svm:
        preds = model(batch_x).numpy().flatten()
        return preds, preds

    batch_x = batch_x.to(device)
    with torch.no_grad():
        reg_pred, cls_logit = model(batch_x)
    return reg_pred.cpu().numpy().flatten(), cls_logit.cpu().numpy().flatten()

def evaluate_model(model, test_loader, scaler_y, test_df, seq_length, device):
    if hasattr(model, 'to') and not hasattr(model, 'model'):
        model.to(device)
    if hasattr(model, 'eval') and not hasattr(model, 'model'):
        model.eval()

    all_reg_predictions = []
    all_cls_predictions = []
    all_actuals = []

    for batch_x, batch_y in test_loader:
        reg_preds, cls_logits = _get_predictions(model, batch_x, device)
        all_reg_predictions.extend(reg_preds)
        all_cls_predictions.extend(cls_logits)
        all_actuals.extend(batch_y.numpy().flatten())

    all_reg_arr = np.array(all_reg_predictions).reshape(-1, 1)
    all_cls_arr = np.array(all_cls_predictions).reshape(-1, 1)
    all_actuals_arr = np.array(all_actuals).reshape(-1, 1)

    actual_returns_unscaled = scaler_y.inverse_transform(all_actuals_arr).flatten()
    pred_returns_unscaled = scaler_y.inverse_transform(all_reg_arr).flatten()
    actual_dir = (actual_returns_unscaled > 0).astype(int)
    
    if hasattr(model, 'model'):
        pred_dir = (pred_returns_unscaled > 0).astype(int)
    else:
        pred_dir = (all_cls_arr.flatten() > 0).astype(int)
        
    mda = np.mean(actual_dir == pred_dir) * 100
    f1 = f1_score(actual_dir, pred_dir, average='binary', zero_division=0) * 100
    close_smooth_prev = test_df["TARGET_Close_Smooth"].values[seq_length - 1: len(test_df) - 1]
    actual_prices = close_smooth_prev * (1.0 + actual_returns_unscaled)
    pred_prices = close_smooth_prev * (1.0 + pred_returns_unscaled)

    if np.isnan(pred_prices).any() or np.isnan(actual_prices).any():
        actual_prices = np.nan_to_num(actual_prices, nan=0.0)
        pred_prices = np.nan_to_num(pred_prices, nan=0.0)

    mse = mean_squared_error(actual_prices, pred_prices)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actual_prices, pred_prices)
    mape = np.mean(np.abs((actual_prices - pred_prices) / actual_prices)) * 100
    r2 = r2_score(actual_prices, pred_prices)

    metrics = {"MDA": mda, "F1": f1, "RMSE": rmse, "MAE": mae, "MAPE": mape, "MSE": mse, "R2": r2}
    return metrics, actual_prices, pred_prices

def evaluate_all(models_dict, test_loaders_dict, scalers_y_dict, test_dfs_dict, seq_length, device, figures_dir):
    results = []
    for model_name, model in models_dict.items():
        for stock_name, loader in test_loaders_dict.items():
            scaler_y = scalers_y_dict[stock_name]
            test_df = test_dfs_dict[stock_name]
            metrics, actual_prices, pred_prices = evaluate_model(model, loader, scaler_y, test_df, seq_length, device)
            dates = test_df.index[seq_length:]

            plot_test_predictions(stock_name, model_name, dates, actual_prices, pred_prices, metrics, figures_dir, is_30_days=False)
            plot_test_predictions(stock_name, model_name, dates, actual_prices, pred_prices, metrics, figures_dir, is_30_days=True)

            results.append({
                "Model": model_name,
                "Stock": stock_name,
                "MDA (%)": round(metrics["MDA"], 2),
                "F1 (%)": round(metrics["F1"], 2),
                "RMSE": round(metrics["RMSE"], 4),
                "MAPE (%)": round(metrics["MAPE"], 4),
                "R2": round(metrics["R2"], 4)
            })

    results_df = pd.DataFrame(results)
    mda_pivot = results_df.pivot(index='Model', columns='Stock', values='MDA (%)')
    f1_pivot = results_df.pivot(index='Model', columns='Stock', values='F1 (%)')
    rmse_pivot = results_df.pivot(index='Model', columns='Stock', values='RMSE')
    mape_pivot = results_df.pivot(index='Model', columns='Stock', values='MAPE (%)')
    r2_pivot = results_df.pivot(index='Model', columns='Stock', values='R2')
    
    print(mda_pivot.to_string())
    print(f1_pivot.to_string())
    print(rmse_pivot.to_string())
    print(mape_pivot.to_string())
    print(r2_pivot.to_string())