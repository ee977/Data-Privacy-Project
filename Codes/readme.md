# Federated Dual-Head MLP-Transformer-LSTM Stock Predictor

This repository implements a state-of-the-art hybrid deep learning pipeline for multi-sector stock price and direction forecasting under privacy-preserving Federated Learning constraints.

## Overview
Financial time-series data suffers from an extremely low signal-to-noise ratio and non-stationary dynamics. Jointly predicting both the future return magnitude (regression) and directional trend (classification) often causes traditional single-head models to suffer from mean-seeking mode collapse (predicting a flatline 0 return). 

This pipeline resolves these issues by introducing:
- **Dual-Head Optimization:** Decouples the prediction layer into a Regression Head (optimizing Mean Squared Error) and a Classification Head (optimizing Binary Cross Entropy with Logits).
- **FedProx Optimization:** Prevents local client weight divergence over heterogeneous, non-IID sector stock data by implementing proximal regularization ($\mu = 0.01$).
- **Deep Hierarchical Features:** Employs step-by-step TimeStep MLP feature compression, sequence-wise LSTMs, and self-attention Transformer blocks to distill highly predictive features.

-------------------------------------------------------------------------------------------------------

## Directory Structure
- `input_generator.py`: Connects to Yahoo Finance, downloads stock records, and calculates 73 scale-invariant, stationary indicators.
- `model.py`: Outlines the core dual-head `MLPLSTMTransformer` network and the `DualHeadLoss` function.
- `other_models.py`: Outlines baseline configurations (SVR, Simple LSTM, MLP-LSTM, Transformer-LSTM) and the training wrappers.
- `test.py`: Reconstructs returns to raw price levels and computes evaluation metrics (MDA, F1, RMSE, MAPE, R2).
- `main.py`: Coordinates dataset pipelines, training regimes, and evaluates results.
- `figure_generator.py`: Generates comparative input visualizations.

-------------------------------------------------------------------------------------------------------

## Feature Engineering and Dataset
Data pipelines dynamically start from **January 4, 2010**, aligning five target stocks (`SISE.IS`, `KCHOL.IS`, `MGROS.IS`, `THYAO.IS`, `TSKB.IS`) with `XU100.IS` (BIST100) and `USDTRY=X` (USD/TRY exchange rate). The resulting 73 indicators are structured as sliding windows of shape `[Batch, 10, 73]`.

-------------------------------------------------------------------------------------------------------

## Installation & Running the Pipeline
To install dependencies and start the model training/evaluation loop:

```bash
pip install numpy pandas torch yfinance scikit-learn matplotlib joblib
python main.py
