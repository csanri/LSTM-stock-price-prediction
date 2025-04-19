# Stock Price Prediction with LSTM, Convolutional Layers, and Attention Mechanism

This repository contains an advanced deep learning model for stock price prediction, combining **LSTM networks**, **1D convolutional layers**, and an **attention mechanism**. Designed for time series forecasting, the model leverages technical indicators, hyperparameter optimization with Optuna, and multi-step future predictions.

## Key Features
- **Hybrid Architecture**: Combines Conv1D for feature extraction, LSTM for temporal dependencies, and attention for focus on critical time steps.
- **Technical Indicators**: Includes SMA, EMA, STD, and RSI (commented but extendable).
- **Optuna Integration**: Automated hyperparameter tuning for optimal performance.
- **Multi-Step Forecasting**: Predicts `future_days` into the future recursively.
- **Robust Preprocessing**: Min-Max scaling with lookback windows for sequential data.
- **Visualization**: Matplotlib plots for actual vs predicted prices and future projections.

## Installation
```bash
pip install requirements.txt
```

## Model Architecture
**1. Conv1D Layer:** Extracts local temporal features with ReLU activation.
```python
self.conv1d = nn.Conv1d(input_size, conv_out, kernel_size, stride, padding)
```
**2. LSTM Layer:** Captures long-term dependencies in sequential data.
```python
self.lstm = nn.LSTM(conv_out, hidden_size, num_layers, dropout=dropout)
```
**3. Attention Mechanism:** Learns to focus on relevant time steps.
```python
self.attn = nn.Linear(hidden_size, hidden_size)  # Softmax-weighted context
```
**4. Linear Layer:** Final prediction for the next time step.

## Usage

Train & Optimize (set train_and_optimize=True in main.py):
```bash
python main.py
```
