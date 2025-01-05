
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import torch
import torch.utils.data as data
import optuna
import json
import warnings

from sklearn.preprocessing import MinMaxScaler
from datetime import datetime

from LSTM import Model
from utils import make_indicators

warnings.filterwarnings("ignore", category=DeprecationWarning)
pd.set_option('display.max_columns', None)

# Setting the initial parameters
batch_size = 32
epochs = 100
lookback = 60
future_days = 30
opt_iter = 50
train_and_optimize = True
model_name = "LSTM_Conv1d_with_attn"

# Moving our model to the gpu if its available
device = "cuda" if torch.cuda.is_available() else "cpu"

scaler = MinMaxScaler(feature_range=(0, 1))

ticker = "AAPL"

stock_data = yf.Ticker(ticker)


def prepare_data(stock_data):
    df = pd.DataFrame(stock_data.history(period="max"))
    df = df.reset_index()
    df = df.drop([
        "Dividends",
        "Stock Splits",
        "High",
        "Low",
        "Open",
        "Volume",
        "Date"
    ], axis=1)

    df = make_indicators(df)
    df["Target"] = df["Close"].shift(-1)

    df = df.dropna()

    return df


df = prepare_data(stock_data)

cols = df.columns.tolist()


# Splitting the dataset with a moving lookback window
def split_dataset(df, lookback):
    data = []

    for i in range(len(df) - lookback):
        data.append(df[i:i + lookback])

    data = np.array(data)

    X_data = torch.tensor(data[:len(data), :-1, :-1],
                          dtype=torch.float32).to(device)
    y_data = torch.tensor(data[:len(data), -1, -1],
                          dtype=torch.float32).to(device)

    return X_data, y_data


# Making a 80:20 split
split = int(len(df) * 0.80)
train = df[0:split]
test = df[split:]

"""
Fitting and transforming both on the train and test data because fitting only
on the train data and then transforming both sets with it
would lead to the test data being distorted due to it having
higher values than the highest value of the train set
"""

train = scaler.fit_transform(train)
test = scaler.fit_transform(test)

X_train, y_train = split_dataset(train, lookback)
X_test, y_test = split_dataset(test, lookback)

y_train = y_train.view(-1, 1)
y_test = y_test.view(-1, 1)

loader = data.DataLoader(
    data.TensorDataset(X_train, y_train),
    shuffle=True,
    batch_size=batch_size
)

if train_and_optimize:
    # Setting up optuna for hyperparamter optimization
    def objective(trial):
        hidden_size = trial.suggest_int("hidden_size", 32, 128)
        num_layers = trial.suggest_int("num_layers", 2, 5)
        dropout = trial.suggest_float("dropout", 0.1, 0.5)
        lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-8, 1e-4, log=True)
        conv_out = trial.suggest_int("conv_out", 32, 128)
        kernel_size = trial.suggest_categorical("kernel_size", [3, 5, 7, 9])

        model = Model(
            input_size=X_train.shape[-1],
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            lr=lr,
            weight_decay=weight_decay,
            conv_out=conv_out,
            kernel_size=kernel_size,
            stride=1,
            padding=1
        ).to(device)

        model.train_model(epochs=epochs, data_loader=loader, log=False)

        model.eval()
        with torch.no_grad():
            y_train_pred = model(X_train)
            y_test_pred = model(X_test)
            train_loss = model.loss_fn(y_train_pred, y_train).item()
            test_loss = model.loss_fn(y_test_pred, y_test).item()

        print("=" * 23)
        print("%11s %11.6f" % ("Train loss:", train_loss))
        print("-" * 23)
        print("%11s %11.6f" % ("Test loss:", test_loss))
        print("=" * 23)

        return test_loss

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=opt_iter)

    best_params = study.best_params

    # best_params = {
    #     "hidden_size" :  128,
    #     "num_layers" : 3,
    #     "dropout" : 0.2,
    #     "lr" : 1e-4,
    #     "weight_decay" : 1e-5,
    #     "conv_out" : 128,
    #     "kernel_size" : 7
    # }

    model = Model(input_size=X_train.shape[-1],
                  stride=1,
                  padding=1,
                  **best_params).to(device)

    model.save_model(model_name=model_name)

    with open(f"models/{model_name}.json", "w") as f:
        json.dump(best_params, f)

else:
    # Initializing a default model
    with open(f"models/{model_name}.json", "r") as f:
        best_params = json.load(f)

    model = Model(input_size=X_train.shape[-1],
                  **best_params,
                  stride=1,
                  padding=1).to(device)

    model.load_model(model_name=model_name)

model.train_model(epochs=epochs, data_loader=loader, log=True)

model.eval()
with torch.no_grad():
    y_pred = model(X_test)
    y_train_pred = model(X_train)

test_loss = model.loss_fn(y_pred, y_test)

print("*" * 21)
print("%10s %10.6f" % ("Test loss:", test_loss))
print("*" * 21)

all_preds = []
x_preds = []
future_loss = []
num_chunks = len(y_test) // future_days

for i in range(num_chunks):
    start_idx = future_days * i
    end_idx = start_idx + future_days
    future_predictions = model.predict_future(X_test[start_idx:end_idx],
                                              future_days=future_days,
                                              scaler=scaler,
                                              cols=cols)

    all_preds.append(list(future_predictions))

    future_predictions = torch.tensor(future_predictions).to(device).view(-1, 1)

    future_x = np.arange(start_idx + future_days, end_idx + future_days)
    x_preds.append(future_x)

    test_chunk = y_test[start_idx:end_idx]

    future_loss.append(model.loss_fn(future_predictions[-1], test_chunk[-1]).item())


avg_future_loss = np.mean(future_loss)

print("*" * 25)
print("%12s %12.6f" % ("Future loss:", avg_future_loss))
print("*" * 25)

x_test = np.arange(0, len(y_test))

y_train = y_train.cpu().numpy()
y_train_pred = y_train_pred.cpu().numpy()
y_test = y_test.cpu().numpy()
y_pred = y_pred.cpu().numpy()

# Plotting the neccesarry data
plt.figure(figsize=(12, 6))

plt.plot(x_test, y_test,
         label='Actual Closing Price',
         color='blue')

plt.plot(x_test, y_pred,
         label='Predicted Closing Price',
         color='orange', linestyle='--')

for i in range(len(all_preds)):
    plt.plot(x_preds[i], all_preds[i], color='red')

plt.title(f"{ticker} Stock Price Movement Prediction")
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.savefig(f'snapshots/{ticker}_prediction_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png', dpi=300)
plt.show()

plt.figure(figsize=(8, 8))

plt.scatter(y_test, y_pred, label="Prediction accuracy")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
         'k--',
         lw=2,
         label="Perfect accuracy")

plt.xlabel('Test')
plt.ylabel('Prediction')
plt.legend()
plt.show()

