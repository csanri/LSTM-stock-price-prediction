import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import functools

from time import time
from utils import make_indicators


class Model(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers,
        dropout,
        lr,
        weight_decay,
        conv_out,
        kernel_size,
        stride,
        padding
    ):
        super(Model, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lr = lr
        self.weight_decay = weight_decay
        self.conv_out = conv_out
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Adding a 1D convolutional layer to extract features
        self.conv1d = nn.Conv1d(
            in_channels=self.input_size,
            out_channels=self.conv_out,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding
        )

        # Adding the LSTM layer for time series modeling
        self.lstm = nn.LSTM(
            input_size=conv_out,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=dropout,
            bias=True,
            batch_first=True
        )

        # Adding an attention layer to capture important features
        self.attn = nn.Linear(hidden_size, hidden_size)

        # A fully connected linear layer for the output
        self.linear = nn.Linear(self.hidden_size, 1)

        # Xavier initialization to improve convergence
        nn.init.xavier_uniform_(self.linear.weight)

        self.optimizer = torch.optim.Adam(self.parameters(),
                                          lr=self.lr,
                                          weight_decay=self.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.5)
        self.loss_fn = functools.partial(F.huber_loss, delta=0.02)

    def forward(self, x):
        x = x.to(self.device)

        # Applying the convolutional layer with ReLU activation
        x = x.transpose(1, 2)
        x = F.relu(self.conv1d(x))
        x = x.transpose(1, 2)

        batch_size = x.size(0)

        # Initializing the hidden and cell states for the LSTM
        h_0 = torch.zeros(self.num_layers,
                          batch_size,
                          self.hidden_size,
                          device=x.device).to(self.device)
        c_0 = torch.zeros(self.num_layers,
                          batch_size,
                          self.hidden_size,
                          device=x.device).to(self.device)

        x, _ = self.lstm(x, (h_0.detach(), c_0.detach()))

        attn_weights = F.softmax(self.attn(x), dim=1)

        # Wighted sum to get the context vector
        context = torch.sum(attn_weights * x, dim=1)

        x = self.linear(context)

        return x

    def train_model(self, epochs, data_loader, log):
        print(f"[Model] Training started for {epochs} epochs")
        # Storing the best loss for evaluation
        best_loss = float("inf")
        start_time = time()

        for epoch in range(epochs):
            epoch_loss = 0.0
            self.train()

            for x, y in data_loader:
                x, y = x.to(self.device), y.to(self.device)

                self.optimizer.zero_grad()

                # Forward pass
                output = self.forward(x)
                loss = self.loss_fn(output, y)

                # Backward pass
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()

            self.scheduler.step()

            # Logging the average loss every 10 epochs
            if log :
                avg_loss = epoch_loss / len(data_loader)
                if epoch == 0:
                    print("%9s %13s %13s" % ("Epoch", "Avg loss", "Elapsed time"))
                    print("=" * 37)

                if (epoch + 1) % 10 == 0:
                    elapsed_time = time() - start_time
                    print("%3d %1s %3d %13.6f %11.1f %1s" % (epoch + 1, "/", epochs, avg_loss, elapsed_time, "s"))

                if avg_loss < best_loss:
                    best_loss = avg_loss

                if (epoch + 1) == 100:
                    print("=" * 37)
                    print("%18s %18.6f" % ("Best loss:", best_loss))


    def predict(self, data):
        self.eval()
        with torch.no_grad():
            y_pred = self(data)
        return y_pred

        
    # Making predictions for an arbitrary amount of days into the future
    def predict_future(self, data, future_days, scaler, cols):
        # Using the most recent data only for predictions
        input_data = data[-1].cpu().numpy()
        preds = []

        self.eval()
        for _ in range(future_days):
            # Adding a batch dimension
            input_tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                pred = self(input_tensor)[-1].item()

            preds.append(pred)

            # Shift input data to incorporate the latest prediction for the next step
            new_input = np.zeros_like(input_data)
            new_input[:-1] = input_data[1:]
            new_input[-1, 0] = pred

            # Add a dummy column to match the scaler's input shape
            zeros_column = np.zeros((new_input.shape[0], 1))
            new_input = np.append(new_input, zeros_column, axis=1)

            # Inverse transform to calculate other features correctly
            df = pd.DataFrame(scaler.inverse_transform(new_input), columns=cols)

            # Create a temporary dataframe with indicators to avoid NaNs in original data frame
            temp_df = make_indicators(df)

            # Update the last row with calculated values
            df.iloc[-1] = temp_df.iloc[-1]
            df = scaler.transform(df)

            # Changing the input data so we can use predicted data for future predictions
            input_data = df[:, :-1]

        return preds


    def save_model(self, model_name):
        path = "models/" + model_name + ".pth"

        os.makedirs(os.path.dirname(path), exist_ok=True)

        torch.save({
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'lr': self.lr,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'conv_out': self.conv1d.out_channels
        }, path)

        print(f"[Model] Model is saved to {path}")


    def load_model(self, model_name):
        path = "models/" + model_name + ".pth"

        checkpoint = torch.load(path, map_location=self.device)

        self.load_state_dict(checkpoint["model_state_dict"], strict=False)

        self.lr = checkpoint["lr"]
        self.hidden_size = checkpoint["hidden_size"]
        self.num_layers = checkpoint["num_layers"]
        self.conv_out = checkpoint["conv_out"]

        print(f"[Model] Model loaded from {path}")

