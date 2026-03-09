"""
Deep Learning — SmartCity Pulse
LSTM neural network for temperature time series forecasting.
"""

# import numpy for array operations
import numpy as np

# import pandas for data manipulation
import pandas as pd

# import tensorflow for deep learning
import tensorflow as tf

# import keras layers for building neural network
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# import sklearn tools for preprocessing and evaluation
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

# import mlflow for experiment tracking
import mlflow
import mlflow.keras

# import matplotlib for plotting
import matplotlib.pyplot as plt

# import os and sys
import os
import sys

# import logging
import logging

# create logger for this file
logger = logging.getLogger(__name__)


def prepare_sequences(series: np.ndarray, window_size: int = 5):
    """
    Converts time series into sequences for LSTM.

    LSTM needs input shape: (samples, timesteps, features)
    Example with window_size=5:
        Input:  [t1, t2, t3, t4, t5] → Output: t6
        Input:  [t2, t3, t4, t5, t6] → Output: t7

    Args:
        series: 1D array of temperature values
        window_size: how many past readings to use

    Returns:
        X: sequences of shape (samples, window_size, 1)
        y: next value after each sequence
    """

    # empty lists to store sequences and targets
    X, y = [], []

    # slide a window across the series
    # each window of 5 readings predicts the next reading
    for i in range(len(series) - window_size):

        # take window_size readings as input
        sequence = series[i : i + window_size]

        # take the next reading as target
        target = series[i + window_size]

        # append to lists
        X.append(sequence)
        y.append(target)

    # convert to numpy arrays
    X = np.array(X)
    y = np.array(y)

    # reshape X for LSTM — needs 3D input
    # shape: (samples, timesteps, features)
    # features=1 because we only use temperature
    X = X.reshape(X.shape[0], X.shape[1], 1)

    # log shapes
    logger.info(f"Sequences created — X: {X.shape} | y: {y.shape}")
    return X, y


def prepare_lstm_data(df: pd.DataFrame):
    """
    Prepares raw weather DataFrame for LSTM training.

    Args:
        df: Raw weather DataFrame from data_fetcher.py

    Returns:
        X_train, X_test, y_train, y_test, scaler
    """

    # extract temperature column as numpy array
    temps = df["temp"].values.reshape(-1, 1)

    # scale temperatures to range 0 to 1
    # MinMaxScaler works better than StandardScaler for LSTM
    scaler = MinMaxScaler(feature_range=(0, 1))

    # fit and transform — learns min and max from data
    temps_scaled = scaler.fit_transform(temps)

    # flatten back to 1D array
    temps_scaled = temps_scaled.flatten()

    # create sequences with window size 5
    X, y = prepare_sequences(temps_scaled, window_size=5)

    # split 80% train 20% test
    # important — do NOT shuffle time series data
    # order matters for temporal patterns
    split = int(len(X) * 0.8)

    # train set — first 80% of time
    X_train = X[:split]
    y_train = y[:split]

    # test set — last 20% of time
    X_test  = X[split:]
    y_test  = y[split:]

    # log split sizes
    logger.info(f"Train sequences: {len(X_train)} | Test sequences: {len(X_test)}")

    return X_train, X_test, y_train, y_test, scaler


def build_lstm_model(window_size: int = 5) -> Sequential:
    """
    Builds LSTM neural network architecture.

    Args:
        window_size: number of past timesteps used as input

    Returns:
        Compiled Keras Sequential model
    """

    # create sequential model — layers stacked in a line
    model = Sequential([

        # first LSTM layer
        # units=64 means 64 memory cells
        # return_sequences=True passes full sequence to next layer
        # input_shape=(window_size, 1) — 5 timesteps, 1 feature
        LSTM(
            units=64,
            return_sequences=True,
            input_shape=(window_size, 1)
        ),

        # dropout — randomly turns off 20% of neurons
        # prevents overfitting
        Dropout(0.2),

        # second LSTM layer
        # return_sequences=False — only return last timestep
        LSTM(units=32, return_sequences=False),

        # dropout again after second LSTM
        Dropout(0.2),

        # dense layer — fully connected
        # relu activation — outputs 0 for negative values
        Dense(units=16, activation="relu"),

        # output layer — single neuron predicts one temperature value
        # no activation — linear output for regression
        Dense(units=1)
    ])

    # compile model
    # adam optimizer adapts learning rate automatically
    # mse loss is standard for regression
    model.compile(
        optimizer="adam",
        loss="mse",
        metrics=["mae"]
    )

    # print model architecture
    model.summary()

    # log that model was built
    logger.info("LSTM model built successfully")
    return model


def train_lstm(df: pd.DataFrame) -> dict:
    """
    Full LSTM training pipeline.

    Args:
        df: Raw weather DataFrame from data_fetcher.py

    Returns:
        Dictionary with model metrics
    """

    # log that training is starting
    logger.info("Starting LSTM training...")

    # step 1 — prepare data
    X_train, X_test, y_train, y_test, scaler = prepare_lstm_data(df)

    # step 2 — build model
    model = build_lstm_model(window_size=5)

    # step 3 — define early stopping
    # stops training if val_loss does not improve for 10 epochs
    # restores best weights automatically
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True
    )

    # step 4 — train model
    logger.info("Training LSTM...")

    # set mlflow experiment
    mlflow.set_experiment("smartcity-deep-learning")

    # start mlflow run
    with mlflow.start_run(run_name="lstm_temperature"):

        # fit model on training data
        history = model.fit(
            X_train, y_train,
            epochs=100,             # maximum 100 training rounds
            batch_size=8,           # process 8 samples at a time
            validation_split=0.2,   # use 20% of train for validation
            callbacks=[early_stopping],
            verbose=1               # show training progress
        )

        # step 5 — evaluate on test data
        # predict scaled values
        y_pred_scaled = model.predict(X_test)

        # inverse transform — convert back to real temperatures
        y_pred = scaler.inverse_transform(y_pred_scaled)
        y_true = scaler.inverse_transform(y_test.reshape(-1, 1))

        # calculate metrics on real temperature values
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae  = mean_absolute_error(y_true, y_pred)

        # log metrics to mlflow
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_param("epochs_trained", len(history.history["loss"]))
        mlflow.log_param("window_size", 5)
        mlflow.log_param("lstm_units_1", 64)
        mlflow.log_param("lstm_units_2", 32)

        # save model in keras format
        model.save("models/lstm_model.keras")

        # log that model was saved
        logger.info("Model saved to models/lstm_model.keras")

        # print results
        print(f"\n✅ LSTM Temperature Forecasting")
        print(f"   Epochs trained : {len(history.history['loss'])}")
        print(f"   RMSE           : {rmse:.4f}°C")
        print(f"   MAE            : {mae:.4f}°C")

        # step 6 — plot training history and predictions
        # create figure with 2 subplots side by side
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # plot 1 — training and validation loss over epochs
        ax1.plot(
            history.history["loss"],
            label="Train Loss",
            color="blue"
        )
        ax1.plot(
            history.history["val_loss"],
            label="Val Loss",
            color="orange"
        )
        ax1.set_title("LSTM Training Loss")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss (MSE)")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # plot 2 — actual vs predicted temperatures
        ax2.plot(
            y_true,
            label="Actual Temp",
            color="blue",
            marker="o"
        )
        ax2.plot(
            y_pred,
            label="Predicted Temp",
            color="red",
            marker="x",
            linestyle="--"
        )
        ax2.set_title("Actual vs Predicted Temperature")
        ax2.set_xlabel("Time Step")
        ax2.set_ylabel("Temperature (°C)")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # adjust layout so plots dont overlap
        plt.tight_layout()

        # save plot to file
        plt.savefig(
            "models/lstm_predictions.png",
            dpi=150,
            bbox_inches="tight"
        )

        # show plot
        plt.show()

        # log plot to mlflow
        mlflow.log_artifact("models/lstm_predictions.png")

        # return results dictionary
        return {
            "rmse"          : rmse,
            "mae"           : mae,
            "epochs_trained": len(history.history["loss"])
        }


# runs only when you run this file directly
if __name__ == "__main__":

    # add pipeline folder to path
    sys.path.append("src/pipeline")

    # import data fetcher to get live weather data
    from data_fetcher import CityDataFetcher

    # fetch live weather data
    print("Fetching live weather data...")
    fetcher = CityDataFetcher()
    df = fetcher.fetch_weather("Mumbai")

    # print data info
    print(f"\nData shape       : {df.shape}")
    print(f"Temperature range: {df['temp'].min():.1f}°C to {df['temp'].max():.1f}°C")
    print(f"First reading    : {df['temp'].iloc[0]:.1f}°C")
    print(f"Last reading     : {df['temp'].iloc[-1]:.1f}°C")

    # create models folder
    os.makedirs("models", exist_ok=True)

    # train LSTM model
    print("\n" + "="*50)
    print("TRAINING LSTM NEURAL NETWORK")
    print("="*50)
    results = train_lstm(df)

    # print final results
    print("\n" + "="*50)
    print("LSTM RESULTS")
    print("="*50)
    print(f"\nEpochs trained : {results['epochs_trained']}")
    print(f"RMSE           : {results['rmse']:.4f}°C")
    print(f"MAE            : {results['mae']:.4f}°C")
    print(f"\nInterpretation:")
    print(f"   MAE = {results['mae']:.2f}°C means predictions are")
    print(f"   off by {results['mae']:.2f}°C on average")
    print(f"\nModel saved to : models/lstm_model.keras")
    print(f"Plot saved to  : models/lstm_predictions.png")