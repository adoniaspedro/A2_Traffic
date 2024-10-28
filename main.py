"""
Traffic Flow Prediction with Neural Networks (SAEs, LSTM, GRU).
"""

import math
import warnings
import numpy as np
import pandas as pd
from data.data import process_data
from keras.models import load_model
import sklearn.metrics as metrics
import matplotlib as mpl
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# DTypePolicy to handle deserialization
class DTypePolicy:
    """ class to handle the DTypePolicy during deserialization."""
    def __init__(self, *args, **kwargs):
        self.name = 'dtype_policys'
        self.compute_dtype = 'float32'
        self.variable_dtype = 'float32'

# Define the custom objects dictionary
custom_objects = {'DTypePolicy': DTypePolicy}

def MAPE(y_true, y_pred):
    """Mean Absolute Percentage Error (MAPE)."""
    y = [x for x in y_true if x > 0]
    y_pred = [y_pred[i] for i in range(len(y_true)) if y_true[i] > 0]

    num = len(y_pred)
    sums = 0

    for i in range(num):
        tmp = abs(y[i] - y_pred[i]) / y[i]
        sums += tmp

    mape = sums * (100 / num)
    return mape

def eva_regress(y_true, y_pred):
    """Evaluation of regression results."""
    mape = MAPE(y_true, y_pred)
    vs = metrics.explained_variance_score(y_true, y_pred)
    mae = metrics.mean_absolute_error(y_true, y_pred)
    mse = metrics.mean_squared_error(y_true, y_pred)
    r2 = metrics.r2_score(y_true, y_pred)
    
    print('Explained Variance Score: %f' % vs)
    print('MAPE: %f%%' % mape)
    print('MAE: %f' % mae)
    print('MSE: %f' % mse)
    print('RMSE: %f' % math.sqrt(mse))
    print('R2: %f' % r2)

def plot_results(y_true, y_preds, names):
    """Plot the true and predicted data."""
    d = '2016-3-4 00:00'
    x = pd.date_range(d, periods=288, freq='5min')

    fig, ax = plt.subplots()
    ax.plot(x, y_true, label='True Data')

    for name, y_pred in zip(names, y_preds):
        ax.plot(x, y_pred, label=name)

    plt.legend()
    plt.grid(True)
    plt.xlabel('Time of Day')
    plt.ylabel('Flow')

    date_format = mpl.dates.DateFormatter("%H:%M")
    ax.xaxis.set_major_formatter(date_format)
    fig.autofmt_xdate()

    plt.show()

def main():
    # Load models with custom_objects
    lstm = load_model('model/lstm.h5', compile=False, custom_objects=custom_objects)
    gru = load_model('model/gru.h5', compile=False, custom_objects=custom_objects)
    saes = load_model('model/saes.h5', compile=False, custom_objects=custom_objects)
    my_model = load_model('model/model3p0.h5', compile=False, custom_objects=custom_objects)

    models = [lstm, gru, saes, my_model]
    names = ['LSTM', 'GRU', 'SAEs', 'model3p0']

    lag = 12
    file1 = 'data/train.csv'
    file2 = 'data/test.csv'
    _, _, X_test, y_test, scaler = process_data(file1, file2, lag)
    y_test = scaler.inverse_transform(y_test.reshape(-1, 1)).reshape(1, -1)[0]

    y_preds = []
    for name, model in zip(names, models):
        if name in ['SAEs', 'model3p0']:
            X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1]))
        else:
            X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

        predicted = model.predict(X_test)
        predicted = scaler.inverse_transform(predicted.reshape(-1, 1)).reshape(1, -1)[0]
        y_preds.append(predicted[:288])

        print(f"Model: {name}")
        eva_regress(y_test, predicted)

    plot_results(y_test[:288], y_preds, names)

if __name__ == '__main__':
    main()

