import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
import pathlib

def run_analysis(symbol):
    stock = pd.read_csv(pathlib.Path(__file__).parent.parent/"fetchedData"/f'{symbol}.csv')

    stock = stock[['Date', 'Close']]

    # Feature preprocessing
    newStock = stock.drop('Date', axis = 1)
    newStock = newStock.reset_index(drop = True)
    T = newStock.values
    T = T.astype('float32')
    T = np.reshape(T, (-1, 1))

    # Min-max scaling to get the values in the range [0,1] to reduce the training time
    from sklearn.preprocessing import MinMaxScaler

    scaler = MinMaxScaler(feature_range = (0, 1))
    T = scaler.fit_transform(T)

    # 80-20 split
    train_size = int(len(T) * 0.80)
    test_size = int(len(T) - train_size)
    train, test = T[0:train_size,:], T[train_size:len(T),:]

    # Method for create features from the time series data
    def create_features(data, window_size):
        X, Y = [], []
        for i in range(len(data) - window_size - 1):
            window = data[i:(i + window_size), 0]
            X.append(window)
            Y.append(data[i + window_size, 0])
        return np.array(X), np.array(Y)

    window_size = 40
    X_train, Y_train = create_features(train, window_size)

    X_test, Y_test = create_features(test, window_size)

    # Reshape to the format of [samples, time steps, features] to fit what the lstm needs
    X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))

    X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

    T_shape = T.shape
    train_shape = train.shape
    test_shape = test.shape
    
    # Setting seed for reproducibility 
    tf.random.set_seed(11)
    np.random.seed(11)

    # Building model
    model = Sequential()

    model.add(LSTM(units = 50, activation = 'relu',input_shape = (X_train.shape[1], window_size)))
    model.add(Dropout(0.2))

    # Output layer
    model.add(Dense(1))
    model.compile(loss = 'mean_squared_error', optimizer = 'adam')

    # Save models
    
    filepath = str(pathlib.Path(__file__).parent.parent.parent/'saved_models/model_epoch_{epoch:02d}.keras')

    checkpoint = ModelCheckpoint(filepath = filepath,
                                monitor = 'val_loss',
                                verbose = 1,
                                save_best_only = True,
                                mode ='min'
                                )

    history = model.fit(X_train, Y_train, epochs = 100, batch_size = 20, validation_data = (X_test, Y_test), 
                        callbacks = [checkpoint], 
                        verbose = 1, shuffle = False)

    all = pathlib.Path(__file__).parent.parent.parent / 'saved_models'

    best_model = load_model(sorted(all.glob('*.keras'))[-1])

    # Predicting and inverse transforming the predictions

    train_predict = best_model.predict(X_train)
    Y_hat_train = scaler.inverse_transform(train_predict)
    test_predict = best_model.predict(X_test)
    Y_hat_test = scaler.inverse_transform(test_predict)

    Y_test = scaler.inverse_transform([Y_test])
    Y_train = scaler.inverse_transform([Y_train])
    Y_hat_train = Y_hat_train.flatten()
    Y_hat_test = Y_hat_test.flatten()

    Y_train = Y_train.flatten()
    Y_test = Y_test.flatten()

    last_window = T[-window_size:, 0]
    future_predictions = []
    current_window = last_window.copy()
    for _ in range(30): # change it to however far you want to predict (days)
        sliding_window = np.reshape(current_window, ( 1, 1, len(current_window)))
        pred = best_model.predict(sliding_window)[0,0] #2D array
        future_predictions.append(pred)
        current_window = np.append(current_window[1:],pred)

    future_predictions = scaler.inverse_transform(np.reshape(future_predictions, (-1, 1))).flatten()

    Y = np.append(Y_train, Y_test)
    Y_hat = np.append(Y_hat_train, Y_hat_test)

    result_df = pd.DataFrame()

    result_df['Actual_Y'] = Y
    result_df['Predicted_Y'] = Y_hat
    predicted_y = np.append(result_df['Predicted_Y'].values, future_predictions) 
 
    from plotly import graph_objs as go
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=result_df['Actual_Y'], name='Actual'))
    fig.add_trace(go.Scatter(y=predicted_y, name='Predicted'))
    fig.update_layout(title='Actual vs Predicted', xaxis_title='Index', yaxis_title='Price')
    fig.show()
    
    csvs = pathlib.Path(__file__).parent.parent / 'fetchedData'
    for file in all.glob('*.keras'):
        file.unlink()
    for file in csvs.glob('*.csv'):
        file.unlink()

if __name__ == "__main__":
    print("please run main.py")
    exit()