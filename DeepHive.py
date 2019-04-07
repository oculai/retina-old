import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib
import tensorflow
from keras.layers import LSTM, Dense, Dropout, TimeDistributed
from keras.models import Sequential

def scale_data(data):
    data = np.array(data).reshape(-1,5)
    scaler = joblib.load('C:/Users/Adrian/Desktop/Hackathons/HackABull2019/Models/scaler.save')
    return scaler.transform(data)

def get_scaler():
    return joblib.load('C:/Users/Adrian/Desktop/Hackathons/HackABull2019/Models/scaler.save')

def make_windows(data, size):
    X = []
    Y = []
    i = 0
    while(i+size) <= len(data)-1:
        X.append(data[i:i+size])
        Y.append(data[i + size])
        i += 1

    return X,Y

def prep_for_model(data, timesteps, features):
    return np.array(data).reshape(len(data), timesteps, features)

def build_model(weights_file):
    model = Sequential()
    model.add(LSTM(16, input_shape=(32,5), return_sequences=True))
    model.add(LSTM(16, return_sequences=True))
    model.add(Dropout(.2))
    model.add(LSTM(8, return_sequences=False))
    model.add(Dense(5))
    model.compile(optimizer='adam', loss='mse')
    model.load_weights(weights_file)

    return model

def get_prediction(data, model, scaler):
    return scaler.inverse_transform(model.predict(data))
"""
def get_next_ten(input, model, scaler):
    output_arr = []
    preds_arr = []
    first_pred = model.predict(input)
    output_arr = output_arr.append(scaler.inverse_transform(first_pred))
    for i in range(10):
        next_pred = model.predict(preds_arr[i])
        pred_arr = preds_arr.append(next_pred)
        output_arr = output_arr.append(scaler.inverse_transform(next_pred))

    return output_arr
"""
