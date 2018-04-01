import os
import time
import warnings
import numpy as np
import pandas as pd
from tensorflow import set_random_seed
from keras.layers.core import Dense, Activation
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from sklearn.preprocessing import StandardScaler
from datetime import datetime
from dateutil import parser
import random as rn
import json

# This file will create your own returns models, feel free to play with the parameters of the neural networks

pd.options.display.max_rows = 100
np.random.seed(1)
set_random_seed(1)
rn.seed(1)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")

# Remember to add full path to btc.csv if you are not using a project or place btc.csv in your project's directory
df = pd.read_csv('btc.csv')
df = df.sort_values(by='Date')
df = df.drop(len(df)-1, axis=0)
df = df.drop(len(df)-1, axis=0)
df['Date'] = [parser.parse(x) for x in list(df['Date'])]
df.index = df['Date']
df = df.drop('Date', axis=1)
df = df.rename(columns={'Close Price': 'Price'})
df = df[100:]
df = df[df.index < datetime(2018, 3, 20)]

df['Returns'] = df['Price'].pct_change()
df = df[1:]

Obs = 12
sc = StandardScaler()
df['Returns'] = sc.fit_transform(df['Returns'].reshape(-1, 1))


for i in range(1, 5):
    df_slice = df[:-50*i]
    Data = np.asarray(df_slice['Returns'])
    Data = np.atleast_2d(Data)
    Data = Data.T
    X = np.atleast_3d(np.array([Data[start:start + Obs] for start in range(0, Data.shape[0] - Obs)]))
    y = Data[Obs:]
    model = Sequential()
    model.add(LSTM(input_shape=(1,), input_dim=1, output_dim=20, return_sequences=True))
    model.add(LSTM(input_shape=(1,), input_dim=1, output_dim=20, return_sequences=False))
    model.add(Dense(1))
    model.add(Activation('linear'))
    model.compile(loss="mse", optimizer="nadam")
    NN = model.fit(X, y, epochs=600, batch_size=50, verbose=2, shuffle=False)
    model_json = model.to_json()
    # Remember to add full path to the files if you are not using a project, for example:
    # lossname = "/home/user/Model_" + str(5-i) + "_loss.csv"
    # paramsname = "/home/user/Model_" + str(5-i) + "_params.json"
    # name = "/home/user/Model_" + str(5-i) + ".json"
    # h5name = "/home/user/Model_" + str(5-i) + "_weights.h5"
    lossname = "Model_" + str(5-i) + "_loss.csv"
    paramsname = "Model_" + str(5-i) + "_params.json"
    name = "Model_" + str(5-i) + ".json"
    h5name = "Model_" + str(5-i) + "_weights.h5"
    pd.DataFrame(NN.history).to_csv(lossname)
    model_params = json.dumps(NN.params)
    with open(paramsname, "w") as json_file:
        json_file.write(model_params)
    with open(name, "w") as json_file:
        json_file.write(model_json)
    model.save_weights(h5name)
    print "Done with model number", 5-i
    time.sleep(120)
