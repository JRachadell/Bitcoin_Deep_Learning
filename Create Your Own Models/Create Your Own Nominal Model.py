import os
import warnings
import numpy as np
import pandas as pd
from tensorflow import set_random_seed
from keras.layers.core import Dense, Activation
from keras.layers.recurrent import LSTM
from keras.models import Sequential
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from datetime import datetime
from dateutil import parser
import random as rn
import json

# This file will create your own nominal model, feel free to play with the parameters of the strategies
# and the neural network

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
df = df[800:]
df = df[df.index < datetime(2018, 3, 20)]

Obs = 20
sc = StandardScaler()
df['Price'] = sc.fit_transform(df['Price'].reshape(-1, 1))

Data = np.asarray(df['Price'])
Data = np.atleast_2d(Data)
Data = Data.T

X = np.atleast_3d(np.array([Data[start:start + Obs] for start in range(0, Data.shape[0] - Obs)]))
y = Data[Obs:]
print len(X), len(y)


model = Sequential()
model.add(LSTM(input_shape=(1,), input_dim=1, output_dim=15, return_sequences=True))
model.add(LSTM(input_shape=(1,), input_dim=1, output_dim=15, return_sequences=False))
model.add(Dense(1))
model.add(Activation('linear'))
model.compile(loss="mape", optimizer="rmsprop")
NN = model.fit(X, y, epochs=200, batch_size=50, verbose=2, shuffle=False)


Predictions = [model.predict(np.asarray(df['Price'])[i:i+Obs].reshape(1, Obs, 1)) for i in range(len(df)-Obs)]
Predictions = [df['Price'].iloc[0]]*Obs + [val[0][0] for val in Predictions]
df['Predictions'] = Predictions
df['Price'] = sc.inverse_transform(df['Price'])
df['Predictions'] = sc.inverse_transform(df['Predictions'])

# Uncomment the next lines to plot the Price and Predictions
# plt.xlabel('Date')
# plt.ylabel('Price')
# plt.plot(df['Price'], 'b', label='Price')
# plt.plot(df['Predictions'], 'r', label='Prediction')
# plt.legend(loc='upper left', shadow=True, fontsize='large')


def ratios(x, y, z):
    predictions = list(df['Predictions'])[:-1]
    price = list(df['Price'])[:-1]
    corrects = []
    for i in range(len(predictions)-y, len(predictions)-z):
        if (predictions[i] * (1.0 + (x/100.0))) > price[i] > (predictions[i] * (1.0 - (x/100.0))):
            corrects.append(1.0)
        else:
            corrects.append(0.0)
    return np.average(corrects) * 100


def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


MAPE = [mape(df['Price'][-500:], df['Predictions'][-500:]),
        mape(df['Price'][-1000:-500], df['Predictions'][-1000:-500]),
        mape(df['Price'][-1500:-1000], df['Predictions'][-1500:-1000]),
        mape(df['Price'][-2000:-1500], df['Predictions'][-2000:-1500])]

Col1 = [ratios(15, i, i-500) for i in range(500, 2001, 500)]
Col2 = [ratios(10, i, i-500) for i in range(500, 2001, 500)]
Col3 = [ratios(5, i, i-500) for i in range(500, 2001, 500)]

print pd.DataFrame([MAPE, Col1, Col2, Col3],
             index=["Mape", "Ratio 15%", "Ratio 10%", "Ratio 5%"],
             columns=['0-500', "500-1000", "1000-1500", "1500-2000"])


df['Vol'] = (df['Price'].pct_change() * 100.0).rolling(30).std(ddof=0)

# Uncomment the next lines to plot the Volatility
# plt.xlabel('Date')
# plt.ylabel('Volatility of Returns')
# plt.plot(df.index, df['Vol'])
# plt.legend(loc='upper center', shadow=True, fontsize='large')

dff = df[-500:]


def full_pred(x, y):
    btc = [5.0] + [np.nan] * (len(dff)-1)
    cash = [0] + [np.nan] * (len(dff)-1)
    prediction = list(dff['Predictions'])
    price = list(dff['Price'])
    for i in range(1, len(dff)-1):
        for j in range(0, 2):
            if prediction[i+1] > (prediction[i] * (1.0 + y/100.0)):
                if cash[i] > price[i]:
                    btc[i] = btc[i-1] + 1.0
                    cash[i] = cash[i-1] - price[i]
                elif price[i] > cash[i-1] > 0.0:
                    btc[i] = btc[i-1] + (cash[i-1] / price[i])
                    cash[i] = 0.0
                else:
                    btc[i] = btc[i - 1]
                    cash[i] = cash[i - 1]
            elif (prediction[i] * (1.0 - x/100.0)) > prediction[i+1]:
                if btc[i-1] >= 1.0:
                    btc[i] = btc[i-1] - 1.0
                    cash[i] = cash[i-1] + price[i]
                elif 1.0 > btc[i-1] > 0.0:
                    btc[i] = 0.0
                    cash[i] = cash[i-1] + (price[i] * btc[i-1])
                else:
                    btc[i] = btc[i - 1]
                    cash[i] = cash[i - 1]
            else:
                btc[i] = btc[i-1]
                cash[i] = cash[i-1]
    dff['BTC'] = btc
    dff['Cash'] = cash
    dff['Strategy'] = dff['Price'] * dff['BTC'] + dff['Cash']
    dff['BuyNHold'] = 5.0 * dff['Price']
    dff['St_VS_BnH_N'] = dff['Strategy'] - dff['BuyNHold']
    dff['St_VS_BnH_P'] = ((dff['Strategy'] - dff['BuyNHold']) / dff['BuyNHold']) * 100.0
    print "Strategy vs BuyNHold:", (dff['Strategy'][len(dff)-2] - dff['BuyNHold'][len(dff)-2])


def out_of_conf(x, y):
    btc = [5.0] + [np.nan] * (len(dff)-1)
    cash = [0] + [np.nan] * (len(dff)-1)
    prediction = list(dff['Predictions'])
    price = list(dff['Price'])
    for i in range(1, len(dff)):
        if (prediction[i] * (1.0 - (y/100.0))) > price[i]:
            if cash[i-1] > price[i]:
                btc[i] = btc[i-1] + 1
                cash[i] = cash[i-1] - price[i]
            elif price[i] > cash[i-1] > 0.0:
                btc[i] = btc[i-1] + (cash[i-1] / price[i])
                cash[i] = 0.0
            else:
                btc[i] = btc[i - 1]
                cash[i] = cash[i - 1]
        elif price[i] > (prediction[i] * (1.0 + (x/100.0))):
            if btc[i-1] >= 1.0:
                btc[i] = btc[i-1] - 1
                cash[i] = cash[i-1] + price[i]
            elif 1.0 > btc[i-1] > 0.0:
                btc[i] = 0
                cash[i] = cash[i-1] + (price[i] * btc[i-1])
            else:
                btc[i] = btc[i - 1]
                cash[i] = cash[i - 1]
        else:
            btc[i] = btc[i-1]
            cash[i] = cash[i-1]
    dff['BTC'] = btc
    dff['Cash'] = cash
    dff['Strategy'] = dff['Price'] * dff['BTC'] + dff['Cash']
    dff['BuyNHold'] = 5.0 * dff['Price']
    dff['St_VS_BnH_N'] = dff['Strategy'] - dff['BuyNHold']
    dff['St_VS_BnH_P'] = ((dff['Strategy'] - dff['BuyNHold']) / dff['BuyNHold']) * 100.0
    print "Strategy vs BuyNHold:", (dff['Strategy'][len(dff)-2] - dff['BuyNHold'][len(dff)-2])


def out_of_conf_vol(x, y):
    btc = [5.0] + [np.nan] * (len(dff)-1)
    cash = [0] + [np.nan] * (len(dff)-1)
    prediction = list(dff['Predictions'])
    price = list(dff['Price'])
    vol = list(dff['Vol'])
    for i in range(1, len(dff)):
        if (prediction[i] * (1.0 - ((vol[i]+y)/100.0))) > price[i]:
            if cash[i-1] > price[i]:
                btc[i] = btc[i-1] + 1
                cash[i] = cash[i-1] - price[i]
            elif price[i] > cash[i-1] > 0.0:
                btc[i] = btc[i-1] + (cash[i-1] / price[i])
                cash[i] = 0.0
            else:
                btc[i] = btc[i - 1]
                cash[i] = cash[i - 1]
        elif price[i] > (prediction[i] * (1.0 + ((vol[i]+x)/100.0))):
            if btc[i-1] >= 1.0:
                btc[i] = btc[i-1] - 1
                cash[i] = cash[i-1] + price[i]
            elif 1.0 > btc[i-1] > 0.0:
                btc[i] = 0
                cash[i] = cash[i-1] + (price[i] * btc[i-1])
            else:
                btc[i] = btc[i - 1]
                cash[i] = cash[i - 1]
        else:
            btc[i] = btc[i-1]
            cash[i] = cash[i-1]
    dff['BTC'] = btc
    dff['Cash'] = cash
    dff['Strategy'] = dff['Price'] * dff['BTC'] + dff['Cash']
    dff['BuyNHold'] = 5.0 * dff['Price']
    dff['St_VS_BnH_N'] = dff['Strategy'] - dff['BuyNHold']
    dff['St_VS_BnH_P'] = ((dff['Strategy'] - dff['BuyNHold']) / dff['BuyNHold']) * 100.0
    print "Strategy vs BuyNHold:", (dff['Strategy'][len(dff)-2] - dff['BuyNHold'][len(dff)-2])


def full_pred_vol(x, y):
    btc = [5.0] + [np.nan] * (len(dff)-1)
    cash = [0] + [np.nan] * (len(dff)-1)
    prediction = list(dff['Predictions'])
    price = list(dff['Price'])
    vol = list(dff['Vol'])
    for i in range(1, len(dff)-1):
        for j in range(0, 2):
            if prediction[i+1] > (prediction[i] * (1.0 + ((vol[i]+y)/100.0))):
                if cash[i] > price[i]:
                    btc[i] = btc[i-1] + 1.0
                    cash[i] = cash[i-1] - price[i]
                elif price[i] > cash[i-1] > 0.0:
                    btc[i] = btc[i-1] + (cash[i-1] / price[i])
                    cash[i] = 0.0
                else:
                    btc[i] = btc[i - 1]
                    cash[i] = cash[i - 1]
            elif (prediction[i] * (1.0 - ((vol[i]+x)/100.0))) > prediction[i+1]:
                if btc[i-1] >= 1.0:
                    btc[i] = btc[i-1] - 1.0
                    cash[i] = cash[i-1] + price[i]
                elif 1.0 > btc[i-1] > 0.0:
                    btc[i] = 0.0
                    cash[i] = cash[i-1] + (price[i] * btc[i-1])
                else:
                    btc[i] = btc[i - 1]
                    cash[i] = cash[i - 1]
            else:
                btc[i] = btc[i-1]
                cash[i] = cash[i-1]
    dff['BTC'] = btc
    dff['Cash'] = cash
    dff['Strategy'] = dff['Price'] * dff['BTC'] + dff['Cash']
    dff['BuyNHold'] = 5.0 * dff['Price']
    dff['St_VS_BnH_N'] = dff['Strategy'] - dff['BuyNHold']
    dff['St_VS_BnH_P'] = ((dff['Strategy'] - dff['BuyNHold']) / dff['BuyNHold']) * 100.0
    dff['BnH_Vol'] = (dff['BuyNHold'].pct_change() * 100.0).rolling(30).std(ddof=0)
    dff['Strategy_Vol'] = (dff['Strategy'].pct_change() * 100.0).rolling(30).std(ddof=0)
    print "Strategy vs BuyNHold:", (dff['Strategy'][len(dff)-2] - dff['BuyNHold'][len(dff)-2])


# Uncomment any of the next lines to run a strategy
# full_pred(3.0, 3.0)
# out_of_conf(3.0, 3.0)
# full_pred_vol(1.0, 1.0)
# out_of_conf_vol(1.0, 1.0)


# Uncomment the next lines to plot the strategy vs BuyNHold
# plt.xlabel('Date')
# plt.ylabel('US$')
# plt.plot(dff.index, dff['Strategy'])
# plt.plot(dff.index, dff['BuyNHold'])
# plt.legend(shadow=True, fontsize='large')


# Uncomment the next lines to plot the out_of_conf with confidence lines
# plt.xlabel('Date')
# plt.ylabel('US$')
# plt.plot(dff.index, dff['Price'], color="red")
# plt.plot(dff.index, dff['Predictions'], color="b", alpha=0.1)
# plt.fill_between(dff.index, dff['Predictions']*1.03, dff['Predictions']*0.97, color='b', alpha=0.3)
# plt.legend(shadow=True, fontsize='large')


# Uncomment the next lines save the model, Remember to add full path if you are not using a project
# pd.DataFrame(NN.history).to_csv("BTC_nominal_loss.csv")
# model_params = json.dumps(NN.params)
# with open("BTC_nominal_params.json", "w") as json_file:
#   json_file.write(model_params)
# model_json = model.to_json()
# with open("BTC_nominal.json", "w") as json_file:
#   json_file.write(model_json)
# model.save_weights("BTC_nominal_weights.h5")
