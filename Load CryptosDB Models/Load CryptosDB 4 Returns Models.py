import os
import warnings
import numpy as np
import pandas as pd
from tensorflow import set_random_seed
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from dateutil import parser
from datetime import datetime
from keras.models import model_from_json
import itertools
import random as rn

# This file will load TheCryptosDB's returns 4 models, feel free to play with the parameters of the strategies

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

# Remember to add full path to the files if you are not using a project or place the files in your project's directory
loaded_models = []
for i in range(1, 5):
    name = "4_NonBias_Models/CryptosDB_Model_" + str(i) + ".json"
    h5name = "4_NonBias_Models/CryptosDB_Model_" + str(i) + "_weights.h5"
    json_file = open(name, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(h5name)
    loaded_models.append(loaded_model)

df_slice = df[:-150]

Predictions = [loaded_models[0].predict(np.asarray(df['Returns'])[j-Obs:j].reshape(1, Obs, 1))
               for j in range(Obs, len(df_slice))]
Predictions = [df['Returns'].iloc[0]]*Obs + [val[0][0] for val in Predictions]

for i in range(2, 5):
    if i != 4:
        df_slice = df[:-50*(4-i)]
    elif i == 4:
        df_slice = df
    floor = len(df_slice)-50
    roof = len(df_slice)
    values = [loaded_models[i-1].predict(np.asarray(df_slice['Returns'])[j-Obs:j].reshape(1, Obs, 1))
                                 for j in range(floor, roof)]
    Predictions = Predictions + [val[0][0] for val in values]
    print i, floor, roof, roof-floor


df['Predictions'] = Predictions
df['Returns'] = sc.inverse_transform(df['Returns'])
df['Predictions'] = sc.inverse_transform(df['Predictions'])

dff = df[-200:]

rets = []
pred_rets = []
equals = []
Returns = list(dff['Returns'])
Prediction = list(dff['Predictions'])

for i in range(len(dff)):
    if Returns[i] > 0.03:
        rets.append("Higher")
    elif 0.03 >= Returns[i] >= -0.03:
        rets.append("Neutral")
    elif -0.03 > Returns[i]:
        rets.append("Lower")
    if Prediction[i] > 0.03:
        pred_rets.append("Higher")
    elif 0.05 >= Prediction[i] >= -0.03:
        pred_rets.append("Neutral")
    elif -0.03 > Prediction[i]:
        pred_rets.append("Lower")

for i in range(len(dff)):
    if rets[i] == pred_rets[i]:
        equals.append(1)
    else:
        equals.append(0)

print "% Total Correct:", np.average(equals) * 100
print confusion_matrix(rets, pred_rets)

class_names = "Higher", "Lower", "Neutral"


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# Uncomment the next lines to plot the confusion matrix
# plt.figure()
# plot_confusion_matrix(confusion_matrix(rets, pred_rets), classes=class_names, title='Confusion matrix')


def ret_pred(x, y):
    btc = [5.0] + [np.nan] * (len(dff)-1)
    cash = [0] + [np.nan] * (len(dff)-1)
    prediction = list(dff['Predictions'])[1:] + [np.nan]
    price = list(dff['Price'])
    for i in range(1, len(dff)-1):
        for j in range(0, 2):
            if prediction[i] > x/100.0:
                if cash[i] > price[i]:
                    btc[i] = btc[i-1] + 1.0
                    cash[i] = cash[i-1] - price[i]*1.005
                elif price[i] > cash[i-1] > 0.0:
                    btc[i] = btc[i-1] + (cash[i-1] / (price[i]*1.005))
                    cash[i] = 0.0
                else:
                    btc[i] = btc[i - 1]
                    cash[i] = cash[i - 1]
            elif -y/100.0 > prediction[i]:
                if btc[i-1] >= 1.0:
                    btc[i] = btc[i-1] - 1.0
                    cash[i] = cash[i-1] + price[i]*0.995
                elif 1.0 > btc[i-1] > 0.0:
                    btc[i] = 0.0
                    cash[i] = cash[i-1] + ((price[i]*0.995) * btc[i-1])
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


# Uncomment the next line to run the strategy
ret_pred(7.0, 7.0)
ret_pred(6.0, 6.0)
ret_pred(5.0, 5.0)
ret_pred(4.0, 4.0)
ret_pred(3.0, 3.0)
ret_pred(2.0, 2.0)
ret_pred(1.0, 1.0)
ret_pred(0.0, 0.0)


# Uncomment the next lines to plot the strategy vs BuyNHold
# plt.xlabel('Date')
# plt.ylabel('US$')
# plt.plot(dff.index, dff['Strategy'])
# plt.plot(dff.index, dff['BuyNHold'])
# plt.legend(loc='upper left', shadow=True, fontsize='large')
