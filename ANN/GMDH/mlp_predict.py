import pandas as pd
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"   # GTX 1050 Ti

from keras import models
from keras import layers
import numpy as np
from keras import backend as K
from keras.models import Sequential #顺序模型
from keras.layers import Dense
from sklearn.preprocessing import StandardScaler
import keras.backend as K
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from sklearn import preprocessing
import numpy as np
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet

def r2(y_true, y_pred):
    a = K.square(y_pred - y_true)
    b = K.sum(a)
    c = K.mean(y_true)
    d = K.square(y_true - c)
    e = K.sum(d)
    f = 1 - b/e
    return f


data = pd.read_csv("test_1.csv")

# train_x = data.loc[0:319, ['A','B','C','D','E','F','G','H','I','J','K']].values
# # print(train_data)
#
# train_y = data.loc[0:319, ['T1','T2']].values
#
# # print(train_targets)
#
# test_x = data.loc[320:324, ['A','B','C','D','E','F','G','H','I','J','K']].values
# # print(test_data)
# test_y = data.loc[320:324, ['T1','T2']]

train_x = data.loc[0:319, ['A','B','C','D','E']].values
# print(train_data)

train_y = data.loc[0:319, ['S','RONS']].values

# print(train_targets)

test_x = data.loc[320:324, ['A','B','C','D','E']].values
# print(test_data)
test_y = data.loc[320:324, ['S','RONS']]

mean = train_x.mean(axis=0)
train_x -= mean
std = train_x.std(axis=0)
train_x /= std

test_x -= mean
test_x /= std

model = models.Sequential()
model.add(layers.Dense(64, activation='relu',
                           input_shape=(train_x.shape[1],)))
# x_tmp = BatchNormalization(x_tmp, training=False)
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(2))

# 训练神经网络
# model.compile(loss='mse', optimizer='rmsprop', metrics=['mae',r2])
model.compile(loss='mse', optimizer='rmsprop', metrics=['mse'])

batch_size = 32
training_epochs = 500

history = model.fit(train_x, train_y, batch_size=batch_size, epochs=training_epochs)

pred_test_y = model.predict(test_x)
test_y = test_y.values

print(pred_test_y)
pred_s = pred_test_y[:,0].reshape(5,1)
pred_RON = pred_test_y[:,1].reshape(5,1)
s = test_y[:,0].reshape(5,1)
RON = test_y[:,1].reshape(5,1)

model_pred = "model_pred.h5"
model.save(model_pred)

# 可视化----1
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

plt.figure(figsize=(8, 4), dpi=80)
plt.subplot(2,1,1)
plt.plot(range(len(s)),s, ls='-.',lw=2,c='r',label='真实值')
plt.plot(range(len(pred_s)), pred_s, ls='-',lw=2,c='b',label='预测值')
plt.legend(loc=1)
plt.xlabel('样本编号') #设置x轴的标签文本
plt.ylabel('硫含量 μg/g') #设置y轴的标签文本

plt.subplot(2,1,2)
plt.plot(range(len(RON)), RON, ls='-.',lw=2,c='r',label='真实值')
plt.plot(range(len(pred_RON)), pred_RON, ls='-',lw=2,c='b',label='预测值')
# 绘制网格
# plt.grid(alpha=0.4, linestyle=':')
plt.legend(loc=2)
plt.xlabel('样本编号') #设置x轴的标签文本
plt.ylabel('RON损失') #设置y轴的标签文本
# 展示
plt.show()
#
