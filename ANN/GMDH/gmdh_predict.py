from gmdhpy.gmdh import Regressor
from gmdhpy.plot_model import PlotModel
import pandas as pd
import tensorflow as tf
import os

from sklearn import preprocessing
import numpy as np



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

data = pd.read_csv("test_1.csv")
#print(data.loc[1])
model = Regressor(manual_best_neurons_selection=True,max_best_neurons_count=100,ref_functions=('linear_cov', 'quadratic', 'cubic', 'linear'))
# data_x = data.loc[0:274, ['A','B','C','D','E','F','G','H','I','J','K']].values
# data_y = data.loc[0:274, ['T1']].values
# test_x = data.loc[275:324, ['A','B','C','D','E','F','G','H','I','J','K']].values
# test_y = data.loc[275:324, ['T1']].values
data_x = data.loc[0:299, ['A','B','C','D','E']].values
#print(data_x)
data_y = data.loc[0:299, ['S']].values
test_x = data.loc[300:324, ['A','B','C','D','E']].values
test_y = data.loc[300:324, ['S']].values

mean = data_x.mean(axis=0)
data_x -= mean
std = data_x.std(axis=0)
data_x /= std

test_x -= mean
test_x /= std

model.fit(data_x, data_y)


pred_test_y = model.predict(test_x)

# print(pred_test_y)
pred_s = pred_test_y.reshape(25,1)
pred_RON = pred_test_y.reshape(25,1)
s = test_y.reshape(25,1)
RON = test_y.reshape(25,1)

#saver = tf.train.Saver()

#model_pred = "model_pred.h5"
#model.save(model_pred)


# 可视化  硫含量
# plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.rcParams['axes.unicode_minus'] = False
#
# plt.figure(figsize=(8, 4), dpi=80)
# # plt.subplot(2,1,1)
# plt.plot(range(len(s)),s, ls='-.',lw=2,c='r',label='真实值')
# plt.plot(range(len(pred_s)), pred_s, ls='-',lw=2,c='b',label='预测值')
# plt.legend()
# plt.xlabel('样本编号') #设置x轴的标签文本
# plt.ylabel('硫含量 μg/g') #设置y轴的标签文本




#可视化RON损失
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

plt.figure(figsize=(8, 4), dpi=80)
# plt.subplot(2,1,2)
plt.plot(range(len(RON)), RON, ls='-.',lw=2,c='r',label='真实值')
plt.plot(range(len(pred_RON)), pred_RON, ls='-',lw=2,c='b',label='预测值')
# 绘制网格
# plt.grid(alpha=0.4, linestyle=':')
plt.legend()
plt.xlabel('样本编号') #设置x轴的标签文本
plt.ylabel('RON损失') #设置y轴的标签文本
# 展示

plt.show()

