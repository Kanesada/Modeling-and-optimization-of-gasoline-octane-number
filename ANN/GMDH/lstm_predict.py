import pandas as pd
from sklearn.preprocessing import LabelEncoder,MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import matplotlib.pyplot as pyplot
from numpy import concatenate
from math import sqrt
import matplotlib.pyplot as plt

data = pd.read_csv("D://PyCharm 2020.1.1//PyCharmProjects//math//test_1.csv")

train_X = data.loc[0:299, ['A','B','C','D','E']].values
train_Y = data.loc[0:299, ['S']].values
test_X = data.loc[300:324, ['A','B','C','D','E']].values
test_Y = data.loc[300:324, ['S']].values

# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))

model = Sequential()
# 50为units，是指每个cell中隐藏层结构的参数数量，即经过一个cell之后，数据的维度变为50。
# 假如我们输入有100个句子，每个句子都由5个单词组成，而每个单词用64维的词向量表示。
# 那么samples=100，timesteps=5，input_dim=64，
# 可以简单地理解timesteps就是输入序列的长度。
# input_shape：输入形状为（1，8）
# model.add(Bidirectional(LSTM(50, return_sequences=True),
#                         input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')
# fit network
history = model.fit(train_X,train_Y,epochs=50,batch_size=72,validation_data=(test_X, test_Y),
                    verbose=2,shuffle=False)

pred_test_y = model.predict(test_X)

# print(pred_test_y)
pred_s = pred_test_y.reshape(25,1)
pred_RON = pred_test_y.reshape(25,1)
s = test_Y.reshape(25,1)
RON = test_Y.reshape(25,1)

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

# model.save('pred_model.model')
# plot history
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
# 显示图例：
pyplot.legend()
pyplot.show()