import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import joblib
import numpy as np

# Split data set and preprocess
def load_dataset(dataInit, target):
    # 读入数据拆分训练集和测试集
    if target == "RON":
        scalerM = joblib.load('scalerM_RON.pt')
    else:
        scalerM = joblib.load('scalerM_S.pt')
    dataInit = scalerM.transform(dataInit)  # 返回值是ndarray类型
    dataInit = torch.from_numpy(dataInit).float()

    return dataInit

# Multilayer Perception
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, dropout=1):  # 特征值数量、隐藏层节点数量([]),类别数量
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size[0], bias=True)
        # self.fc3 = nn.Linear(hidden_size[0], hidden_size[1], bias=True)
        self.final = nn.Linear(hidden_size[0], num_classes, bias=True)  # output layer

    def forward(self, x_in, apply_sigmoid=False):  # 网络的输出未经过softmax运算
        # 定义前向之后，反向自动实现
        x = torch.relu(self.fc1(x_in))
        # x = torch.tanh(self.fc3(x))
        y_pred = self.final(x)
        return y_pred


def train_model(dataInit, target = "RON"):
    # Load Dataset and Return the Tensor
    # Device configuration
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # Model configuration
    [m_train, n_train] = dataInit.shape
    input_size = n_train
    hidden_size = [9, 0]
    # Train configuration

    model = MLP(input_size=input_size,
                hidden_size=hidden_size,
                num_classes=1)

    # initNetParams(model)
    model = model.to(device)
    if target == "RON":
        dataInit = load_dataset(dataInit, target=target)
        x_train = dataInit.to(device)
        # Load the model
        best_model_wts = torch.load("RONModel_RON.pt")  # best_model_wts是前面所保存下来的最好的模型参数in test set（字典参数）
    else:
        dataInit = load_dataset(dataInit, target=target)
        x_train = dataInit.to(device)
        # Load the model
        best_model_wts = torch.load("RONModel_S.pt")  # best_model_wts是前面所保存下来的最好的模型参数in test set（字典参数）
    model.load_state_dict(best_model_wts)
    model.eval()
    #print(x_train.shape)
    pre_train = model(x_train).cpu().detach().numpy().reshape(1, -1)[0]

    return pre_train



if __name__ == '__main__':
    dataInit = pd.read_csv("data_test.csv").values
    data_back = np.array([[188, 90.6, 53.23, 24.4, 22.37, 2.32, 7.3, 1.84, 5.98, 2.27700000e-01, 5.21000918e+00, 5.27178002e+01, 4.25009823e+02, 4.12256000e+02]])
    #pre_train = train_model(dataInit=dataInit, target = "S")  # target=RON or S
    pre_train = train_model(data_back, target="RON")  # target=RON or S
    print(pre_train)
