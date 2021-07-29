from gmdhpy.gmdh import Regressor
from gmdhpy.plot_model import PlotModel
import pandas as pd
import os
import xgboost as xgb
from sklearn import preprocessing
import numpy as np
from xgboost import plot_importance
from xgboost import XGBClassifier
from xgboost import XGBRegressor


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
import h5py

model = load_model("model_pred.h5")
data, label = load_data(<the path of the data>)
predict = model.predict(data)