import pandas as pd
import numpy as np
def Read_selected_variables():
    data_354 = pd.read_csv('354.csv')
    data_need = pd.read_csv('data_test.csv')
    data_need.shape[1]
    mylist=[]
    for i in range(0, data_need.shape[1]):
        #mylist.append([data_354.values == data_need.columns[i]].index)
        print([data_354.values == data_need.columns[i]].index)


def Read_range():
    p = r'delta Lower Upper .csv'
    with open(p, encoding='utf-8-sig') as f:
        data = np.loadtxt(f, delimiter=",", skiprows=0)
        print(data)
        Upper = data[:, 2]
        Lower = data[:, 1]

        #print(Lower)
        #print(Upper)


if __name__ == "__main__":
    Read_range()
