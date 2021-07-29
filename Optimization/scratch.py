import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

def Read_ratio():
    p = r'Solution/编号为8的解集.csv'
    with open(p, encoding='utf-8-sig') as f:
        data = np.loadtxt(f, delimiter=",", skiprows=0)
        ratio = data[:, 0]
        print(ratio.shape)
        x = np.arange(0, ratio.shape[0])
        plt.title('Reduction rate of Pareto')
        plt.xlabel('Index')
        plt.ylabel('Rate')
        plt.plot(x, ratio)
        plt.show()

def Read_ratio_set():
    p = r'Solution/Ratio Set'
    with open(p, encoding='utf-8-sig') as f:
        ratio_set = np.loadtxt(f, delimiter=",", skiprows=0)
        #print(ratio_set)
        n = 0
        for i in ratio_set:
            if ratio_set[n] < 0:
                ratio_set[n] = ratio_set[n-1]
            n += 1
        x = np.arange(0, ratio_set.shape[0])
        plt.title('Reduction rate of Pareto')
        plt.xlabel('Index of sample in 325')
        plt.ylabel('Rate')
        plt.plot(x, ratio_set, 'ob', markersize=1)
        plt.axhline(0.3, 0, 324, color="red")  # 横线
        #plt.plot(x, ratio_set)
        plt.show()

def Read_ratio_pareto():
    p = r'Solution/编号为0的解集.csv'
    with open(p, encoding='utf-8-sig') as f:
        data = np.loadtxt(f, delimiter=",", skiprows=0)
        data = data[np.argsort(data[:, 0])]
        #print(data)
        plt.title('Reduction rate of the Sample 1 Pareto Solution Set')
        plt.xlabel('Reduction rate')
        plt.ylabel('S')
        #plt.plot(data[:, 0], data[:, 2], 'ob', markersize=1)
        plt.plot(data[:, 0], data[:, 2])
        plt.show()

#Read_ratio()
#Read_ratio_set()
Read_ratio_pareto()
