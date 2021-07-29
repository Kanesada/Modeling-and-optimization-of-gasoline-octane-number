# -*- coding: utf-8 -*-
import numpy as np
import geatpy as ea
from ann_model_pre import train_model
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

def Run_net_1(X):    # Predict the RON
    RON = train_model(X, 'RON')
    return RON

def Run_net_2(X):   # Predict the S
    S = train_model(X, 'S')
    return S


def Read_range():
    p = r'ublb.csv'
    with open(p, encoding='utf-8-sig') as f:
        data = np.loadtxt(f, delimiter=",", skiprows=0)
        Ub = data[:, 0]
        Lb = data[:, 1]
        return Ub, Lb

def Read_a(index):
    p = r'325select.csv'
    with open(p, encoding="utf-8-sig") as f:
        data = np.loadtxt(f, delimiter=",", skiprows=0)
        a = data[index]
        return a

def Read_label(index):
    p = r'325label.csv'
    with open(p, encoding="utf-8-sig") as f:
        data = np.loadtxt(f, delimiter=",", skiprows=0)
        Ron_Loss = data[index, 0]
        S = data[index, 1]
        return Ron_Loss, S

class RON_Optimization(ea.Problem):
    def __init__(self):
        name = '优化RON与S值'
        M = 2 # 初始化M（目标维数）
        maxormins = [1] * M # 初始化maxormins
        Dim = 5 # 初始化Dim（决策变量维数）
        varTypes = [0] * Dim # 初始化varTypes
        ub, lb = Read_range()
        lbin = [0] * Dim # 决策变量下边界
        ubin = [0] * Dim # 决策变量上边界
        # 调用父类构造方法完成实例化
        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)
    def aimFunc(self, pop): # 目标函数
        Vars = pop.Phen # 得到决策变量矩阵
        A = np.tile(a, (NIND, 1))
        X = np.hstack((A, Vars))
        ObjV1 = Run_net_1(X)   # 调用神经网络预测模型 计算辛烷值损失
        ObjV1 = ObjV1.reshape(-1, 1)
        ObjV2 = Run_net_2(X)   # 调用神经网络预测模型 计算硫含量
        ObjV2 = ObjV2.reshape(-1, 1)
        pop.ObjV = np.hstack([ObjV1, ObjV2]) # 把结果赋值给ObjV
        # 采用可行性法则处理约束
        pop.CV = np.hstack([-1*ObjV1, ObjV2 - 4.99, -1*ObjV2])



def Run_Solver():
    """================================实例化问题对象============================="""
    problem = RON_Optimization()          # 生成问题对象
    """==================================种群设置================================"""
    Encoding = 'RI'           # 编码方式
    Field = ea.crtfld(Encoding, problem.varTypes, problem.ranges, problem.borders) # 创建区域描述器
    population = ea.Population(Encoding, Field, NIND) # 实例化种群对象
    """================================算法参数设置==============================="""
    myAlgorithm = ea.moea_NSGA2_templet(problem, population) # 实例化一个算法模板对象
    myAlgorithm.MAXGEN = 400  # 最大进化代数
    myAlgorithm.drawing = 0
    NDSet = myAlgorithm.run() # 执行算法模板，得到非支配种群
    print('用时：%f 秒'%(myAlgorithm.passTime))
    Solution = np.hstack((NDSet.ObjV, NDSet.Phen))
    Solution = Solution[np.argsort(Solution[:, 0])]
    before, _ = Read_label(index)
    after = Solution[:, 0]
    ratio = (before - after) / before
    Solution = np.hstack((ratio.reshape(-1, 1), Solution))
    global ratio_set#, Operator
    ratio_set = np.append(ratio_set, Solution[0, 0])
    #Operator = np.append((Operator, Solution[0, :]))
    if Solution[0, 0] < 0.3:
        print(index)
        global failed
        failed = np.append(failed, index)
    solution_name = '编号为' + str(index) + '的解集.csv'
    np.savetxt('Solution/' + solution_name, Solution, fmt='%f', delimiter=',')

if __name__ == "__main__":
    global a, NIND, index
    n_failed = 0
    index = 0
    failed = np.array([])
    ratio_set = np.array([])
    #Operator = np.array([])
    for i in range(0, 324):
        a = Read_a(index)
        NIND = 400                # 种群规模
        Run_Solver()
        index += 1
    print(failed)
    np.savetxt('Solution/' + 'Failed index', failed, fmt='%f', delimiter=',')
    np.savetxt('Solution/' + 'Ratio Set', ratio_set, fmt='%f', delimiter=',')
    #np.savetxt('Solution/' + 'Operator', Operator, fmt='%f', delimiter=',')

    x = np.arange(0, ratio_set.shape[0])
    plt.title('Reduction rate of Pareto')
    plt.xlabel('Index of Sample in 325')
    plt.ylabel('Rate')
    plt.plot(x, ratio_set)
    plt.show()



    '''index = 0
    a = Read_a(index)
    NIND = 400                # 种群规模
    Run_Solver()
    index += 1'''
