# -*- coding: utf-8 -*-
import numpy as np
import geatpy as ea
from ann_model_pre import train_model
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
        # print(data)
        Ub = data[:, 0]
        Lb = data[:, 1]
        return Ub, Lb

def Read_a(index):
    p = r'325select.csv'
    with open(p, encoding="utf-8-sig") as f:
        data = np.loadtxt(f, delimiter=",", skiprows=0)
        #print(data)
        a = data[index]
        #print(a)
        return a

def Read_label(index):
    p = r'325label.csv'
    with open(p, encoding="utf-8-sig") as f:
        data = np.loadtxt(f, delimiter=",", skiprows=0)
        #print(data)
        Ron_Loss = data[index, 0]
        S = data[index, 1]
        #print(a)
        return Ron_Loss, S

class RON_Optimization(ea.Problem): # 继承Problem父类
    def __init__(self):
        name = '优化RON与S值' # 初始化name（函数名称，可以随意设置）
        M = 2 # 初始化M（目标维数）
        maxormins = [1] * M # 初始化maxormins（目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标）
        Dim = 5 # 初始化Dim（决策变量维数）
        varTypes = [0] * Dim # 初始化varTypes（决策变量的类型，0：实数；1：整数）
        ub, lb = Read_range()
        lbin = [0] * Dim # 决策变量下边界（0表示不包含该变量的下边界，1表示包含）
        ubin = [0] * Dim # 决策变量上边界（0表示不包含该变量的上边界，1表示包含）
        # 调用父类构造方法完成实例化
        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)
    def aimFunc(self, pop): # 目标函数
        Vars = pop.Phen # 得到决策变量矩阵
        A = np.tile(a, (NIND, 1))
        X = np.hstack((A, Vars))
        ObjV1 = Run_net_1(X)
        ObjV1 = ObjV1.reshape(-1, 1)
        #print(type(ObjV1))
        ObjV2 = Run_net_2(X)
        ObjV2 = ObjV2.reshape(-1, 1)
        pop.ObjV = np.hstack([ObjV1, ObjV2]) # 把结果赋值给ObjV
        # 采用可行性法则处理约束
        pop.CV = np.hstack([-1*ObjV1, ObjV2 - 4.99, -1*ObjV2])
        # print('ObjV的ndim为： '+ str(pop.ObjV.ndim))
        # print('ObjV的数据类型为： ' + str(type(pop.ObjV)))
        # print('ObjV的shape[0]为： ' + str(pop.ObjV.shape[0]))
        # print('ObjV的shape[1]为： ' + str(pop.ObjV.shape[1]))


def Run_Solver():
    """================================实例化问题对象============================="""
    problem = RON_Optimization()          # 生成问题对象
    """==================================种群设置================================"""
    Encoding = 'RI'           # 编码方式
    Field = ea.crtfld(Encoding, problem.varTypes, problem.ranges, problem.borders) # 创建区域描述器
    population = ea.Population(Encoding, Field, NIND) # 实例化种群对象（此时种群还没被初始化，仅仅是完成种群对象的实例化）
    """================================算法参数设置==============================="""
    myAlgorithm = ea.moea_NSGA2_templet(problem, population) # 实例化一个算法模板对象
    myAlgorithm.MAXGEN = 300  # 最大进化代数
    myAlgorithm.drawing = 1   # 设置绘图方式（0：不绘图；1：绘制结果图；2：绘制过程动画）
    """===========================调用算法模板进行种群进化===========================
    调用run执行算法模板，得到帕累托最优解集NDSet。NDSet是一个种群类Population的对象。
    NDSet.ObjV为最优解个体的目标函数值；NDSet.Phen为对应的决策变量值。
    详见Population.py中关于种群类的定义。
    """
    NDSet = myAlgorithm.run() # 执行算法模板，得到非支配种群
    print('用时：%f 秒'%(myAlgorithm.passTime))
    # print(NDSet.Phen.shape)
    # print(NDSet.ObjV.shape)
    Solution = np.hstack((NDSet.ObjV, NDSet.Phen))
    #print(Solution)
    Solution = Solution[np.argsort(Solution[:, 0])]
    # for i in Solution:
    #     before = a[1]
    #     after = i[0]
    #     ratio = (before - after) / before
    before, _ = Read_label(index)
    after = Solution[:, 0]
    ratio = (before - after) / before
    Solution = np.hstack((ratio.reshape(-1, 1), Solution))
    solution_name = '编号为' + str(index) + '的解集.csv'
    b = np.savetxt(solution_name, Solution, fmt='%f', delimiter=',')

if __name__ == "__main__":
    global a, NIND, index
    #a = np.array([188, 90.6, 53.23, 24.4, 22.37, 2.32, 7.3, 1.84, 5.98])
    index = 8
    a = Read_a(index)
    NIND = 400                # 种群规模
    Run_Solver()

