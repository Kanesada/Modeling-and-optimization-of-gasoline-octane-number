# -*- coding: utf-8 -*-
import numpy as np
import geatpy as ea
'''
def Run_net_1(X, A):
    #Load ANN 1

    return RON

def Run_net_2(X, A)
    #Load ANN 2

    return S
'''


class ZDT1(ea.Problem): # 继承Problem父类
    def __init__(self):
        name = 'ZDT1' # 初始化name（函数名称，可以随意设置）
        M = 2 # 初始化M（目标维数）
        maxormins = [1] * M # 初始化maxormins（目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标）
        Dim = 30 # 初始化Dim（决策变量维数）
        varTypes = [0] * Dim # 初始化varTypes（决策变量的类型，0：实数；1：整数）
        lb = [0] * Dim # 决策变量下界
        ub = [1] * Dim # 决策变量上界
        lbin = [1] * Dim # 决策变量下边界（0表示不包含该变量的下边界，1表示包含）
        ubin = [1] * Dim # 决策变量上边界（0表示不包含该变量的上边界，1表示包含）
        # 调用父类构造方法完成实例化
        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)
    def aimFunc(self, pop): # 目标函数
        Vars = pop.Phen # 得到决策变量矩阵
        ObjV1 = Vars[:, 0]
        print(type(ObjV1))
        print(ObjV1.shape)
        gx = 1 + 9 * np.sum(Vars[:, 1:], 1) / (self.Dim - 1)
        hx = 1 - np.sqrt(np.abs(ObjV1) / gx) # 取绝对值是为了避免浮点数精度异常带来的影响
        ObjV2 = gx * hx
        pop.ObjV = np.array([ObjV1, ObjV2]).T # 把结果赋值给ObjV

def Run_solver():
    """================================实例化问题对象============================="""
    problem = ZDT1()          # 生成问题对象
    """==================================种群设置================================"""
    Encoding = 'RI'           # 编码方式
    NIND = 40                 # 种群规模
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
    print(type(NDSet.Phen))
    print(NDSet.Phen.shape)

if __name__ == "__main__":
    Run_solver()