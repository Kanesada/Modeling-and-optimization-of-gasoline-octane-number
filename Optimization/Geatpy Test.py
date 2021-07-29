"""demo.py"""
import numpy as np
import geatpy as ea  # 导入geatpy库
import matplotlib.pyplot as plt
import time

"""============================目标函数============================"""


def aim(x):  # 传入种群染色体矩阵解码后的基因表现型矩阵
    return x * np.sin(10 * np.pi * x) + 2.0


x = np.linspace(-1, 2, 200)
plt.plot(x, aim(x))  # 绘制目标函数图像
"""============================变量设置============================"""
x1 = [-1, 2]  # 自变量范围
b1 = [1, 1]  # 自变量边界
varTypes = np.array([0])  # 自变量的类型，0表示连续，1表示离散
Encoding = 'BG'  # 'BG'表示采用二进制/格雷编码
codes = [1]  # 变量的编码方式，2个变量均使用格雷编码
precisions = [4]  # 变量的编码精度
scales = [0]  # 采用算术刻度
ranges = np.vstack([x1]).T  # 生成自变量的范围矩阵
borders = np.vstack([b1]).T  # 生成自变量的边界矩阵
"""=========================遗传算法参数设置========================="""
NIND = 40;  # 种群个体数目
MAXGEN = 25;  # 最大遗传代数
FieldD = ea.crtfld(Encoding, varTypes, ranges, borders, precisions, codes, scales)  # 调用函数创建区域描述器
Lind = int(np.sum(FieldD[0, :]))  # 计算编码后的染色体长度
obj_trace = np.zeros((MAXGEN, 2))  # 定义目标函数值记录器
var_trace = np.zeros((MAXGEN, Lind))  # 定义染色体记录器，记录每一代最优个体的染色体
"""=========================开始遗传算法进化========================"""
start_time = time.time()  # 开始计时
Chrom = ea.crtbp(NIND, Lind)  # 生成种群染色体矩阵
variable = ea.bs2real(Chrom, FieldD)  # 对初始种群进行解码
ObjV = aim(variable)  # 计算初始种群个体的目标函数值
best_ind = np.argmax(ObjV)  # 计算当代最优个体的序号
# 开始进化
for gen in range(MAXGEN):
    FitnV = ea.ranking(-ObjV)  # 根据目标函数大小分配适应度值(由于遵循目标最小化约定，因此最大化问题要对目标函数值乘上-1)
    SelCh = Chrom[ea.selecting('rws', FitnV, NIND - 1), :]  # 选择，采用'rws'轮盘赌选择
    SelCh = ea.recombin('xovsp', SelCh, 0.7)  # 重组(采用两点交叉方式，交叉概率为0.7)
    SelCh = ea.mutbin(Encoding, SelCh)  # 二进制种群变异
    # 把父代精英个体与子代合并
    Chrom = np.vstack([Chrom[best_ind, :], SelCh])
    variable = ea.bs2real(Chrom, FieldD)  # 对育种种群进行解码(二进制转十进制)
    ObjV = aim(variable)  # 求育种个体的目标函数值
    # 记录
    best_ind = np.argmax(ObjV)  # 计算当代最优个体的序号
    obj_trace[gen, 0] = np.sum(ObjV) / NIND  # 记录当代种群的目标函数均值
    obj_trace[gen, 1] = ObjV[best_ind]  # 记录当代种群最优个体目标函数值
    var_trace[gen, :] = Chrom[best_ind, :]  # 记录当代种群最优个体的变量值
# 进化完成
end_time = time.time()  # 结束计时
"""============================输出结果及绘图================================"""
best_gen = np.argmax(obj_trace[:, [1]])
print('目标函数最大值：', obj_trace[best_gen, 1])  # 输出目标函数最大值
variable = ea.bs2real(var_trace[[best_gen], :], FieldD)  # 解码得到表现型
print('对应的决策变量值为：')
print(variable[0][0])  # 因为此处variable是一个矩阵，因此用[0][0]来取出里面的元素
print('用时：', end_time - start_time)
plt.plot(variable, aim(variable), 'bo')
ea.trcplot(obj_trace, [['种群个体平均目标函数值', '种群最优个体目标函数值']])