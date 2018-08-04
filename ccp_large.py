#！/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time     : 2018/7/15  15:13
# @Author   : Cardinal Operations
# @Site     : https://www.shanshu.ai
# @File     : ccp_large.py
# @Software : PyCharm


from gurobipy import *
import pandas as pd
import scipy.spatial as spatial
import numpy as np
import matplotlib.pyplot as plt
import random
import os


# ###########################
# 设置工作环境路径
# ###########################
wkd = 'D:/Learning/OR/CapacitatedClustering'
os.chdir(wkd)


# ##########################
# 随机生成需求点数据
# 生成距离矩阵信息，只保留临近
# ##########################
# points = pd.read_excel('./data/capacitated_clustering_500.xlsx', sheet_name='point_information')
NUM_PTS = 1000
points = pd.DataFrame({'id': range(1, NUM_PTS + 1),
                       'lng': [121 + 2 * random.random() for i in range(NUM_PTS)],
                       'lat': [31 + 2 * random.random() for i in range(NUM_PTS)],
                       'weight': [10 * random.random() for i in range(NUM_PTS)]})

# 构建ckdtree，方便后面计算每个点与临近点的距离
# 当点超过1000的时候，计算1000*1000距离矩阵不现实
tree = spatial.cKDTree(points[['lng', 'lat']].values)
# sparse_distance_matrix返回距离在0.5以内的点与点之间的距离
sparse_dist_mat = tree.sparse_distance_matrix(tree, 0.5)
# 每个点/类都有一个weight，代表这个类的某种量
weight = points['weight'].values


# #########################
# 进行模型构建
# #########################
m = Model('CCP')

"""
声明变量
"""
print('正在声明变量')
# assign变量是分配变量，assign[i,j]=1代表i点属于以j为中心的类
assign = m.addVars(range(sparse_dist_mat.shape[0]),
                   range(sparse_dist_mat.shape[1]),
                   vtype=GRB.BINARY, name='assignment')
# center变量是类中心变量，center[j]=1代表j点为类中心
center = m.addVars(range(sparse_dist_mat.shape[0]),
                   vtype=GRB.BINARY, name='cluster_center')


"""
声明约束
"""
print('正在声明约束')
# 最少类别数目限制
m.addConstr(
    quicksum(center) >= 1,
    'lower bound of number of cluster'
)

# 最大类别数目限制
m.addConstr(
    quicksum(center) <= 100,
    'upper bound of number of cluster'
)

# 每一个点只能归属于一个类，不能同时归属于两个不同的类
m.addConstrs(
    (quicksum([assign[i,j] for _,j in sparse_dist_mat[i].keys()]) == 1 for i in range(sparse_dist_mat.shape[0])),
    'each point can only belong to one specific cluster'
)

# 保证只有当j点是类中心，才会有i点属于j点为中心的类
m.addConstrs(
    (assign[i, j] <= center[j] for i in range(sparse_dist_mat.shape[0]) for j in range(sparse_dist_mat.shape[0]))
)

# 每个类某种量的限制不能超过某个限制
m.addConstrs(
    (quicksum([assign[i, j] * weight[i] for i in range(sparse_dist_mat.shape[0])]) <= 100 for j in range(sparse_dist_mat.shape[0]))
)

# 添加约束，相隔远的点肯定不会归属于j类
m.addConstrs(
    (assign[i,j]==0 for j in range(sparse_dist_mat.shape[0]) for i in range(sparse_dist_mat.shape[0]) if i not in list(zip(*list(sparse_dist_mat[j].keys())))[1])
)

# 各点到所属类的总距离
sum_of_dist = quicksum([assign[i, j] * sparse_dist_mat[(i, j)] for j in range(sparse_dist_mat.shape[1]) for _,i in sparse_dist_mat[j].keys()])
# 类的数量作为惩罚，尽量减少类的数量
num_of_cluster = quicksum([center[i] for i in range(sparse_dist_mat.shape[0])])
# 权衡总距离与类数量
m.setObjective(
    sum_of_dist + 0.5 * num_of_cluster, GRB.MINIMIZE)

print('正在求解')
m.optimize()

"""
提取分类结果
并进行可视化
"""
label = list()
for i in range(sparse_dist_mat.shape[0]):
    for j in range(sparse_dist_mat.shape[0]):
        if assign[i, j].x > 0.5:
            label.append(j)
            break

points['label'] = label
print('有{}点'.format(points.shape[0]))
print('分成{}类别'.format(len(np.unique(points['label']))))
plt.scatter(points['lng'], points['lat'], c=points['label'])
plt.show()
