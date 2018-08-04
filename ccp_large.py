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
tree = spatial.cKDTree(points[['lng', 'lat']].values)
sparse_dist_mat = tree.sparse_distance_matrix(tree, 0.5)
weight = points['weight'].values


# #########################
# 进行模型构建
# #########################
m = Model('CCP')

"""
声明变量
"""
print('正在声明变量')
assign = m.addVars(range(sparse_dist_mat.shape[0]),
                   range(sparse_dist_mat.shape[1]),
                   vtype=GRB.BINARY, name='assignment')
center = m.addVars(range(sparse_dist_mat.shape[0]),
                   vtype=GRB.BINARY, name='cluster_center')


"""
声明约束
"""
print('正在声明约束')
m.addConstr(
    quicksum(center) >= 1,
    'lower bound of number of cluster'
)
m.addConstr(
    quicksum(center) <= 100,
    'upper bound of number of cluster'
)

m.addConstrs(
    (quicksum([assign[i,j] for _,j in sparse_dist_mat[i].keys()]) == 1 for i in range(sparse_dist_mat.shape[0])),
    'each point can only belong to one specific cluster'
)

m.addConstrs(
    (assign[i, j] <= center[j] for i in range(sparse_dist_mat.shape[0]) for j in range(sparse_dist_mat.shape[0]))
)

m.addConstrs(
    (quicksum([assign[i, j] * weight[i] for i in range(sparse_dist_mat.shape[0])]) <= 100 for j in range(sparse_dist_mat.shape[0]))
)

m.addConstrs(
    (assign[i,j]==0 for j in range(sparse_dist_mat.shape[0]) for i in range(sparse_dist_mat.shape[0]) if i not in list(zip(*list(sparse_dist_mat[j].keys())))[1])
)

sum_of_dist = quicksum([assign[i, j] * sparse_dist_mat[(i, j)] for j in range(sparse_dist_mat.shape[1]) for _,i in sparse_dist_mat[j].keys()])
num_of_cluster = quicksum([center[i] for i in range(sparse_dist_mat.shape[0])])
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
