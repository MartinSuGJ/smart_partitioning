#！/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time     : 2018/7/15  15:13
# @Author   : Cardinal Operations
# @Site     : https://www.shanshu.ai
# @File     : ccp.py
# @Software : PyCharm

from gurobipy import *
import pandas as pd
import scipy.spatial as spatial
import numpy as np
import matplotlib.pyplot as plt
import random

# points = pd.read_excel('./data/capacitated_clustering_500.xlsx', sheet_name='point_information')
NUM_PTS = 500
points = pd.DataFrame({'id': range(1, NUM_PTS + 1),
                       'lng': [121 + 2 * random.random() for i in range(NUM_PTS)],
                       'lat': [31 + 2 * random.random() for i in range(NUM_PTS)],
                       'weight': [10 * random.random() for i in range(NUM_PTS)]})
dist_mat = spatial.distance_matrix(points[['lng', 'lat']].values, points[['lng', 'lat']].values)
weight = points['weight'].values

"""
进行模型构建
"""
m = Model('CCP')
# #########################
# 声明变量
# #########################
assign = m.addVars(range(dist_mat.shape[0]), range(dist_mat.shape[1]), vtype=GRB.BINARY, name='assignment')
center = m.addVars(range(dist_mat.shape[0]), vtype=GRB.BINARY, name='cluster_center')

# #########################
# 声明约束
# #########################
m.addConstr(
    quicksum(center) >= 1,
    'lower bound of number of cluster'
)
m.addConstr(
    quicksum(center) <= 100,
    'upper bound of number of cluster'
)

m.addConstrs(
    (quicksum([assign[i, j] for j in range(dist_mat.shape[0])]) == 1 for i in range(dist_mat.shape[0]))
)

m.addConstrs(
    (assign[i, j] <= center[j] for i in range(dist_mat.shape[0]) for j in range(dist_mat.shape[0]))
)

m.addConstrs(
    (quicksum([assign[i, j] * weight[i] for i in range(dist_mat.shape[0])]) <= 100 for j in range(dist_mat.shape[0]))
)

m.setObjective(
    quicksum([assign[i, j] * dist_mat[i, j] for i in range(dist_mat.shape[0]) for j in range(dist_mat.shape[1])])
    + 0.5 * quicksum([center[i] for i in range(dist_mat.shape[0])]), GRB.MINIMIZE)

m.optimize()

"""
提取分类结果
并进行可视化
"""
label = list()
for i in range(dist_mat.shape[0]):
    for j in range(dist_mat.shape[0]):
        if assign[i, j].x > 0.5:
            label.append(j)
            break

points['label'] = label
print('有{}点'.format(points.shape[0]))
print('分成{}类别'.format(len(np.unique(points['label']))))
plt.scatter(points['lng'], points['lat'], c=points['label'])
plt.show()
