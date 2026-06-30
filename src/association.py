import numpy as np
from scipy.optimize import linear_sum_assignment


def buildCostMatrix(trackers, detections):
    cost = np.zeros((len(trackers),len(detections)))  # 2x3 sıfır matrisi oluşturmuş oluyor

    for i, t in enumerate(trackers):
        for j , d in enumerate(detections):
            cost[i, j] = np.sqrt((t[0]-d[0])**2 + (t[1]-d[1])**2)
    
    return cost

def associate(trackers, detections, max_distance = 50):

    if len(trackers) == 0 or len(detections) == 0 :
        return [] , list(range(len(detections)))
    
    cost = buildCostMatrix(trackers, detections)
    row_ind , col_ind = linear_sum_assignment(cost)

    matches = []

    unmatched_detections = []

    for r, c in zip(row_ind, col_ind):
        if cost[r,c] > max_distance:
            unmatched_detections.append(c)
        else:
            matches.append((r,c))
    
    for j in range(len(detections)):
        if j not in col_ind:
            unmatched_detections.append(j)
    
    return matches, unmatched_detections