import cv2
import numpy as np
import os
import math


def CIEDE2000_distance(Lab1, Lab2):
    Lab1_avg = np.mean(Lab1, axis = 0)
    Lab2_avg = np.mean(Lab2, axis = 0)
    diff = math.sqrt(math.pow(Lab1_avg[0]-Lab2_avg[0], 2) + pow(Lab1_avg[1]-Lab2_avg[1], 2) + pow(Lab1_avg[2]-Lab2_avg[2], 2))

    normalized_diff = diff / 376
    return normalized_diff


def depthAvg_distance(depth1, depth2):
    depth1_avg = np.mean(depth1)
    depth2_avg = np.mean(depth2)
    diff = abs(depth1_avg-depth2_avg)

    normalized_diff = diff / 5000
    return normalized_diff


def find_center(labels, label_list):
    result = []
    for idx, num in enumerate(label_list):
        left_point = np.min(np.where(labels==num)[0])
        right_point = np.max(np.where(labels==num)[0])
        up_point = np.min(np.where(labels==num)[1])
        bottom_point = np.max(np.where(labels==num)[1])
        center = [(left_point+right_point) // 2, (up_point+bottom_point) // 2]

        result.append([num, np.array(center)])
    return np.array(result)


def find_nearest(idx, label_list, k=3):
    result = [[label[0], np.linalg.norm(label[1] - label_list[idx][1])] for label in label_list]
    result.sort(key=lambda x:x[1])
    return np.array(result[1:k])


def Clustering(rgb, depth, labels):
    mat_list = []
    LAB = cv2.cvtColor(rgb, cv2.COLOR_BGR2Lab)
    h, w = labels.shape[:2]
    label_list = find_center(labels, np.unique(labels)[2:])
    modified_label_list = np.column_stack((np.unique(labels)[2:], np.full((len(np.unique(labels)[2:])), -1)))
    
    global_idx = 0
    for idx, (num, center) in enumerate(modified_label_list):
        if (modified_label_list[idx][1] == -1):
            modified_label_list[idx][1] = global_idx
            global_idx += 1
            nearest_list = find_nearest(idx, label_list)
            temp_lab1 = LAB[np.where(labels == num)]
            temp_depth1 = depth[np.where(labels == num)]
            temp_label1 = labels[np.where(labels == num)]

            for [num2, dist] in nearest_list:
                temp_lab2 = LAB[np.where(labels == num2)]
                temp_depth2 = depth[np.where(labels == num2)]
                temp_label2 = labels[np.where(labels == num2)]

                color_dist = CIEDE2000_distance(temp_lab1, temp_lab2)
                depth_dist = depthAvg_distance(temp_depth1, temp_depth2)
                geometric_dist = dist

                print(color_dist, depth_dist, geometric_dist)

    return mat_list
    