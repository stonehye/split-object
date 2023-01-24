import numpy as np
import cv2
from utils.normalization import *


def find_center(num, labeled_img): # TODO: get_bbox_with_mask 모듈과 합치기
    left_point = np.min(np.where(labeled_img==num)[0])
    right_point = np.max(np.where(labeled_img==num)[0])
    up_point = np.min(np.where(labeled_img==num)[1])
    bottom_point = np.max(np.where(labeled_img==num)[1])

    center = [(left_point+right_point) // 2, (up_point+bottom_point) // 2]

    return np.array(center)


def find_3Dcenter(num, labeled_img, depth):
    left_point = np.min(np.where(labeled_img==num)[0])
    right_point = np.max(np.where(labeled_img==num)[0])
    up_point = np.min(np.where(labeled_img==num)[1])
    bottom_point = np.max(np.where(labeled_img==num)[1])

    depth_avg = np.mean(depth[np.where(labeled_img == num)]) / 10

    center = [
        (left_point+right_point) // 2,
        (up_point+bottom_point) // 2,
        # depth[(left_point+right_point) // 2][(up_point+bottom_point) // 2] / 10
        depth_avg
    ]

    return np.array(center)


def region_merging(labeled_img, mask, thr):
    h, w = labeled_img.shape[:2]
    label_list = np.unique(labeled_img) # label num 목록

    small_regions = []
    result_regions = []

    for num in label_list[2:]: # 배경 및 contour 영역 제외
        temp_mask = np.zeros((h, w))
        temp_mask[np.where(labeled_img == num)] = 255
        temp_center = find_center(num, labeled_img)

        # TODO: VisibleDeprecationWarning 해결
        if (len(np.where(labeled_img == num)[0]) / len(np.where(mask != 0)[0]) < thr):
            small_regions.append(np.array([temp_mask, temp_center]))
        else: 
            result_regions.append(np.array([temp_mask, temp_center]))

    # print(f'small region 개수: {len(small_regions)}')
    
    if result_regions:
        for item in small_regions:
            dist_list = [np.linalg.norm(center - item[1]) for (mask, center) in result_regions]
            idx = np.argmin(dist_list) # center 간 dist가 제일 적은 region 선택
            result_regions[idx][0] += item[0]
    else:
        result_regions = small_regions
    
    return result_regions


def calculate_area(num, labeled_img):
    return len(np.where(labeled_img==num)[0])


def is_small(thr, num, labeled_img, mask):
    return calculate_area(num, labeled_img) / len(np.where(mask != 0)[0]) <= thr


def getOrientation(pts):
    sz = len(pts)
    data_pts = np.empty((sz, 2), dtype=np.float64)
    for i in range(data_pts.shape[0]):
        data_pts[i, 0] = pts[i, 0, 0]
        data_pts[i, 1] = pts[i, 0, 1]
    mean = np.empty((0))
    mean, eigenvectors, eigenvalues = cv2.PCACompute2(data_pts, mean)
    center = (int(mean[0, 0]), int(mean[0, 1]))

    # print(f'mean: {mean}')
    # print(f'eigenvectors: {eigenvectors}')
    # print(f'eigenvalues: {eigenvalues}')
    # print('-'*40)

    return center