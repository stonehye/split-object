import cv2
import numpy as np
import os
from skimage import filters

from utils.normalization import min_max_normalize
from utils.filters import *

# parameter
var_temp = 1 # variation 계산 윈도우 사이즈
cvx_temp = 2


def extract_varContour(depth, mask, filename=None):
    # depth variation 계산
    # depth = min_max_normalize(depth, mask)  # depth normalization (0 ~ 255)
    var = var_filter(depth, mask, var_temp)

    var[np.where(var >= np.mean(var))] = 255
    var[np.where(var <= np.mean(var))] = 0

    # hysteresis thresholding
    high_thr = np.max(var[np.where(var != 0)]) * 0.8
    low_thr = np.mean(var[np.where(var != 0)])
    # print(high_thr, low_thr)
    contour = hysteresis_thresholding(var, high_thr, low_thr)

    return contour


def extract_maskContour(mask, thick=2):
    extracted_contour, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    mask_contour = np.full(mask.shape[:2], 0)
    for cnt in extracted_contour:
        cv2.drawContours(mask_contour, [cnt], -1, (255, 0, 0), thick)

    return mask_contour


def extract_convexityContour(contour, mask):
    # contour = median_filter(convexity_map, mask, ?cvx_temp, INMASK=False)

    high_thr = np.max(contour[np.where(contour != 0)]) * 0.9
    low_thr = np.max(contour[np.where(contour != 0)]) * 0.8
    contour = hysteresis_thresholding(contour, high_thr, low_thr)

    contour = median_filter(contour, mask, cvx_temp, INMASK=False)

    return contour


def thinning_contour(contour):
    return cv2.ximgproc.thinning(contour.astype(np.uint8))


def extract_labelimg_contour(labelimg):
    labelimg[np.where(labelimg != 0)] = 1
    labelimg = np.uint8(labelimg)

    kernel = np.uint8([[1,  1, 1],
                       [1, 10, 1],
                       [1,  1, 1]])
    filtered = cv2.filter2D(labelimg, -1, kernel)

    filtered[np.where((filtered < 10) & (filtered > 0))] = 255
    filtered[np.where(filtered != 255)] = 0

    return filtered


def find_connected_labels(num, contour, labeled_img):
    except_labels = np.array([0, 1, num])
    contour_points = np.vstack(np.where(contour == 255)[:2]).T

    neighbor_labels = np.array([])
    for point in contour_points:
        y, x = point
        y_start = y - 1
        y_end = y + 2
        x_start = x - 1
        x_end = x + 2

        neighbor_labels = np.concatenate((neighbor_labels, labeled_img[y_start:y_end, x_start:x_end].flatten()))

    neighbor_labels = np.unique(neighbor_labels)
    neighbor_labels = np.setdiff1d(neighbor_labels, except_labels)

    return neighbor_labels.astype(np.uint8)
