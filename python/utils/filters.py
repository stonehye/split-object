import cv2
import numpy as np
from skimage import filters


def hysteresis_thresholding(mat, high_thr, low_thr):
    high_edge = np.zeros(mat.shape[:2])
    high_edge[np.where(mat >= high_thr)] = 255
    hyst = filters.apply_hysteresis_threshold(mat, low_thr, high_thr)
    high_edge[np.where(hyst == True)] = 255

    return high_edge


def var_filter(mat, mask, term, INMASK = True):
    h, w = mat.shape[:2]
    var = np.zeros((h, w))
    zeros = np.array([0])

    for y in range(0, h):
        for x in range(0, w):
            if INMASK and mask[y][x] == 0:
                continue

            y_start = y - term if y - term >= 0 else 0
            y_end = y + term + 1 if y + term + 1 <= h else h
            x_start = x - term if x - term >= 0 else 0
            x_end = x + term + 1 if x + term + 1 <= w else w

            temp_mat = mat[y_start:y_end, x_start:x_end].flatten()
            temp_mat = np.setdiff1d(temp_mat, zeros) # mask 범위 외 영역은 계산 범위에 포함하지 않음.

            var[y][x] = np.var(temp_mat)

    var[np.where(var >= 255)] = 255
    var[np.where(var < 0)] = 0

    return var


def median_filter(mat, mask, term, INMASK = True):
    h, w = mat.shape[:2]
    med = np.zeros((h, w))

    for y in range(0, h):
        for x in range(0, w):
            if INMASK and mask[y][x] == 0:
                continue

            y_start = y - term if y - term >= 0 else 0
            y_end = y + term + 1 if y + term + 1 <= h else h
            x_start = x - term if x - term >= 0 else 0
            x_end = x + term + 1 if x + term + 1 <= w else w

            temp_mat = mat[y_start:y_end, x_start:x_end].flatten()
            med[y][x] = np.median(temp_mat)

    med[np.where(med >= 255)] = 255
    med[np.where(med < 0)] = 0

    return med


def average_filter(mat, mask, term, INMASK = True):
    h, w = mat.shape[:2]
    avg = np.zeros((h, w))

    for y in range(0, h):
        for x in range(0, w):
            if INMASK and mask[y][x] == 0:
                continue

            y_start = y - term if y - term >= 0 else 0
            y_end = y + term + 1 if y + term + 1 <= h else h
            x_start = x - term if x - term >= 0 else 0
            x_end = x + term + 1 if x + term + 1 <= w else w

            temp_mat = mat[y_start:y_end, x_start:x_end].flatten()
            avg[y][x] = np.average(temp_mat)

    avg[np.where(avg >= 255)] = 255
    avg[np.where(avg < 0)] = 0

    return avg


def max_filter(mat, term):
    h, w = mat.shape[:2]
    max = np.zeros((h, w))

    for y in range(0, h):
        for x in range(0, w):
            y_start = y - term if y - term >= 0 else 0
            y_end = y + term + 1 if y + term + 1 <= h else h
            x_start = x - term if x - term >= 0 else 0
            x_end = x + term + 1 if x + term + 1 <= w else w

            temp_mat = mat[y_start:y_end, x_start:x_end].flatten()
            max[y][x] = np.max(temp_mat)

    return max


def min_filter(mat, term):
    h, w = mat.shape[:2]
    min = np.zeros((h, w))

    for y in range(0, h):
        for x in range(0, w):
            y_start = y - term if y - term >= 0 else 0
            y_end = y + term + 1 if y + term + 1 <= h else h
            x_start = x - term if x - term >= 0 else 0
            x_end = x + term + 1 if x + term + 1 <= w else w

            temp_mat = mat[y_start:y_end, x_start:x_end].flatten()
            min[y][x] = np.min(temp_mat)

    return min


def morphology_closing(mat, kernel_size):
    kernel = np.ones(kernel_size, np.uint8)
    result = cv2.morphologyEx(mat, cv2.MORPH_CLOSE, kernel)

    return result


def Laplacian_filter(gray):
    laplacian_mask1 = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
    laplacian_mask2 = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]])
    laplacian_mask3 = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])

    laplacian1 = cv2.filter2D(gray, -1, laplacian_mask1)
    laplacian2 = cv2.filter2D(gray, -1, laplacian_mask2)
    laplacian3 = cv2.filter2D(gray, -1, laplacian_mask3)
    laplacian4 = cv2.Laplacian(gray, -1)

    return laplacian4


def LoG_filter(gray):
    laplacian_mask1 = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])

    gaussian = cv2.GaussianBlur(gray, (0, 0), 1)
    LoG = cv2.filter2D(gaussian, -1, laplacian_mask1)

    return LoG


def DoG_filter(gray):
    height, width = gray.shape[:2]

    gaussian1 = cv2.GaussianBlur(gray, (5, 5), 10)
    gaussian2 = cv2.GaussianBlur(gray, (5, 5), 1)

    DoG = np.zeros_like(gray)
    for i in range(height):
        for j in range(width):
            DoG[i][j] = float(gaussian1[i][j]) - float(gaussian2[i][j])

    return DoG


def Sobel_filter(gray):
    dx = cv2.Sobel(gray, -1, 1, 0, ksize=3, delta=128)
    dy = cv2.Sobel(gray, -1, 0, 1, ksize=3, delta=128)

    return dx, dy


def Scharr_filter(gray):
    dx = cv2.Scharr(gray, -1, 1, 0, delta=128)
    dy = cv2.Scharr(gray, -1, 0, 1, delta=128)

    return dx, dy


def sharpening(gray):
    sharpening_mask1 = np.array([[-1, -1, -1], [-1, 12, -1], [-1, -1, -1]])
    sharpening_mask2 = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpening_out1 = cv2.filter2D(gray, -1, sharpening_mask1)
    sharpening_out2 = cv2.filter2D(gray, -1, sharpening_mask2)

    return sharpening_out1