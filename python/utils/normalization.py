import numpy as np


def normalize(l):
    return l/(np.linalg.norm(l) + 1e-15)


def z_score_normalize(mat, mask):
    h, w = mat.shape[:2]
    MEAN = np.mean(mat[np.where(mask!=0)])
    STD = np.std(mat[np.where(mask!=0)])

    for y in range(0, h):
        for x in range(0, w):
            if mask[y][x] == 0.0:
                mat[y][x] = 0
            else:
                mat[y][x] = (mat[y][x] - MEAN) / STD * 255
    
    return mat


def min_max_normalize(mat, mask):
    h, w = mat.shape[:2]
    MAX = np.max(mat[np.where(mask!=0)])
    MIN = np.min(mat[np.where(mask!=0)])
    
    for y in range(0, h):
        for x in range(0, w):
            if mask[y][x] == 0:
                continue

            mat[y][x] = int((MAX - mat[y][x]) / (MAX - MIN) * 255)

    return mat


def min_max_normalize_(mat, mask):
    h, w = mat.shape[:2]
    MAX = np.max(mat[np.where(mask != 0)])
    MIN = np.min(mat[np.where(mask != 0)])

    for y in range(0, h):
        for x in range(0, w):
            if mask[y][x] == 0:
                continue

            mat[y][x] = int((MAX - mat[y][x]) / (MAX - MIN) * 600)

    return mat


def min_max_normalize_without_mask(mat):
    h, w = mat.shape[:2]
    MAX = np.max(mat)
    MIN = np.min(mat)

    result = np.zeros((h,w), np.float64)

    for y in range(0, h):
        for x in range(0, w):
            result[y][x] = int((MAX-mat[y][x])/(MAX-MIN) * 255)

    return result