import numpy as np
import cv2

from utils.normalization import z_score_normalize, min_max_normalize_, normalize

term = 2 # normal 계산 윈도우 사이즈: (term * 2 + 1) x (term * 2 + 1)


def compute_convexity(n1, c1, n2, c2):
    c = c1 - c2
    c = c / (np.linalg.norm(c) + 1e-15)

    cos1 = np.dot(n1, c)
    cos2 = np.dot(n2, c)

    return (cos1 - cos2 + 2) / 4.0


def compute_normal(depth, save=False):
    h, w = depth.shape[:2]
    norm = np.zeros((h, w, 3))
    for y in range(0, h):
        for x in range(0, w):
            next_x = depth[y][x + term] if x + term < w else depth[y][x]
            prev_x = depth[y][x - term] if x - term >= 0 else depth[y][x]
            next_y = depth[y + term][x] if y + term < h else depth[y][x]
            prev_y = depth[y - term][x] if y - term >= 0 else depth[y][x]

            dzdx = (next_x - prev_x) / 2.0
            dzdy = (next_y - prev_y) / 2.0

            d = np.array([(-1) * dzdy, (-1) * dzdx, 1.0])
            n = normalize(d)

            norm[y][x] = n

    return norm


def compute_convexitymap(norm, mask, depth, window_ratio=0.1, INMASK=True):
    h, w = norm.shape[:2]

    # window_ratio에 따른 convexity 계산 범위 설정
    window_height = int(h * window_ratio)
    window_width = int(w * window_ratio)
    height_term = window_height // 2 if window_height // 2 >= 1 else 1
    width_term = window_width // 2 if window_width // 2 >= 1 else 1

    # print(f'height: {h}, weight: {w}, window height: {window_height}, window width: {window_width}, height term: {height_term}, width_term: {width_term}')

    convexity_map = np.zeros((h, w))
    for y in range(0, h):
        for x in range(0, w):
            if (mask[y][x] == 0):
                continue # 마스크 영역 내 픽셀들에 대해서만 계산

            y_start = y - height_term if y - height_term >= 0 else 0
            y_end = y + height_term + 1 if y + height_term + 1 <= h else h
            x_start = x - width_term if x - width_term >= 0 else 0
            x_end = x + width_term + 1 if x + width_term + 1 <= w else w

            # convexity 계산 영역에 해당되는 픽셀이 없을 경우
            if y_start == y_end: y_end += 1
            if x_start == x_end: x_end += 1

            neighbors = []
            for i in range(y_start, y_end):
                for j in range(x_start, x_end):
                    if INMASK and mask[i][j] == 0:
                        continue # INMASK 설정 시 마스크 내 픽셀들만 고려

                    if not (i == y and j == x):
                        coordinate1 = np.array([y, x, depth[y][x]])
                        coordinate2 = np.array([i, j, depth[i][j]])
                        neighbors.append(compute_convexity(norm[y][x], coordinate1, norm[i][j], coordinate2))

            # convexity_map[y][x] = max(neighbors) if neighbors else 0 # maximun
            convexity_map[y][x] = sum(neighbors) / len(neighbors) # average

    # 계산된 convexity map normalization
    convexity_map = z_score_normalize(convexity_map, mask)
    # convexity_map = min_max_normalize(convexity_map, mask)

    convexity_map[np.where(convexity_map >= 255)] = 255
    convexity_map[np.where(convexity_map <= 0)] = 0

    return convexity_map


def create_convexitymap(depth, mask, r):
    norm = compute_normal(depth)
    convexity_map = compute_convexitymap(norm, mask, depth, window_ratio=r)
    convexity_map = 255 - convexity_map # 색상 반전
    convexity_map = min_max_normalize_(convexity_map, mask)
    convexity_map[np.where(mask == 0)] = 0

    return convexity_map