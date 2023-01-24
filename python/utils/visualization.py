import cv2
import numpy as np
import os
import random

from utils.normalization import min_max_normalize
from utils.filters import morphology_closing


def visualize_map(mat, mask, result_path, filename, filetype = "None"):
    mat = min_max_normalize(mat, mask)
    file_path = os.path.join(result_path, f'{filename}_{filetype}.jpg')
    cv2.imwrite(file_path, mat)


# TODO: closing 연산 따로 빼기
def visualize_result(rgb, mask, region_list, alpha, result_path, filename):
    rgb[np.where(mask == 0)] = [0, 0, 0]  # 배경 제거
    output = rgb.copy()

    for idx, (region, _) in enumerate(region_list):
        rand_color = [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]
        output[np.where(morphology_closing(region, (2, 2)) == 255)] = rand_color
    cv2.addWeighted(output, alpha, rgb, 1 - alpha, 0, rgb)

    # cv2.imshow('result', rgb)
    # cv2.waitKey(0)

    file_path = os.path.join(result_path, filename + "_final.jpg")
    cv2.imwrite(file_path, rgb)