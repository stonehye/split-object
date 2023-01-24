import cv2
import numpy as np
import os
import random
import argparse

from utils.convexity import create_convexitymap
from utils.contour import *
from utils.filters import *
from utils.region_merging import *
from utils.visualization import visualize_map, visualize_result
from utils.find_endpoints import *


def get_bbox_with_mask(mask, depth_width=192, depth_height=256):
    if mask is None:
        return 0, depth_width, 0, depth_height
    if mask.ndim != 2:
        return 0, mask.shape[1], 0, mask.shape[0]

    mask_index = np.where(mask == 255)

    if len(mask_index[0]) == 0 or len(mask_index[1]) == 0:
        return 0, mask.shape[1], 0, mask.shape[0]

    x_max = mask_index[1].max() + 5
    x_min = mask_index[1].min() - 5
    y_max = mask_index[0].max() + 10
    y_min = mask_index[0].min() - 5

    x_min = max(x_min, 0)
    y_min = max(y_min, 0)
    x_max = min(x_max, mask.shape[1])
    y_max = min(y_max, mask.shape[0])

    return x_min, x_max, y_min, y_max


def crop_image_with_mask(img, mask):
    x_min, x_max, y_min, y_max = get_bbox_with_mask(mask)
    ratio = img.shape[0] / mask.shape[0]
    return img[int(y_min * ratio):int(y_max * ratio), int(x_min * ratio):int(x_max * ratio)]


def remove_nonconnected(contour):
    # TODO: 속도 개선 필요
    num_labels, labeled_img = cv2.connectedComponents(contour.astype(np.uint8), connectivity=4)  # CCL
    labeled_img[np.where((labeled_img != 0) & (labeled_img != 1))] = 0
    contour = labeled_img * 255

    return contour


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="인접 사물 분리 (Segmentation)")
    parser.add_argument('-rgb', '--rgb', type=str, required=True, help='입력 컬러 이미지 (required)')
    parser.add_argument('-m', '--mask', type=str, required=True, help='입력 마스크 이미지 (required)')
    parser.add_argument('-d', '--depth', type=str, required=True, help='입력 깊이 이미지 (required)')
    parser.add_argument('-o', '--output_dir', type=str, default=f'./', help='결과 출력 디렉토리 경로 (default = ./)')
    parser.add_argument('-rs', '--region_size', type=float, default=0.05,
                        help='영역 병합 시 최소 영역 임계값 (객체 크기 대비 비율, default = 0.05)')

    args = parser.parse_args()

    f_rgb = args.rgb
    f_mask = args.mask
    f_depth = args.depth
    output_dir = args.output_dir
    region_size_thr = args.region_size

    os.makedirs(output_dir, exist_ok=True)

    filename = f_depth.split('%')[0]
    print(f_rgb)

    depth_ = cv2.imread(f_depth, cv2.IMREAD_ANYDEPTH)
    depth = depth_.astype('float64')
    mask = cv2.imread(f_mask, 0)
    rgb = cv2.imread(f_rgb)
    rgb = cv2.resize(rgb, (depth.shape[1], depth.shape[0]))  # depth 해상도에 맞게 리사이징

    original_shape = depth.shape
    x_min, x_max, y_min, y_max = get_bbox_with_mask(mask)

    depth = crop_image_with_mask(depth, mask)
    rgb = crop_image_with_mask(rgb, mask)
    mask = crop_image_with_mask(mask, mask)

    h, w = depth.shape[:2]  # crop 된 이미지 크기

    convexity_map = create_convexitymap(depth, mask, 0.05)
    cvx_contour = extract_convexityContour(convexity_map, mask)
    var_contour = extract_varContour(depth, mask)
    mask_contour = extract_maskContour(mask, thick=2)

    cvx_contour_ = cvx_contour.copy()
    cvx_contour_[np.where(extract_maskContour(mask, thick=2) != 0)] = 255
    cvx_contour_ = remove_nonconnected(cvx_contour_)
    cvx_contour_[np.where(extract_maskContour(mask, thick=4) != 0)] = 0

    var_contour_ = var_contour.copy()
    var_contour_[np.where(extract_maskContour(mask, thick=2) != 0)] = 255
    var_contour_ = remove_nonconnected(var_contour_)
    var_contour_[np.where(extract_maskContour(mask, thick=4) != 0)] = 0

    cvx_size = len(np.where(cvx_contour_ == 255)[0])
    var_size = len(np.where(var_contour_ == 255)[0])

    contour = mask_contour.copy()
    if cvx_size > var_size:
        contour[np.where(cvx_contour_ != 0)] = 255
        if var_size / (h * w) <= 1e-2:
            contour[np.where(var_contour_ != 0)] = 255
    else:
        contour[np.where(var_contour_ != 0)] = 255
        if cvx_size / (h * w) <= 1e-2:
            contour[np.where(cvx_contour_ != 0)] = 255

    contour = median_filter(contour, mask, 1, INMASK=False)  # median filtering
    contour = cv2.blur(contour, (3, 3))
    contour[np.where(contour != 0)] = 255
    contour = min_filter(contour, 1)

    # TODO: 속도 개선 필요
    final_contour = thinning_contour(contour.astype(np.uint8))  # Thinning

    # contour close 화
    endpoints = skeleton_endpoints(final_contour)
    if len(endpoints):
        final_contour = connect_endpoint_and_mask(final_contour, endpoints, extract_maskContour(mask, thick=1))

    final_contour_ = 255 - final_contour  # final contour 색상 반전
    num_labels, labeled_img = cv2.connectedComponents(final_contour_.astype(np.uint8), connectivity=4)  # CCL

    region_list = region_merging(labeled_img, mask, region_size_thr)  # 영역 병합

    # 결과 저장
    labeled_img = np.zeros(original_shape)
    for idx, (region, _) in enumerate(region_list, start=1):
        (x, y) = np.where(morphology_closing(region, (2, 2)) == 255) # TODO: 속도 개선 필요

        temp_mask = np.zeros(original_shape)
        temp_mask[(x + y_min, y + x_min)] = 255
        labeled_img[(x + y_min, y + x_min)] = idx

        mask_path = os.path.join(output_dir, f'{idx}.jpg')
        cv2.imwrite(mask_path, temp_mask)
    labeled_img_path = os.path.join(output_dir, 'label.jpg')
    cv2.imwrite(labeled_img_path, labeled_img)