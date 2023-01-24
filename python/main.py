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


def remove_nonconnected(contour):
    # TODO: 속도 개선 필요
    num_labels, labeled_img = cv2.connectedComponents(contour.astype(np.uint8), connectivity=4)  # CCL
    labeled_img[np.where((labeled_img != 0) & (labeled_img != 1))] = 0
    contour = labeled_img * 255

    return contour


def crop_image_with_mask(img, mask):
    x_min, x_max, y_min, y_max = get_bbox_with_mask(mask)
    ratio = img.shape[0] / mask.shape[0]
    return img[int(y_min * ratio):int(y_max * ratio), int(x_min * ratio):int(x_max * ratio)]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Obeject Segmentation")
    parser.add_argument('-i', '--input_dir', type=str, required=True, help='입력 데이터셋 디렉토리 경로 (required)')
    parser.add_argument('-o', '--output_dir', type=str, default=f'./result', help='결과 출력 디렉토리 경로')
    parser.add_argument('-s', '--save', type=bool, default=False, help='로직 중간 결과물 출력 여부 (default: False)')
    parser.add_argument('-rs', '--region_size', type=float, default=0.05,
                        help='영역 병합 시 최소 영역 임계값 (객체 크기 대비 비율, default = 0.05)')

    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir
    region_size_thr = args.region_size
    SAVE = args.save

    # 결과 저장 디렉토리 생성
    result_path = os.path.join(output_dir, input_dir)
    os.makedirs(result_path, exist_ok=True)

    # 데이터 읽기
    file_list = os.listdir(input_dir)
    file_list_depth = [file for file in file_list if file.endswith('%depth.png')]
    file_list_mask = [file for file in file_list if file.endswith('%final_mask.png')]
    file_list_rgb = [file for file in file_list if file.endswith('%rgb.jpg')]

    for (f_depth, f_mask, f_rgb) in zip(file_list_depth, file_list_mask, file_list_rgb):
        filename = f_depth.split('%')[0]
        print(f_rgb)

        depth_ = cv2.imread(os.path.join(input_dir, f_depth), cv2.IMREAD_ANYDEPTH)
        depth = depth_.astype('float64')
        mask = cv2.imread(os.path.join(input_dir, f_mask), 0)
        rgb = cv2.imread(os.path.join(input_dir, f_rgb))
        rgb = cv2.resize(rgb, (depth.shape[1], depth.shape[0])) # depth 해상도에 맞게 리사이징

        blur_depth = cv2.medianBlur(depth_, 5)

        # mask 영역 크기만큼 crop
        depth = crop_image_with_mask(depth, mask)
        blur_depth = crop_image_with_mask(blur_depth, mask)
        rgb = crop_image_with_mask(rgb, mask)
        mask = crop_image_with_mask(mask, mask)

        h, w = depth.shape[:2] # crop 된 이미지 크기

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

        # print(cvx_size / (h*w), var_size / (h*w))

        if SAVE:
            visualize_map(convexity_map, mask, result_path, filename, filetype="convexity")
            visualize_map(depth, mask, result_path, filename, filetype="depth")

            file_path = os.path.join(result_path, filename + "_1.jpg")
            cv2.imwrite(file_path, cvx_contour_)

            file_path = os.path.join(result_path, filename + "_2.jpg")
            cv2.imwrite(file_path, var_contour_)

        contour = mask_contour.copy()
        if cvx_size > var_size:
            contour[np.where(cvx_contour_ != 0)] = 255
            if var_size / (h*w) <= 1e-2:
                contour[np.where(var_contour_ != 0)] = 255
        else:
            contour[np.where(var_contour_ != 0)] = 255
            if cvx_size / (h*w) <= 1e-2:
                contour[np.where(cvx_contour_ != 0)] = 255

        contour = median_filter(contour, mask, 1, INMASK=False)  # median filtering
        contour = cv2.blur(contour, (3, 3))
        contour[np.where(contour!=0)] = 255
        contour = min_filter(contour, 1)

        if SAVE:
            file_path = os.path.join(result_path, filename + "_3.jpg")
            cv2.imwrite(file_path, contour)

        final_contour = thinning_contour(contour.astype(np.uint8)) # Thinning

        if SAVE:
            file_path = os.path.join(result_path, filename + "_4.jpg")
            cv2.imwrite(file_path, contour)

        # contour close 화
        endpoints = skeleton_endpoints(final_contour)
        if len(endpoints):
            final_contour = connect_endpoint_and_mask(final_contour, endpoints, extract_maskContour(mask, thick=1))

        if SAVE:
            file_path = os.path.join(result_path, filename + "_5.jpg")
            cv2.imwrite(file_path, final_contour)

        final_contour_ = 255 - final_contour  # final contour 색상 반전
        num_labels, labeled_img = cv2.connectedComponents(final_contour_.astype(np.uint8), connectivity=4)  # CCL

        region_list = region_merging(labeled_img, mask, region_size_thr) # 영역 병합

        visualize_result(rgb, mask, region_list, 0.8, result_path, filename)


