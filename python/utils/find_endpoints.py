import numpy as np
import cv2


def skeleton_endpoints(skel):
    skel = skel.copy()
    skel[skel!=0] = 1
    skel = np.uint8(skel)

    kernel = np.uint8([[1,  1, 1],
                       [1, 10, 1],
                       [1,  1, 1]])
    filtered = cv2.filter2D(skel, -1, kernel)

    # 끝 점만 255 처리한 이미지 행렬
    out = np.zeros_like(skel)
    out[np.where(filtered==11)] = 255

    # 끝 점 배열
    endpoints = np.vstack((np.where(filtered==11)[:2])).T

    return endpoints


def connect_endpoint_and_mask(final_contour, endpoints, mask_contour):
    contour_points = np.vstack(np.where(mask_contour == 255)[:2]).T

    for point in endpoints:
        diff = contour_points - point
        dist = np.array([np.linalg.norm(i) for i in diff])
        closed_point = contour_points[np.argmin(dist)]

        cv2.line(final_contour, (point[1], point[0]), (closed_point[1], closed_point[0]), 255, thickness = 1)
        cv2.line(final_contour, (closed_point[1] + 1, closed_point[0]), (closed_point[1] + 1, closed_point[0]), 255,
                 thickness=1)
        cv2.line(final_contour, (closed_point[1] - 1, closed_point[0]), (closed_point[1] - 1, closed_point[0]), 255,
                 thickness=1)
        cv2.line(final_contour, (closed_point[1], closed_point[0] + 1), (closed_point[1], closed_point[0] + 1), 255,
                 thickness=1)
        cv2.line(final_contour, (closed_point[1], closed_point[0] - 1), (closed_point[1], closed_point[0] - 1), 255,
                 thickness=1)

    return final_contour