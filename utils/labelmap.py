from PIL import Image
import numpy as np
import cv2
"""
REFER: https://hub.packtpub.com/opencv-detecting-edges-lines-shapes/
2018-06-30 Yonv1943
2018-07-01 comment to test.png
2018-07-01 gray in threshold, hierarchy
2018-07-01 draw_approx_hull_polygon() no [for loop]
2018-11-24 
"""

InternalSize = (512, 512)


def scaled_size(scale, size, or_mask=0):
    return (int(size[0] * scale) | or_mask, int(size[1] * scale) | or_mask)


def area_of_size(size):
    return size[0] * size[1]


def filter_coutours_by_area(contours, min_area=0, max_area=np.inf):
    filtered_contours = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area >= min_area and area <= max_area:
            filtered_contours.append(cnt)

    return filtered_contours


def draw_min_rect_circle(img, contours):
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)  # blue

        min_rect = cv2.minAreaRect(cnt)  # min_area_rectangle
        min_rect = np.int0(cv2.boxPoints(min_rect))
        cv2.drawContours(img, [min_rect], 0, (0, 255, 0), 2)  # green

        (x, y), radius = cv2.minEnclosingCircle(cnt)
        center, radius = (int(x),
                          int(y)), int(radius)  # center and radius of minimum enclosing circle
        img = cv2.circle(img, center, radius, (0, 0, 255), 2)  # red
    return img


def draw_approx_hull_polygon(img, contours):
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    cv2.drawContours(img, contours, -1, (255, 0, 0), 2)  # blue

    min_side_len = img.shape[0] / 32  # 多边形边长的最小值 the minimum side length of polygon
    min_poly_len = img.shape[0] / 16  # 多边形周长的最小值 the minimum round length of polygon
    min_side_num = 3  # 多边形边数的最小值
    approxs = [cv2.approxPolyDP(cnt, min_side_len, True) for cnt in contours]  # 以最小边长为限制画出多边形
    approxs = [approx for approx in approxs
               if cv2.arcLength(approx, True) > min_poly_len]  # 筛选出周长大于 min_poly_len 的多边形
    approxs = [approx for approx in approxs
               if len(approx) > min_side_num]  # 筛选出边长数大于 min_side_num 的多边形
    # Above codes are written separately for the convenience of presentation.
    cv2.polylines(img, approxs, True, (0, 255, 0), 2)  # green

    hulls = [cv2.convexHull(cnt) for cnt in contours]
    cv2.polylines(img, hulls, True, (0, 0, 255), 2)  # red

    for cnt in contours:
        cv2.drawContours(img, [
            cnt,
        ], -1, (255, 0, 0), 2)  # blue

        epsilon = 0.02 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        cv2.polylines(img, [
            approx,
        ], True, (0, 255, 0), 2)  # green

        hull = cv2.convexHull(cnt)
        cv2.polylines(img, [
            hull,
        ], True, (0, 0, 255), 2)  # red
    return img


def convert_labelmap(image,
                     internal_size=InternalSize,
                     labelmap_size=None,
                     gamma=1.0,
                     blur_radius=0.1,
                     thresh_type='global',
                     thresh_min=30,
                     block_size=0.1,
                     thresh_C=30,
                     morph_iteration=3,
                     morph_kernel_size=0.005):
    assert image.ndim == 2
    original_size = image.shape
    labelmap_size = labelmap_size or original_size

    if internal_size != original_size:
        image = cv2.resize(image, internal_size, interpolation=cv2.INTER_LANCZOS4)

    if gamma != 1.0:
        image = (255 * np.power(image / 255.0, gamma)).astype(np.uint8)

    blurred_image = cv2.GaussianBlur(image, scaled_size(blur_radius, internal_size, or_mask=1), 0)

    if thresh_type == 'global':
        _, thresh = cv2.threshold(blurred_image, thresh_min, 255, 0)
    elif thresh_type == 'otsu':
        _, thresh = cv2.threshold(blurred_image, thresh_min, 255, cv2.THRESH_OTSU)
    elif thresh_type == 'adaptive_mean' or thresh_type == 'adaptive_gaussian':
        adaptive_method = cv2.ADAPTIVE_THRESH_MEAN_C if thresh_type == 'adaptive_mean' else cv2.ADAPTIVE_THRESH_GAUSSIAN_C
        block_size = scaled_size(block_size, internal_size, or_mask=1)[0]
        thresh = cv2.adaptiveThreshold(blurred_image, 255, adaptive_method, 0, block_size,
                                       thresh_C)
    else:
        raise NotImplementedError(f'thresh type [{thresh_type}] not implemented')

    if morph_iteration is not None:
        kernel = np.ones(scaled_size(morph_kernel_size, internal_size, or_mask=3), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=morph_iteration)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=morph_iteration)

    if internal_size != labelmap_size:
        labelmap = cv2.resize(thresh, labelmap_size, interpolation=cv2.INTER_LANCZOS4)
    else:
        labelmap = thresh

    return labelmap


def extract_edgemap(labelmap, color=(255, 255, 255), thickness=2, base_image=None):
    assert labelmap.ndim == 2
    contours, _ = cv2.findContours(labelmap, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if base_image is None:
        base_image = np.zeros_like(labelmap)
    else:
        assert base_image.shape == labelmap.shape
    edgemap = cv2.drawContours(base_image, contours, -1, color, thickness=thickness)
    return edgemap


def extract_bounding_boxes(labelmap, min_area_ratio=0, max_area_ratio=1):
    assert labelmap.ndim == 2
    contours, _ = cv2.findContours(labelmap, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_area = area_of_size(scaled_size(min_area_ratio, labelmap.shape))
    max_area = area_of_size(scaled_size(max_area_ratio, labelmap.shape))
    contours = filter_coutours_by_area(contours, min_area, max_area)

    bboxes = [cv2.boundingRect(cnt) for cnt in contours]
    return bboxes


if __name__ == '__main__':
    pass