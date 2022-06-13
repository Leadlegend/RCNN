import cv2
import numpy as np

from model.regoin_proposal import selective_search


def region_proposal(img, img_size):
    '''
    use selective-search to obtain image's region proposals;
    add some hand-crafted rules to refine proposals
    '''
    img_lbl, regions = selective_search(img, scale=500, sigma=0.9, min_size=10)
    candidates = set()
    images = []
    vertices = []
    for r in regions:
        if r['rect'] in candidates:
            continue
        if r['size'] < 200:
            continue
        if (r['rect'][2] * r['rect'][3]) < 500:
            continue
        proposal_img, proposal_vertice = clip_pic(img, r['rect'])
        if len(proposal_img) == 0:
            continue
        x, y, w, h = r['rect']
        if w == 0 or h == 0:
            continue
        [a, b, c] = np.shape(proposal_img)
        if a == 0 or b == 0 or c == 0:
            continue
        resized_proposal_img = resize_image(proposal_img, img_size, img_size)
        candidates.add(r['rect'])
        img_float = np.asarray(resized_proposal_img, dtype="float32")
        images.append(img_float)
        vertices.append(proposal_vertice)
    return images, vertices


def iou(ver1, vertice2, if_wh=False):
    '''
    calculate the IOU of two verts
    :param ver1: the first vert [x_min, y_min, x_max, y_max] / [x, y, w, h]
    :param vertice2: the second vert [x_min, y_min, x_max, y_max, w, h]
    :param if_wh: whether ver1 is of [x, y, w, h]
    :return: IOU value
    '''
    if if_wh:
        vertice1 = [ver1[0], ver1[1], ver1[0] + ver1[2], ver1[1] + ver1[3]]
    else:
        vertice1 = ver1
    area_inter = if_intersection(vertice1[0], vertice1[2], vertice1[1],
                                 vertice1[3], vertice2[0], vertice2[2],
                                 vertice2[1], vertice2[3])
    if area_inter:
        area_1 = ver1[2] * \
            ver1[3] if if_wh else (ver1[2]-ver1[0]) * (ver1[3]-ver1[1])
        area_2 = vertice2[4] * vertice2[5]
        iou = float(area_inter) / (area_1 + area_2 - area_inter)
        return iou
    else:
        return False


def if_intersection(xmin_a, xmax_a, ymin_a, ymax_a, xmin_b, xmax_b, ymin_b,
                    ymax_b):
    if_intersect = False
    if xmin_a < xmax_b <= xmax_a and (ymin_a < ymax_b <= ymax_a
                                      or ymin_a <= ymin_b < ymax_a):
        if_intersect = True
    elif xmin_a <= xmin_b < xmax_a and (ymin_a < ymax_b <= ymax_a
                                        or ymin_a <= ymin_b < ymax_a):
        if_intersect = True
    elif xmin_b < xmax_a <= xmax_b and (ymin_b < ymax_a <= ymax_b
                                        or ymin_b <= ymin_a < ymax_b):
        if_intersect = True
    elif xmin_b <= xmin_a < xmax_b and (ymin_b < ymax_a <= ymax_b
                                        or ymin_b <= ymin_a < ymax_b):
        if_intersect = True
    else:
        return if_intersect

    x_sorted_list = sorted([xmin_a, xmax_a, xmin_b, xmax_b])
    y_sorted_list = sorted([ymin_a, ymax_a, ymin_b, ymax_b])
    x_intersect_w = x_sorted_list[2] - x_sorted_list[1]
    y_intersect_h = y_sorted_list[2] - y_sorted_list[1]
    area_inter = x_intersect_w * y_intersect_h
    return area_inter


def resize_image(in_image,
                 new_width,
                 new_height,
                 out_image=None,
                 resize_mode=cv2.INTER_CUBIC):
    '''
    :param in_image: input image
    :param new_width: width of the new image
    :param new_height: height of the new image
    :param out_image: new path of the new image
    :param resize_mode: resize mode in CV2
    :return: the new image
    '''
    img = cv2.resize(in_image, (new_width, new_height), resize_mode)
    if out_image is not None:
        cv2.imwrite(out_image, img)
    return img


def clip_pic(img, rect):
    '''

    :param img: input image
    :param rect: four params of the rect
    :return: the clipped image and the rect
    '''
    assert len(rect) == 4
    x_min, y_min, w, h = rect[0], rect[1], rect[2], rect[3]
    x_max = x_min + w
    y_max = y_min + h
    return img[y_min:y_max, x_min:x_max, :], [x_min, y_min, x_max, y_max, w, h]
