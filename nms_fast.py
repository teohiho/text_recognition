import numpy as np
import cv2

import nms_helpers as help


def rectangle_iou(rectA, rectB):
    """
    Computes the ratio of the intersection area of the input rectangles to the (sum of rectangle areas - intersection area)
    Used with the nms_fast function
    @param rectA: a rectangle described by (x,y,w,h)
    @type rectA:  tuple
    @param rectB: a rectangle describe by (x,y,w,h)
    @type rectB: tuple
    @return: The ratio of overlap between rectA and rectB  (intersection area/(rectA area + rectB area - intersection area)
    @rtype: float
    """
    rectAd = ((rectA[0], rectA[1]),(rectA[2],rectA[3]), 0)
    rectBd = ((rectB[0], rectB[1]),(rectB[2], rectB[3]), 0)

    # rotatedRectangleIntersection expects (rrect, rrect) as ((cx, cy), (w, h), deg)
    retVal, region = cv2.rotatedRectangleIntersection(rectAd, rectBd)

    # cv2.rotatedRectangleIntersection -- rectangles passed are rotated around their centers.

    if cv2.INTERSECT_NONE == retVal:
        return 0
    elif cv2.INTERSECT_FULL == retVal:
        return 1.0
    else:
        # https://docs.opencv.org/master/d3/dc0/group__imgproc__shape.html#ga2c759ed9f497d4a618048a2f56dc97f1
        intersection_area = cv2.contourArea(region)
        rectA_area = rectA[2] * rectA[3]
        rectB_area = rectB[2] * rectB[3]

        return intersection_area / (rectA_area + rectB_area - intersection_area)


def rotated_rect_iou(rectA, rectB):
    """
    Computes the ratio of the intersection area of the input rectangles to the (sum of rectangle areas - intersection area)
    Used with the nms_fast function
    @param rectA: a polygon (rectangle) described by its verticies
    @type rectA: list
    @param rectB: a polygon (rectangle) describe by it verticies
    @type rectB: list
    @return: The ratio of the intersection area / (sum of rectangle areas - intersection area)
    @rtype: float
    """
    return polygon_iou(rectA, rectB)


def polygon_iou(poly1, poly2, useCV2=True):
    """
    Computes the ratio of the intersection area of the input polygons to the (sum of polygon areas - intersection area)
    Used with the nms_fast function
    @param rectA: a polygon described by its verticies
    @type rectA: list
    @param rectB: a polygon describe by it verticies
    @type rectB: list
    @return: The ratio of the intersection area / (sum of rectangle areas - intersection area)
    @rtype: float
    """

    intersection_area = help.polygon_intersection_area([poly1, poly2])
    if intersection_area == 0:
        return 0

    if(useCV2):
        poly1_area = cv2.contourArea(np.array(poly1, np.int32))
        poly2_area = cv2.contourArea(np.array(poly2, np.int32))
    else:
        poly1_area = help.polygon_intersection_area([poly1])
        poly2_area = help.polygon_intersection_area([poly2])

    return intersection_area / (poly1_area + poly2_area - intersection_area)


def NMS(boxes, scores, **kwargs):
    """
    Do Non Maximal Suppression as translated from the openCV c++ source in nms.cpp and nms.inl.hpp
    // Inspired by Piotr Dollar's NMS implementation in EdgeBox.
    // https://goo.gl/jV3JYS
    This function is usually called from nms_boxes, nms_rrects, nms_polys
    @param boxes:  the boxes to compare, the structure of the boxes must be compatible with the compare_function.
    @type boxes:  list
    @param scores: the scores associated with boxes
    @type scores: list
    @param kwargs:
        score_threshold: the minimum score necessary to be a viable solution, default 0.3, float
        nms_threshold: the minimum nms value to be a viable solution, default: 0.4, float
        compare_function: function that accepts two boxes and returns their overlap ratio, this function must accept
        two boxes and return an overlap ratio
        eta: a coefficient in adaptive threshold formula: nms_thresholdi+1=eta⋅nms_threshold, default: 1.0, float
        top_k: if >0, keep at most top_k picked indices. default:0, int
    @type kwargs: dict
    @return: an array of indicies of the best boxes
    @rtype: list
    """

    if 'eta' in kwargs:
        eta = kwargs['eta']
    else:
        eta = 1.0
    assert 0 < eta <= 1.0

    if 'top_k' in kwargs:
        top_k = kwargs['top_k']
    else:
        top_k = 0
    assert 0 <= top_k

    if 'score_threshold' in kwargs:
        score_threshold = kwargs['score_threshold']
    else:
        score_threshold = 0.3
    assert score_threshold > 0

    if 'nms_threshold' in kwargs:
        nms_threshold = kwargs['nms_threshold']
    else:
        nms_threshold = 0.4
    assert 0 < nms_threshold < 1

    if 'compare_function' in kwargs:
        compare_function = kwargs['compare_function']
    else:
        compare_function = None
    assert compare_function is not None

    if len(boxes) == 0:
        return []

    assert len(scores) == len(boxes)
    assert scores is not None

    # sort scores descending and convert to [[score], [indexx], . . . ]
    scores = help.get_max_score_index(scores, score_threshold, top_k)

    # Do Non Maximal Suppression
    # This is an interpretation of NMS from the openCV source in nms.cpp and nms.
    adaptive_threshold = nms_threshold
    indicies = []

    for i in range(0, len(scores)):
        idx = int(scores[i][1])
        keep = True
        for k in range(0, len(indicies)):
            if not keep:
                break
            kept_idx = indicies[k]
            overlap = compare_function(boxes[idx], boxes[kept_idx])
            keep = (overlap <= adaptive_threshold)

        if keep:
            indicies.append(idx)

        if keep and (eta < 1) and (adaptive_threshold > 0.5):
                adaptive_threshold = adaptive_threshold * eta

    return indicies