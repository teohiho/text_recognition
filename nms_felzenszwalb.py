import numpy as np
import cv2

import nms_helpers as help


def rect_areas(rects):
    # rect = x,y,w,h
    rects = np.array(rects)
    w = rects[:,2]
    h = rects[:,3]
    return w * h


def rect_compare(rect1, rect2, area):
    # rect = x,y, w, h
    xx1 = max(rect1[0], rect2[0])
    yy1 = max(rect1[1], rect2[1])
    xx2 = min(rect1[0] + rect1[2], rect2[0] + rect2[2])
    yy2 = min(rect1[1] + rect1[3], rect2[1] + rect2[3])
    w = max(0, xx2 - xx1 + 1)
    h = max(0, yy2 - yy1 + 1)
    return float(w * h) / area


def poly_areas(polys):
    areas = []
    for poly in polys:
        areas.append(cv2.contourArea(np.array(poly, np.int32)))
    return areas


def poly_compare(poly1, poly2, area):
    assert area > 0
    intersection_area = help.polygon_intersection_area([poly1, poly2])
    return intersection_area/area


def NMS(boxes, scores, **kwargs):
    """
    NMS using Felzeszwalb et al. method
    Modified from  non_max_suppression_slow(boxes, overlapThresh)
    https://www.pyimagesearch.com/2014/11/17/non-maximum-suppression-object-detection-python/

    @param boxes: a list of boxes to perform NMS on
    @type boxes: list
    @param scores: a list of scores corresponding to boxes
    @type scores: list
    @param kwargs: a dictionary containing arguments necessary, required args do not have default values below
        top_k: if >0, keep at most top_k picked indices. default:0, int
        score_threshold: the minimum score necessary to be a viable solution, default 0.3, float
        nms_threshold: the minimum nms value to be a viable solution, default: 0.4, float
        compare_function: function that accepts two boxes and returns their overlap ratio, this function must accept
        two boxes and return an overlap ratio between 0 and 1
        area_function: function used to calculate the area of an element of boxes
    @type kwargs: dict
    @return: the indicies of the best boxes
    @rtype: list
    """

    if 'top_k' in kwargs:
        top_k = kwargs['top_k']
    else:
        top_k = 0
    assert 0 <= top_k

    if 'score_threshold' in kwargs:
        score_threshold = kwargs['score_threshold']
    else:
        score_threshold = 0.3
    assert 0 < score_threshold

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

    if 'area_function' in kwargs:
        area_function = kwargs['area_function']
    else:
        area_function = None
    assert area_function is not None

    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    if scores is not None:
        assert len(scores) == len(boxes)

    # initialize the list of picked indexes
    pick = []

    # compute the area of the bounding boxes
    area = area_function(boxes)

    # sort the boxes by score or the bottom-right y-coordinate of the bounding box
    if scores is not None:
        # sort the bounding boxes by the associated scores
        scores = help.get_max_score_index(scores, score_threshold, top_k, False)
        idxs = np.array(scores, np.int32)[:,1]
        #idxs = np.argsort(scores)
    else:
        # sort the bounding boxes by the bottom-right y-coordinate of the bounding box
        y2 = boxes[:3]
        idxs = np.argsort(y2)

    # keep looping while some indexes still remain in the indexes list
    while len(idxs) > 0:
        # grab the last index in the indexes list, add the index
        # value to the list of picked indexes, then initialize
        # the suppression list (i.e. indexes that will be deleted)
        # using the last index
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        suppress = [last]

        # loop over all indexes in the indexes list
        for pos in range(0, last):
            # grab the current index
            j = idxs[pos]

            # compute the ratio of overlap between the two boxes and the area of the second box
            overlap = compare_function(boxes[i], boxes[j], area[j])

            # if there is sufficient overlap, suppress the current bounding box
            if overlap > nms_threshold:
                suppress.append(pos)

        # delete all indexes from the index list that are in the suppression list
        idxs = np.delete(idxs, suppress)

    # return only the indicies of the bounding boxes that were picked
    return pick