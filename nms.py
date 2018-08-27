import cv2

import nms_fast
import nms_felzenszwalb as felz
import nms_malisiewicz as mali


felzenswalb = felz.NMS
malisiewicz = mali.NMS
fast = nms_fast.NMS

nms_functions = [felzenswalb, malisiewicz, fast]
default_nms = malisiewicz


def nms_rboxes(rrects, scores, nms_function=default_nms, **kwargs):
    """
    Do NMS for rotated rectangles
    @param rrects: a list of polygons, each described by ((cx, cy), (w,h), deg)
    @type rrects: list
    @param scores: a list of the scores associated with the rects
    @type scores: list
    @param nms_function: the NMS comparison function to use, kwargs will be passed to this function
    @type nms_function: function
    @return: an array of indicies of the best rrects
    @rtype: list
    """

    # convert the rrects to polys
    polys = []
    for rrect in rrects:
        r = cv2.boxPoints(rrect)
        print(r)
        polys.append(r)

    return nms_polygons(polys, scores, nms_function, **kwargs)


def nms_polygons(polys, scores,  nms_function=default_nms, **kwargs):
    """
    Do NMS for polygons (NMSBoxes with rotated rects)
    @param polys: a list of polygons, each described by their xy verticies
    @type polys: list
    @param scores: a list of the scores associated with the polygons
    @type scores: list
    @param nms_function: the NMS comparison function to use, kwargs will be passed to this function
    @type nms_function: function
    @return: an array of indicies of the best polys
    @rtype: list
    """

    assert nms_function in nms_functions

    if nms_function == fast:
        kwargs['compare_function'] = nms_fast.polygon_iou

    if nms_function == felzenswalb:
        kwargs['area_function'] = felz.poly_areas
        kwargs['compare_function'] = felz.poly_compare

    if nms_function == malisiewicz:
        kwargs['area_function'] = mali.poly_areas
        kwargs['compare_function'] = mali.poly_compare

    return nms_function(polys, scores, **kwargs)


def nms_boxes(rects, scores, nms_function=default_nms, **kwargs):
    """
    Do NMS for rectangles
    @param rects: a list of rectangles, each described by (x, y, w, h) (same as cv2.NMSBoxes)
    @type rects: list
    @param scores: a list of the scores associated with rects
    @type scores: list
    @param nms_function: the NMS comparison function to use, kwargs will be passed to this function
    @type nms_function: function
    @return: an array of indicies of the best rects
    @rtype: list
    """

    assert nms_function in nms_functions

    if nms_function == fast:
        kwargs['compare_function'] = nms_fast.rectangle_iou

    if nms_function == felzenswalb:
        kwargs['area_function'] = felz.rect_areas
        kwargs['compare_function'] = felz.rect_compare

    if nms_function == malisiewicz:
        kwargs['area_function'] = mali.rect_areas
        kwargs['compare_function'] = mali.rect_compare

    return nms_function(rects, scores, **kwargs)