import cv2
import math
import numpy as np



def createImage(width=800, height=800, depth=3):
    """
    Return a black image with an optional scale on the edge
    @param width: width of the returned image
    @type width: int
    @param height: height of the returned image
    @type height: int
    @param depth: either color.depth_rgb/bgr or color.depth_mono.  If mono, no scale is drawn
    @type depth: int
    @return: A zero'd out matrix/black image of size (width, height)
    @rtype: np.ndarray
    """
    # create a black image and put a scale on the edge
    hashDistance = 50
    hashLength = 20

    img = np.zeros((int(height), int(width), depth), np.uint8)

    if(depth == 3):
        for x in range(0, int(width / hashDistance)):
            cv2.line(img, (x * hashDistance, 0), (x * hashDistance, hashLength), (0,0,255), 1)

        for y in range(0, int(width / hashDistance)):
            cv2.line(img, (0, y * hashDistance), (hashLength, y * hashDistance), (0,0,255), 1)

    return img


def polygon_intersection_area(polygons, showImages=False):
    """
    Compute the area of intersection of an array of polygons
    @param polygons: a list of polygons
    @type polygons: list
    @param showImages: if True, display an image of each polygon and their intersection
    @type showImages: bool
    @return: the area of intersection of the polygons
    @rtype: int
    """
    if len(polygons) == 0:
        return 0

    dx = 0
    dy = 0

    maxx = np.amax(np.array(polygons)[...,0])
    minx = np.amin(np.array(polygons)[...,0])
    maxy = np.amax(np.array(polygons)[...,1])
    miny = np.amin(np.array(polygons)[...,1])

    if minx < 0:
        dx = -int(minx)
        maxx = maxx + dx
    if miny < 0:
        dy = -int(miny)
        maxy = maxy + dy
    # (dx, dy) is used as an offset in fillPoly

    for i, polypoints in enumerate(polygons):

        newImage = createImage(maxx, maxy, 1)

        polypoints = np.array(polypoints, np.int32)
        polypoints = polypoints.reshape(-1, 1, 2)

        cv2.fillPoly(newImage, [polypoints], (255, 255, 255), cv2.LINE_8, 0, (dx, dy))

        if(i == 0):
            compositeImage = newImage
        else:
            compositeImage = cv2.bitwise_and(compositeImage, newImage)

        area = cv2.countNonZero(compositeImage)

        if showImages:
            winname = "images[{}]".format(i)
            cv2.imshow(winname, newImage)
            cv2.moveWindow(winname, int(i * maxx), 0)
            cv2.imshow("compositeImage", compositeImage)
            cv2.moveWindow("compositeImage", 0, int(maxy)+60)

    if showImages:
        cv2.waitKey(0)

    return area


def get_max_score_index(scores, threshold=0, top_k=0, descending=True):
    """
    Get the max scores with corresponding indicies
    Translation from the openCV c++ source in nms.cpp and nms.inl.hpp
    @param scores: a list of scores
    @type scores: list
    @param threshold: consider scores higher than this threshold
    @type threshold: float
    @param top_k: return at most top_k scores; if 0, keep all
    @type top_k: int
    @param descending: if True, list is returened in descending order, else ascending
    @return: a  sorted by score list  of [score, index]
    @rtype: list
    """
    score_index = []

    # Generate index score pairs
    for i, score in enumerate(scores):
        if (threshold > 0) and (score > threshold):
            score_index.append([score, i])
        else:
            score_index.append([score, i])

    # Sort the score pair according to the scores in descending order
    npscores = np.array(score_index)

    if descending:
        npscores = npscores[npscores[:,0].argsort()[::-1]] #descending order
    else:
        npscores = npscores[npscores[:,0].argsort()] # ascending order

    if top_k > 0:
        npscores = npscores[0:top_k]

    return npscores.tolist()

def rects2polys(rects, baggage, ratioWidth, ratioHeight):

    polygons = []
    for index, box in enumerate(rects):
        upperLeftX = box[0]
        upperLeftY = box[1]
        lowerRightX = box[0] + box[2]
        lowerRightY = box[1] + box[3]

        # scale the bounding box coordinates based on the respective ratios
        upperLeftX = int(upperLeftX * ratioWidth)
        upperLeftY = int(upperLeftY * ratioHeight)
        lowerRightX = int(lowerRightX * ratioWidth)
        lowerRightY = int(lowerRightY * ratioHeight)

        # unpack the corresponding baggage
        # pp.pprint(baggage[index])
        offset = baggage[index]['offset']
        theta = baggage[index]['angle']

        # create an array of the rectangle's verticies
        points = [
            (upperLeftX, upperLeftY),
            (lowerRightX, upperLeftY),
            (lowerRightX, lowerRightY),
            (upperLeftX, lowerRightY)
        ]

        # the offset is the point at which the rectangle is rotated
        rotationPoint = (int(offset[0] * ratioWidth), int(offset[1] * ratioHeight))

        polygons.append(rotatePoints(points, theta, rotationPoint))

    return polygons


def rotatePoints(points, theta, around):
    negaround = (-around[0], -around[1])
    rotated = []
    for xy in points:
        # rotated.append(translate(rotate(translate(xy, negaround), -theta), around))
        rotated.append(rotate_around_point(xy, theta, around))

    return rotated


def rotate_around_point(xy, radians, origin=(0, 0)):
    """Rotate a point around a given point.
    # https://gist.github.com/LyleScott/e36e08bfb23b1f87af68c9051f985302
    I call this the "high performance" version since we're caching some
    values that are needed >1 time. It's less readable than the previous
    function but it's faster.
    """
    x, y = xy
    offset_x, offset_y = origin
    adjusted_x = (x - offset_x)
    adjusted_y = (y - offset_y)
    cos_rad = math.cos(radians)
    sin_rad = math.sin(radians)
    qx = offset_x + cos_rad * adjusted_x + sin_rad * adjusted_y
    qy = offset_y + -sin_rad * adjusted_x + cos_rad * adjusted_y

    return qx, qy