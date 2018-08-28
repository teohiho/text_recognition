# USAGE
# python text_detection.py --image images/lebron_james.jpg --east frozen_east_text_detection.pb

# import the necessary packages
import numpy as np
import argparse
import time
import cv2

from decode import decode
from draw import drawPolygons, drawBoxes
import nms
import nms_helpers as help


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str,
    help="path to input image")
ap.add_argument("-east", "--east", type=str,
    help="path to input EAST text detector")
ap.add_argument("-c", "--min-confidence", type=float, default=0.5,
    help="minimum probability required to inspect a region")
ap.add_argument("-w", "--width", type=int, default=320,
    help="resized image width (should be multiple of 32)")
ap.add_argument("-e", "--height", type=int, default=320,
    help="resized image height (should be multiple of 32)")
args = vars(ap.parse_args())

# load the input image and grab the image dimensions
image = cv2.imread(args["image"])
orig = image.copy()
(origHeight, origWidth) = image.shape[:2]

# set the new width and height and then determine the ratio in change
# for both the width and height
(newW, newH) = (args["width"], args["height"])
ratioWidth = origWidth / float(newW)
ratioHeight = origHeight / float(newH)

# resize the image and grab the new image dimensions
image = cv2.resize(image, (newW, newH))
(imageHeight, imageWidth) = image.shape[:2]

# define the two output layer names for the EAST detector model that
# we are interested -- the first is the output probabilities and the
# second can be used to derive the bounding box coordinates of text
layerNames = [
    "feature_fusion/Conv_7/Sigmoid",
    "feature_fusion/concat_3"]

# load the pre-trained EAST text detector
print("[INFO] loading EAST text detector...")
net = cv2.dnn.readNet(args["east"])

# construct a blob from the image and then perform a forward pass of
# the model to obtain the two output layer sets
blob = cv2.dnn.blobFromImage(image, 1.0, (imageWidth, imageHeight), (123.68, 116.78, 103.94), swapRB=True, crop=False)

start = time.time()
net.setInput(blob)
(scores, geometry) = net.forward(layerNames)
end = time.time()

# show timing information on text prediction
print("[INFO] text detection took {:.6f} seconds".format(end - start))


# NMS on the the unrotated rects
confidenceThreshold = args['min_confidence']
nmsThreshold = 0.4

# decode the blob info
(rects, confidences, baggage) = decode(scores, geometry, confidenceThreshold)

##########################################################

functions = nms.nms_functions
names = ["Felz", "Fast", "Mali"]

for i, function in enumerate(functions):

    start = time.time()
    indicies = nms.nms_boxes(rects, confidences, nms_function=function, confidence_threshold=confidenceThreshold,
                             nsm_threshold=nmsThreshold)
    end = time.time()

    indicies = np.array(indicies).reshape(-1)

    drawrects = np.array(rects)[indicies]
    #confidences = np.array(confidences)[indicies]

    print("[INFO] {} NMS took {:.6f} seconds and found {} boxes".format(names[i], end - start, len(drawrects)))

    drawOn = orig.copy()
    drawBoxes(drawOn, drawrects, ratioWidth, ratioHeight, (0, 255, 0), 2)

    title = "nmx_boxes {}".format(names[i])
    cv2.imshow(title,drawOn)
    cv2.moveWindow(title, 150+i*300, 150)

cv2.waitKey(0)


# convert rects to polys
polygons = help.rects2polys(rects, baggage, ratioWidth, ratioHeight)


for i, function in enumerate(functions):

    start = time.time()
    indicies = nms.nms_polygons(polygons, confidences, nms_function=function, confidence_threshold=confidenceThreshold,
                             nsm_threshold=nmsThreshold)
    end = time.time()

    indicies = np.array(indicies).reshape(-1)

    drawpolys = np.array(polygons)[indicies]
    #confidences = np.array(confidences)[indicies]

    print("[INFO] {} NMS took {:.6f} seconds and found {} boxes".format(names[i], end - start, len(drawpolys)))

    drawOn = orig.copy()
    drawPolygons(drawOn, drawpolys, ratioWidth, ratioHeight, (0, 255, 0), 2)

    title = "nmx_boxes {}".format(names[i])
    cv2.imshow(title,drawOn)
    cv2.moveWindow(title, 150+i*300, 150)


    import os
    filename = os.path.basename(args['image'])
    print(filename)
    cv2.imwrite("images/out/{}".format(filename),drawOn)

cv2.waitKey(0)