#  https://www.pyimagesearch.com/2014/09/01/build-kick-ass-mobile-document-scanner-just-5-minutes/
# USAGE
# python scan.py --image images/page.jpg

# import the necessary packages
# from pyimageSearch.transform import four_point_transform

from skimage.filters import threshold_local
import numpy as np
import argparse
import cv2
import imutils


def scan(image):
	point = 4
	# construct the argument parser and parse the arguments

	# load the image and compute the ratio of the old height
	# to the new height, clone it, and resize it
	ratio = image.shape[0] / 500.0
	orig = image.copy()
	image = imutils.resize(image, height = 500)

	# convert the image to grayscale, blur it, and find edges
	# in the image
	# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	
	gray = cv2.GaussianBlur(gray, (5, 5), 0)
	edged = cv2.Canny(gray, 75, 200)

	# show the original image and the edge detected image
	print("STEP 1: Edge Detection")
	# cv2.imshow("Image", image)
	# cv2.imshow("Edged", edged)
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()

	# find the contours in the edged image, keeping only the
	# largest ones, and initialize the screen contour
	cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
	
	cnts = imutils.grab_contours(cnts)
	cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]
	
	# loop over the contours
	for c in cnts:
		
		# approximate the contour
		peri = cv2.arcLength(c, True) # chu vi	
		
		
		approx = cv2.approxPolyDP(c, 0.02 * peri, True)
		print(">>> approx: " + str(len(approx)))
		# print("approx" + str([approx]))

		# print("[approx][1]: " + str(approx[1][0][1]))

		# if our approximated contour has four points, then we
		# can assume that we have found our screen
		if len(approx) == 4 :
			screenCnt = approx
			break
		else:
			imDraw = drawROI(orig)
			image = imDraw
			orig = image.copy()
			
			ratio = image.shape[0] / 500.0
			
			image = imutils.resize(image, height = 500)
			
			gray = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
			gray = cv2.GaussianBlur(gray, (5, 5), 0)
			
			edged1 = cv2.Canny(gray, 75, 200)
			
			cnts1 = cv2.findContours(edged1.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
		
			cnts1 = imutils.grab_contours(cnts1)
			cnts1 = sorted(cnts1, key = cv2.contourArea, reverse = True)[:5]
			
			for c1 in cnts1:
				peri1 = cv2.arcLength(c1, True) # chu vi			
				approx1 = cv2.approxPolyDP(c1, 0.02 * peri1, True)
				print(">>> approx1: " + str(len(approx1)))

				cv2.drawContours(image, [approx1], 0, (0,255,0), 2)
				cv2.imshow("image: ",  image)
				cv2.waitKey(0)

				point = len(approx1)

				if len(approx1) == point :
					screenCnt = approx1
					# break
				break
			break
	
	# show the contour (outline) of the piece of paper
	print(">>> screenCnt: " + str(screenCnt))
	print("STEP 2: Find contours of paper")
	cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)
	# cv2.imshow("Outline", image)
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()
	scanImage = scanFromPoint(screenCnt.reshape(point, 2) * ratio, orig, 650)
	cv2.waitKey(0)
	return scanImage
	
def scanFromPoint(pts, image, height):
	# apply the four point transform to obtain a top-down
	# view of the original image
	warped = four_point_transform(image, pts)

	# convert the warped image to grayscale, then threshold it
	# to give it that 'black and white' paper effect
	# warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
	warped = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)
	# T = threshold_local(warped, 11, offset = 10, method = "gaussian")
	# warped = (warped > T).astype("uint8") * 255

	# show the original and scanned images
	print("STEP 3: Apply perspective transform")
	# cv2.imshow("Original", imutils.resize(image, height = 650))
	cv2.imshow("Scanned", imutils.resize(warped, height = height))
	
	return imutils.resize(warped, height = height)

def order_points(pts):
	# initialzie a list of coordinates that will be ordered
	# such that the first entry in the list is the top-left,
	# the second entry is the top-right, the third is the
	# bottom-right, and the fourth is the bottom-left
	rect = np.zeros((4, 2), dtype = "float32")

	# the top-left point will have the smallest sum, whereas
	# the bottom-right point will have the largest sum
	s = pts.sum(axis = 1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]

	# now, compute the difference between the points, the
	# top-right point will have the smallest difference,
	# whereas the bottom-left will have the largest difference
	diff = np.diff(pts, axis = 1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]

	# return the ordered coordinates
	return rect

def four_point_transform(image, pts):
	# obtain a consistent order of the points and unpack them
	# individually
	rect = order_points(pts)
	(tl, tr, br, bl) = rect

	# compute the width of the new image, which will be the
	# maximum distance between bottom-right and bottom-left
	# x-coordiates or the top-right and top-left x-coordinates
	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	maxWidth = max(int(widthA), int(widthB))

	# compute the height of the new image, which will be the
	# maximum distance between the top-right and bottom-right
	# y-coordinates or the top-left and bottom-left y-coordinates
	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	maxHeight = max(int(heightA), int(heightB))

	# now that we have the dimensions of the new image, construct
	# the set of destination points to obtain a "birds eye view",
	# (i.e. top-down view) of the image, again specifying points
	# in the top-left, top-right, bottom-right, and bottom-left
	# order
	dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype = "float32")

	# compute the perspective transform matrix and then apply it
	M = cv2.getPerspectiveTransform(rect, dst)
	warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

	# return the warped image
	return warped

# ############################# Xoay ảnh (nhưng cái này không dùng) #####################
def textSkewCorrection(pathImg):
	image = cv2.imread(pathImg)
	image = cv2.resize(image,(500, 550))
	# convert the image to grayscale and flip the foreground
	# and background to ensure foreground is now "white" and
	# the background is "black"
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	gray = cv2.bitwise_not(gray)
	
	# threshold the image, setting all foreground pixels to
	# 255 and all background pixels to 0
	thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

	# grab the (x, y) coordinates of all pixel values that
	# are greater than zero, then use these coordinates to
	# compute a rotated bounding box that contains all
	# coordinates
	coords = np.column_stack(np.where(thresh > 0))
	angle = cv2.minAreaRect(coords)[-1]
	
	# the `cv2.minAreaRect` function returns values in the
	# range [-90, 0); as the rectangle rotates clockwise the
	# returned angle trends to 0 -- in this special case we
	# need to add 90 degrees to the angle
	if angle < -45:
		angle = -(90 + angle)
	
	# otherwise, just take the inverse of the angle to make
	# it positive
	else:
		angle = -angle


	# rotate the image to deskew it
	(h, w) = image.shape[:2]
	center = (w // 2, h // 2)
	M = cv2.getRotationMatrix2D(center, angle, 1.0)
	rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
	# draw the correction angle on the image so we can validate it
	cv2.putText(rotated, "Angle: {:.2f} degrees".format(angle),
		(10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
	
	# show the output image
	print("[INFO] angle: {:.3f}".format(angle))
	cv2.imshow("Input", image)
	cv2.imshow("Rotated", rotated)
	cv2.waitKey(0)


# ############################# Tự động vẽ hình theo viền slide (ngũ giác, lục giác,... nói chung tự do theo mét viề) #######################
def drawROI(im):
	gaus = cv2.GaussianBlur(im, (5, 5), 1)
	# mask1 = cv2.dilate(gaus, np.ones((15, 15), np.uint8, 3))
	mask2 = cv2.erode(gaus, np.ones((5, 5), np.uint8, 1))
	imgray = cv2.cvtColor(mask2, cv2.COLOR_BGR2GRAY)
	ret, thresh = cv2.threshold(imgray, 127, 255, 0)
	contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

	maxArea1=0
	maxI1=0

	for i in range(len(contours)):
		area = cv2.contourArea(contours[i])
		epsilon = 0.01 * cv2.arcLength(contours[i], True)
		approx = cv2.approxPolyDP(contours[i], epsilon, True)
		if area > maxArea1 :
			maxArea1 = area

	print(maxArea1)
	print(maxI1)
	print("[drawROI] approx: " + str(len(approx)))
	cv2.drawContours(im, contours, maxI1, (0,255,255), 3)

	cv2.imshow("yay",im)
	# cv2.imshow("gray",imgray)
	cv2.waitKey(0)

	# cv2.destroyAllWindows()
	return im

################################# Mình phải vẽ viền theo hình chữ nhật ##############
def drawROIRectangle(image):
	# Select ROI
	r = cv2.selectROI(image)
	# Crop image
	# im_crop = image[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
	
	fourpoint = [
		[[int(r[0]) , int(r[1]+r[3])]],
		[[int(r[0]+r[2]) ,int(r[1]+r[3])]],
		[[int(r[0]+r[2]) , int(r[1])]],
		[[int(r[0]) , int(r[1])]],
	]
	screenCnt = np.array(fourpoint, np.int32)
	return screenCnt