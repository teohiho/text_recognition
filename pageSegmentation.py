# import the necessary packages
import argparse
import os
import time

# https://stackoverflow.com/questions/47260277/cluster-bounding-boxes-and-draw-line-on-them-opencv-python
import cv2
import numpy as np
import tesserocr as tr
from PIL import Image
from scanFixed import scan
from text_detection import text_detection
from word import word



# #################################### PAGE SEGMENTATION ################################# 
# def pageSegmentation(image):
#     cv_img = cv2.imread(image)
    
#     cv_img = scan(cv_img)

#     # filename1 = './images/croped/warpedabc.jpg' 
#     # cv2.imwrite(filename1, cv_img)

#     # since tesserocr accepts PIL images, converting opencv image to pil
#     pil_img = Image.fromarray(cv2.cvtColor(cv_img,cv2.COLOR_BGR2RGB))

#     #initialize api
#     api = tr.PyTessBaseAPI()
#     try:
#         # set pil image for ocr
#         api.SetImage(pil_img)
#         # Google tesseract-ocr has a page segmentation methos(psm) option for specifying ocr types
#         # psm values can be: block of text, single text line, single word, single character etc.
#         # api.GetComponentImages method exposes this functionality
#         # function returns:
#         # image (:class:`PIL.Image`): Image object.
#         # bounding box (dict): dict with x, y, w, h keys.
#         # block id (int): textline block id (if blockids is ``True``). ``None`` otherwise.
#         # paragraph id (int): textline paragraph id within its block (if paraids is True).
#         # ``None`` otherwise.
#         boxes = api.GetComponentImages(tr.RIL.TEXTLINE,True)
#         # get text
#         text = api.GetUTF8Text()
#         # iterate over returned list, draw rectangles
#         for (im,box,_,_) in boxes:
#             x,y,w,h = box['x'],box['y'],box['w'],box['h']
#             # cv2.rectangle(cv_img, (x-25,y-25), (x+w+25,y+h+25), color=(0,0,255))
            
#             imgSegmentation = cv_img[y-25:y+h+35, x:x+w]
#             # print("imgSegmentation.shape[:2] : " + str(imgSegmentation.shape[:2]))
            
#             filename = './images/croped/savedImage-' + str((x+y)) + '.jpg'
#             cv2.imwrite(filename, imgSegmentation)

#             # cv2.imshow("imgSegmentation", imgSegmentation)
#             # cv2.waitKey(0)

#             return imgSegmentation
#     finally:
#         api.End()

# #################################################################

# def pageSegmentation(image):
#     large = cv2.imread(image)
#     large = scan(large)

#     # large = cv2.imread("./images/warpedabcxyz.jpg")
#     rgb = cv2.pyrDown(large)
#     small = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)

#     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
#     kernel = np.ones((5, 5), np.uint8)
#     grad = cv2.morphologyEx(small, cv2.MORPH_GRADIENT, kernel)

#     _, bw = cv2.threshold(grad, 0.0, 255.0, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

#     kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1))
#     connected = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)

#     # using RETR_EXTERNAL instead of RETR_CCOMP
#     contours, hierarchy = cv2.findContours(connected.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     #For opencv 3+ comment the previous line and uncomment the following line
#     #_, contours, hierarchy = cv2.findContours(connected.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

#     mask = np.zeros(bw.shape, dtype=np.uint8)

#     for idx in range(len(contours)):
#         x, y, w, h = cv2.boundingRect(contours[idx])
#         mask[y:y+h, x:x+w] = 0
#         cv2.drawContours(mask, contours, idx, (255, 255, 255), -1)
#         r = float(cv2.countNonZero(mask[y:y+h, x:x+w])) / (w * h)

#         if r > 0.45 and w > 8 and h > 8:
#             # cv2.rectangle(cv_img, (x-25,y-25), (x+w+25,y+h+25), color=(0,0,255))
#             # cv2.rectangle(rgb, (x, y), (x+w-1, y+h-1), (0, 255, 0), 2)
#             imgSegmentation = rgb[y : y + h , x : x + w]
#             cv2.imshow('imgSegmentation', imgSegmentation)
#             cv2.waitKey(0)
#             return imgSegmentation


#     cv2.imshow('rects', rgb)
#     cv2.waitKey(0)

# #################################### SCAN ################################# 
def pageSegmentation(image):
    imgSegmentation = cv2.imread(image)
    imgScan = scan(imgSegmentation)
    # filename1 = './images/croped/scanImage-' + '.jpg' 
    # cv2.imwrite(filename1, scanImage)
    return imgScan


# #################################### MAIN RECOGNITION ################################# 
def text_recognition_command():
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", type=str,
        help="path to input image")
    ap.add_argument("-east", "--east", type=str, default=os.path.join(os.path.dirname(os.path.realpath(__file__)), 'frozen_east_text_detection.pb'),
        help="path to input EAST text detector")
    ap.add_argument("-c", "--min-confidence", type=float, default=0.5,
        help="minimum probability required to inspect a region")
    ap.add_argument("-w", "--width", type=int, default=320,
        help="resized image width (should be multiple of 32)")
    ap.add_argument("-e", "--height", type=int, default=320,
        help="resized image height (should be multiple of 32)")
    args = vars(ap.parse_args())
    
    # Đáng lẽ là phải Scan rồi cắt từng đoạn như mà Cắt đoạn ra rồi cắt từ thì bị lỗi
    imgSegmentation = pageSegmentation(image=args["image"])
    # Crop Word
    # text_detection(imgSegmentation, east=args["east"], min_confidence=args['min_confidence'], width=args["width"], height=args["height"], )   ## text_detection(image=args["image"], east=args["east"], min_confidence=args['min_confidence'], width=args["width"], height=args["height"], )
    word(imgSegmentation)
    cv2.waitKey(0)

if __name__ == '__main__':
    text_recognition_command()