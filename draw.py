import numpy as np
import cv2
import pytesseract
from PIL import Image
import imutils
import math  
from matplotlib import pyplot as plt
from simple import simple

# #################################### DRAW AND CUTTING WORDS ################################# 
def drawPolygons(drawOn, polygons, ratioWidth, ratioHeight, color=(0, 0, 255), width=1):
    # # print("polygons: " + str(len(polygons))) #các khung nhận dạng được trong 1 hình
    np_array=[]
    # cv2.imshow("drawOn", drawOn)
    for polygon in polygons:
        pts = np.array(polygon, np.int32)
        pts = pts.reshape((-1, 1, 2))

        #### draw the polygon
        # img = cv2.polylines(drawOn, [pts], True, color, width)
        ####
        # cv2.imshow("pts", pts)
        # config = ("-l eng --oem 1 --psm 7")
        # text = pytesseract.image_to_string(pts, config=config)

        # print("pts: "+ str([pts][1]))   https://www.aiworkbox.com/lessons/convert-numpy-array-to-mxnet-ndarray
        # print("pts: "+ str([pts(1)]))

        # points = [
        #     (upperLeftX, upperLeftY),
        #     (lowerRightX, upperLeftY),
        #     (lowerRightX, lowerRightY),
        #     (upperLeftX, lowerRightY)
        # ]

        # ===========================
        # Given 4 points, how to crop a quadrilateral from an image in pytorch/torchvision?
        # polygon[0][0] = polygon[0][0] - 2
        polygon[0][1] = polygon[0][1] - 10
        # polygon[1][0] = polygon[1][0] + 5
        polygon[1][1] = polygon[1][1] - 10
        # polygon[2][0] = polygon[2][0] + 5
        polygon[2][1] = polygon[2][1] + 10
        # polygon[3][0] = polygon[3][0] - 2
        polygon[3][1] = polygon[3][1] + 10
        # pts = np.array([[polygon[0][0] - 5, polygon[0][1] - 5], [polygon[1][0] + 5, polygon[1][1] - 5], [polygon[2][0] + 5, polygon[2][1] + 5], [polygon[3][0] - 5, polygon[3][1] + 5]], dtype=np.int32)
        pts = np.array([[polygon[0][0] , polygon[0][1] ], [polygon[1][0] , polygon[1][1]], [polygon[2][0] , polygon[2][1]], [polygon[3][0] , polygon[3][1]]], dtype=np.int32)
        mask = np.zeros((drawOn.shape[0], drawOn.shape[1]))
        cv2.fillConvexPoly(mask, pts, 1)
        mask = mask.astype(np.bool)
        out = np.zeros_like(drawOn)
        out[mask] = drawOn[mask]
        # print("out.shape[:2] : " + str(out.shape[:2])) 
        ### 
        # cv2.imshow('out', out)

        # ===========================
        # # Cắt ảnh từ khung nghiêng
        # # print("polygon[3][1]: "+ str(polygon[3][1])) #246.98870153288397
        # # print("polygon[1][1]: "+ str(polygon[1][1])) #195.3879830750784
        # # print("polygon[3][0]: "+ str(polygon[3][0])) #222.65554937590102
        # # print("polygon[1][0]: "+ str(polygon[1][0])) #360.8632403870683
        # rect_img1 = drawOn[int(polygon[1][1]) : int(polygon[3][1]), int(polygon[3][0]) : int(polygon[1][0])]
        # cv2.imshow('rect_img1', rect_img1)

        # ===========================
        # # Text xoay hình
        # cv2.imshow('drawOn', drawOn)
        # rotated = imutils.rotate_bound(drawOn, 20)
        # cv2.imshow("Rotated (Correct)", rotated)
        #print("polygon[0][1]: "+ str(polygon[0][1])) #195.3879830750784
        
        # rotated = imutils.rotate_bound(out, -20)
        # cv2.imshow("Rotated (Correct)", rotated)
        l = polygon[1][1] - polygon[0][1]
        h = polygon[1][0] - polygon[0][0]
        a = l/h
        do = math.degrees(math.atan(a)) # arctan rồi chuyển radian sang độ
        
        # rotated = imutils.rotate_bound(out, -do)
        rotated = imutils.rotate(out, do)
        

        img_crop = cropImage(rotated, pts, polygon[0][1], polygon[1][1])

        # filename1 = './images/croped/croped-' + str(polygon[0][1]) + str(polygon[1][1])  + '.jpg' 
        # cv2.imwrite(filename1, img_crop)
    
        cv2.imshow("img_crop", img_crop)
        simple(img_crop)


        # print("do: "+ str(-do))
        # print("l: "+ str(l))
        # print("h: "+ str(h))
        # print("a: "+ str(a))
        # print("polygon[0][1] * (662/600): "+ str(polygon[0][1] ))
        # print("polygon[2][1] * (662/600): "+ str(polygon[2][1] ))
        # print("polygon[0][0] * (468/340): "+ str(polygon[0][0] ))
        # print("polygon[2][0] * (468/340): "+ str(polygon[2][0] ))

        # cv2.imshow("Rotated (Correct)", rotated)
        # print("rotated : " + str(rotated))



        # cv2.circle(drawOn, (int(polygon[2][0]), int(polygon[2][1])), 3, (255,0,0), -1)

        # ===========================
        # # ### tạo khung chữ nhật (https://stackoverflow.com/questions/30901019/extracting-polygon-given-coordinates-from-an-image-using-opencv/30902423#30902423)
        # (meanx, meany) = pts.mean(axis=0)
        # (cenx, ceny) = (drawOn.shape[1]/2, drawOn.shape[0]/2)
        # (meanx, meany, cenx, ceny) = np.floor([meanx, meany, cenx, ceny]).astype(np.int32)
        # (offsetx, offsety) = (-meanx + cenx, -meany + ceny)

        # (mx, my) = np.meshgrid(np.arange(drawOn.shape[1]), np.arange(drawOn.shape[0]))
        # ox = (mx - offsetx).astype(np.float32)
        # oy = (my - offsety).astype(np.float32)
        # out_translate = cv2.remap(out, ox, oy, cv2.INTER_LINEAR)
        # topleft = pts.min(axis=0) + [offsetx, offsety]
        # bottomright = pts.max(axis=0) + [offsetx, offsety]
        # print("tuple(topleft) : " + str(topleft[0])) # KQ: (72, 252)(31, 72)(98, 140)
        # cv2.rectangle(out_translate, tuple(topleft), tuple(bottomright), color=(255,0,0))  #https://www.programcreek.com/python/example/89445/cv2.rectangle
        # cv2.imshow("out_translate", out_translate)
        
        # # cắt hình  (https://stackoverflow.com/questions/46795669/python-opencv-how-to-draw-ractangle-center-of-image-and-crop-image-inside-rectan/46803516)
        # rect_img = out_translate[topleft[1] : bottomright[1], topleft[0] : bottomright[0]]
        # cv2.imshow('rect_img', rect_img)

        
        # # cv2.imshow("roi", roi)
        # # cv2.imshow("roi", out)
        # # print("out.shape[:2] : " + str(out.shape[:2]))
        # # print("out: " + str(out.shape))

        
        # =========================== recognition text
        # config = ("-l eng --oem 1 --psm 7")
        # config = ("-l eng --oem 3 --psm 12")
        # text = pytesseract.image_to_string(img_crop, config=config)
        # print("text: " + str(text))
        
        # np_im = np.array(img_crop)
        # np_im = np_im - 18
        # new_im = Image.fromarray(np_im)
        # # new_im.save("numpy_altered_sample2" + str(polygon[1][1] - polygon[0][1]) +".png")
        # np_array.append(new_im)
        cv2.waitKey(0)
        # print("np_array: " + str([np_array]))

        
    # return np_array
        

# #################################### CROP IMAGE ################################# 
def cropImage(rotated, pts, y0, y1):
    cX, cY = findCenterOfBlob(rotated)
    rect = cv2.minAreaRect(pts)
    # print("rect: {}".format(rect))
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    # cv2.drawContours(rotated, [box], 0, (0, 0, 255), 2)
    # img_crop, img_rot = crop_rect(drawOn, rect)
    center, size, angle = rect[0], rect[1], rect[2]
    center, size = tuple(map(int, center)), tuple(map(int, size))
    # cv2.imwrite("cropped_img.jpg", img_crop)
    if(y0 < y1):
        img_crop = cv2.getRectSubPix(rotated, (size[1], size[0]), (cX, cY))
    else:
        img_crop = cv2.getRectSubPix(rotated, size, (cX, cY))
    return img_crop


# #################################### FIND THE CENTER OF THE FIGURE ################################# 
### Tìm điểm trung tâm từ ảnh đã chỉnh thẳng nhưng chưa được cắt
def findCenterOfBlob(img):
   # convert image to grayscale image
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # convert the grayscale image to binary image
    ret,thresh = cv2.threshold(gray_image,127,255,0)
    
    # calculate moments of binary image
    M = cv2.moments(thresh)
    
    # calculate x,y coordinate of center
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    
    # put text and highlight the center
    # cv2.circle(img, (cX, cY), 5, (0, 0, 255), -1)
    # cv2.putText(img, "centroid", (cX - 25, cY - 25),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    # display the image
    # cv2.imshow("Image", img)
    return cX, cY


# #################################### DRAW BOXES ################################# 
# Dùng cho khung thẳng, nhưng ở đây đã chỉnh khung nghiêng theo chữ nên không cần
def drawBoxes(drawOn, boxes, ratioWidth, ratioHeight, color=(0, 255, 0), width=1):

    for(x,y,w,h) in boxes:
        startX = int(x*ratioWidth)
        startY = int(y*ratioHeight)
        endX = int((x+w)*ratioWidth)
        endY = int((y+h)*ratioHeight)

        # draw the bounding box on the image
        cv2.rectangle(drawOn, (startX, startY), (endX, endY), color, width)


# def remove_noise_and_smooth(img):
#     filtered = cv2.adaptiveThreshold(img.astype(np.uint8), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 41)
#     kernel = np.ones((1, 1), np.uint8)
#     opening = cv2.morphologyEx(filtered, cv2.MORPH_OPEN, kernel)
#     closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
#     img = image_smoothening(img)
#     or_image = cv2.bitwise_or(img, closing)
#     return or_image