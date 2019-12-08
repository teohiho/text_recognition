# https://stackoverflow.com/questions/50431647/how-to-detect-symbols-on-a-image-and-save-it
# https://stackoverflow.com/questions/44601734/cv2-findcontours-not-able-to-detect-contours

import cv2
import numpy as np
from text_recognition import image_color_to_gray_size
from imutils import contours


def simple(img):
    # img = cv2.imread('./images/croped-75.8153573485758274.19087879578598.jpg') 

    (origHeight, origWidth) = img.shape[:2]

    img = cv2.resize(img, (origWidth * 3, origHeight * 3))
    #--- create a blank image of the same size for storing the green rectangles (boundaries) ---
    black = np.zeros_like(img)

    #--- convert your image to grayscale and apply a threshold ---
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    # Nếu giá trị pixel lớn hơn giá trị ngưỡng, nó được gán một giá trị (có thể là màu trắng), nếu không, nó được gán giá trị khác (có thể là màu đen)
    # ret2, th2 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # ret2, th2 = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    
    ret2, th2 = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY_INV)
    

    #--- perform morphological operation to ensure smaller portions are part of a single character ---
    # https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    threshed = cv2.morphologyEx(th2, cv2.MORPH_CLOSE, kernel)
    threshed = cv2.dilate(th2, kernel, iterations=2)

    #--- find contours ---
    # Contours, Hierarchy = cv2.findContours(threshed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # https://docs.opencv.org/master/d4/d73/tutorial_py_contours_begin.html
    # Nếu bạn vượt qua cv.CHAIN_APPROX_NONE, tất cả các điểm biên được lưu trữ. Nhưng thực sự chúng ta cần tất cả các điểm? Ví dụ, bạn đã tìm thấy đường viền của một đường thẳng. Bạn có cần tất cả các điểm trên dòng để đại diện cho dòng đó? Không, chúng tôi chỉ cần hai điểm cuối của dòng đó. Đây là những gì cv.CHAIN_APPROX_SIMPLE làm. Nó loại bỏ tất cả các điểm dư thừa và nén đường viền, do đó tiết kiệm bộ nhớ.
    # cv.CHAIN_APPROX_NONE (734 điểm) và hình ảnh thứ hai hiển thị điểm có cv.CHAIN_APPROX_SIMPLE (chỉ 4 điểm).
    Contours, Hierarchy = cv2.findContours(threshed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    print("Sắp xếp")
    Contours = contours.sort_contours(Contours, method="left-to-right")[0]
    # cv2.drawContours(img, Contours, -1, (0,255,0), 3)


    mask = np.zeros(th2.shape, dtype=np.uint8)


    textRecognition = img.copy()

    checkPositionOfI = 0
    countCharacter = 0
    for idx in range(len(Contours)):
        [X, Y, W, H] = cv2.boundingRect(Contours[idx])
        mask[Y:Y+H, X:X+W] = 0
        cv2.drawContours(mask, Contours, idx, (255, 255, 255), -1)
        r = float(cv2.countNonZero(mask[Y:Y+H, X:X+W])) / (W * H)

        if r > 0.45 and W > 9 and H > 9:
            countCharacter = countCharacter + 1
            if(idx > 0):
                if(cv2.boundingRect(Contours[idx])[0] == cv2.boundingRect(Contours[idx-1])[0]):
                    checkPositionOfI = countCharacter - 1
   
    countCharacter = 0
    for idx in range(len(Contours)):

        #--- select contours above a certain area ---
        # if cv2.contourArea(contour) > 100:

        #--- store the coordinates of the bounding boxes ---
        [X, Y, W, H] = cv2.boundingRect(Contours[idx])


        mask[Y:Y+H, X:X+W] = 0
        cv2.drawContours(mask, Contours, idx, (255, 255, 255), -1)
        r = float(cv2.countNonZero(mask[Y:Y+H, X:X+W])) / (W * H)

        if r > 0.45 and W > 9 and H > 9:
            countCharacter = countCharacter + 1
            #--- cut 
            img_crop_simple = img[Y : Y + H, X : X + W]

            #--- draw those bounding boxes in the actual image as well as the plain blank image ---
            
            # cv2.rectangle(textRecognition, (X, Y), (X + W, Y + H), (0,0,255), 2)
            # cv2.rectangle(black, (X, Y), (X + W, Y + H), (0,255,0), 2)

            cv2.imshow('img_crop_simple', img_crop_simple)

            if  (checkPositionOfI != 0):
                if(countCharacter == checkPositionOfI):
                    image_color_to_gray_size(img_crop_simple, 'yes')      
                elif(countCharacter == checkPositionOfI + 1):
                    image_color_to_gray_size(img_crop_simple, 'double')
                else:
                    image_color_to_gray_size(img_crop_simple, 'no')
            else:
                image_color_to_gray_size(img_crop_simple, 'no')

            
            # cv2.waitKey(0)

    # cv2.imshow('contour', textRecognition)
    # cv2.imshow('black', black)
    cv2.waitKey(0)
    
    cv2.destroyAllWindows()
