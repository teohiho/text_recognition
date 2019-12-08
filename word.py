import cv2
import numpy as np
from imutils import contours
from simple import simple
from text_recognition import signalToTheEndOfAWord

def word(img):
    large = img

    # rgb = cv2.pyrDown(large)
    rgb = large
    small = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    grad = cv2.morphologyEx(small, cv2.MORPH_GRADIENT, kernel)

    _, bw = cv2.threshold(grad, 0.0, 255.0, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (6, 6)) #(9, 1)
    connected = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)
    # using RETR_EXTERNAL instead of RETR_CCOMP
    Contours, hierarchy = cv2.findContours(connected.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    #For opencv 3+ comment the previous line and uncomment the following line
    #_, contours, hierarchy = cv2.findContours(connected.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    mask = np.zeros(bw.shape, dtype=np.uint8)

    # Contours = contours.sort_contours(Contours, method="left-to-right")[0]

    Contours.sort(key=lambda x:get_contour_precedence(x, large.shape[1]))


    for idx in range(len(Contours)):
        x, y, w, h = cv2.boundingRect(Contours[idx])
        mask[y:y+h, x:x+w] = 0
        cv2.drawContours(mask, Contours, idx, (255, 255, 255), -1)
        r = float(cv2.countNonZero(mask[y:y+h, x:x+w])) / (w * h)

        if r > 0.45 and w > 9 and h > 9:
            # cv2.rectangle(rgb, (x, y), (x+w-1, y+h-1), (0, 255, 0), 2)
            imgWordCroped = rgb[y:y+h, x:x+w]

            cv2.imshow('imgWordCroped', imgWordCroped)

            # Crop characrer 
            simple(imgWordCroped)
            # Nhận biết đã kết thúc một từ để chèn dấu cách vào
            signalToTheEndOfAWord('end')
            
            # cv2.waitKey(0)
        if cv2.waitKey(20) & 0xFF == 27:
            cv2.destroyAllWindows()
            break
    cv2.waitKey(0)

def get_contour_precedence(contour, cols):
    tolerance_factor = 30
    origin = cv2.boundingRect(contour)
    return ((origin[1] // tolerance_factor) * tolerance_factor) * cols + origin[0]
