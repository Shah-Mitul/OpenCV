import cv2
import numpy as np

img = cv2.imread("D:\\Images\\green.jpg")
img = cv2.resize(img, (600, 650))
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

def nothing(x):
    pass

cv2.namedWindow("Color Adjustments",cv2.WINDOW_NORMAL)
cv2.resizeWindow("Color Adjustments", (300,300))
cv2.createTrackbar("Thresh", "Color Adjustments", 0, 255, nothing)

roi = cv2.imread("D:\\Images\\g.jpg")
hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

while True:
    roi_hist = cv2.calcHist([hsv_roi], [0, 1], None,[180, 256], [0, 180, 0, 256], 1)
    mask = cv2.calcBackProject([hsv], [0,1], roi_hist, [0,180,0,256], 1)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
    mask = cv2.filter2D(mask, -1, kernel)


    min1 = cv2.getTrackbarPos("Thresh","Color Adjustments")
    _,mask = cv2.threshold(mask, min1,255, cv2.THRESH_BINARY)

    mask = cv2.merge((mask,mask,mask))
    result = cv2.bitwise_or(img, mask)


    cv2.imshow("img", img)
    cv2.imshow("hsv", hsv)
    cv2.imshow("mask", mask)
    cv2.imshow("res", result)
    
    key = cv2.waitKey(25) & 0xFF
    if key==27:
        break

cv2.destroyAllWindows()
