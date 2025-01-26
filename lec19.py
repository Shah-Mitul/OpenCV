import cv2 as cv
import numpy as np

cap = cv.VideoCapture(0,cv.CAP_DSHOW)

def nothing():
    pass
cv.namedWindow("Color Adjustments")

cv.createTrackbar("Lower_H", "Color Adjustments", 0, 255, nothing)
cv.createTrackbar("Lower_S", "Color Adjustments", 0, 255, nothing)
cv.createTrackbar("Lower_V", "Color Adjustments", 0, 255, nothing)

cv.createTrackbar("Upper_H", "Color Adjustments", 255, 255, nothing)
cv.createTrackbar("Upper_S", "Color Adjustments", 255, 255, nothing)
cv.createTrackbar("Upper_V", "Color Adjustments", 255, 255, nothing)

while True:
    _,frame = cap.read()
    frame = cv.resize(frame, (400,300))
    
    hsv = cv.cvtColor(frame,cv.COLOR_BGR2HSV)
    
    lh = cv.getTrackbarPos("Lower_H", "Color Adjustments")
    ls = cv.getTrackbarPos("Lower_S", "Color Adjustments")
    lv = cv.getTrackbarPos("Lower_V", "Color Adjustments")
    
    uh = cv.getTrackbarPos("Upper_H", "Color Adjustments")
    us = cv.getTrackbarPos("Upper_S", "Color Adjustments")
    uv = cv.getTrackbarPos("Upper_V", "Color Adjustments")
    
    lower_bound = np.array([lh,ls,lv])
    upper_bound = np.array([uh,us,uv])
    
    mask = cv.inRange(hsv,lower_bound,upper_bound)
    
    res = cv.bitwise_and(frame, frame, mask=mask)
    
    cv.imshow("original",frame)
    cv.imshow("mask",mask)
    cv.imshow("res",res)
    
    key = cv.waitKey(1)
    if key==27:
        break
    
cv.destroyAllWindows()  