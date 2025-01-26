import cv2
import numpy as np

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

def nothing(x):
    pass

cv2.namedWindow("Color Adjustments",cv2.WINDOW_NORMAL)
cv2.resizeWindow("Color Adjustments", (300,300))
cv2.createTrackbar("Thresh", "Color Adjustments", 0, 255, nothing)

cv2.createTrackbar("Lower_H", "Color Adjustments", 0, 255, nothing)
cv2.createTrackbar("Upper_H", "Color Adjustments", 255, 255, nothing)
cv2.createTrackbar("Lower_S", "Color Adjustments", 0, 255, nothing)
cv2.createTrackbar("Upper_S", "Color Adjustments", 255, 255, nothing)
cv2.createTrackbar("Lower_V", "Color Adjustments", 0, 255, nothing)
cv2.createTrackbar("Upper_V", "Color Adjustments", 255, 255, nothing)

while True:
    _,frame = cap.read()
    frame = cv2.resize(frame, (400,400))
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    lh = cv2.getTrackbarPos("Lower_H", "Color Adjustments")
    uh = cv2.getTrackbarPos("Upper_H", "Color Adjustments")
    ls = cv2.getTrackbarPos("Lower_S", "Color Adjustments")
    us = cv2.getTrackbarPos("Upper_S", "Color Adjustments")   
    lv = cv2.getTrackbarPos("Lower_V", "Color Adjustments")
    uv = cv2.getTrackbarPos("Upper_V", "Color Adjustments")
    
    lower_bound = np.array([lh,ls,lv])
    upper_bound = np.array([uh,us,uv])
    
    mask  = cv2.inRange(hsv, lower_bound, upper_bound)
    
    filtr = cv2.bitwise_and(frame, frame, mask = mask)
    
    mask1 = cv2.bitwise_not(mask)
    m_g = cv2.getTrackbarPos("Thresh","Color Adjustments")
    ret,thresh = cv2.threshold(mask1, m_g, 255, cv2.THRESH_BINARY)
    dilata = cv2.dilate(thresh, (1,1),iterations=6)
    
    cnts,hier = cv2.findContours(mask1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    #frame = cv2.drawContours(frame, cnts, -1, (176,10,15),4)
    
    for c in cnts:
        epsilon = 0.0001*cv2.arcLength(c, True)
        data = cv2.approxPolyDP(c, epsilon, True)
        hull = cv2.convexHull(data)
        cv2.drawContours(frame, [c], -1, (50,50,150),2)
        cv2.drawContours(frame, [hull], -1, (0,255,0),2)
    
    
    cv2.imshow("Thresh", thresh)
    cv2.imshow("Mask", mask)
    cv2.imshow("Filter", filtr)
    cv2.imshow("Result", frame)
    
    key = cv2.waitKey(25) & 0xFF
    if key==27:
        break

cap.release()
cv2.destroyAllWindows()
    
    
    
    
    
    
    
    
    
    
