import cv2
import numpy as np

img = cv2.imread("D:\\Images\\hand1.jpg")
img = cv2.resize(img, (600,700))
img1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

blur = cv2.medianBlur(img1, 9)
ret,thresh = cv2.threshold(blur,240,255,cv2.THRESH_BINARY_INV)

cnts,hier = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print("Total contours:",len(cnts))

#cv2.drawContours(img, cnts, -1, (50,50,150),2)

for c in cnts:
    epsilon = 0.0001*cv2.arcLength(c, True)
    data = cv2.approxPolyDP(c, epsilon, True)
    
    hull = cv2.convexHull(data)
    
    cv2.drawContours(img,[c],-1,(50,50,150),2)
    cv2.drawContours(img,[hull],-1,(0,255,0),2)

hull2 = cv2.convexHull(cnts[0],returnPoints=False)

defect = cv2.convexityDefects(cnts[0], hull2)

for i in range(defect.shape[0]):
    s,e,f,d = defect[i,0]    
    print(s,e,f,d)
    start = tuple(c[s][0])
    end = tuple(c[e][0])
    far = tuple(c[f][0])
    
    cv2.circle(img, far, 5, [0,0,255],-1)

c_max = max(cnts, key = cv2.contourArea)
    
exLeft = tuple(c_max[c_max[:,:,0].argmin()][0])
exRight = tuple(c_max[c_max[:,:,0].argmax()][0])
exTop = tuple(c_max[c_max[:,:,1].argmin()][0])
exBottom = tuple(c_max[c_max[:,:,1].argmax()][0])

cv2.circle(img, exLeft, 8, (255,0,255),-1)
cv2.circle(img, exRight, 8, (0,125,255),-1)
cv2.circle(img, exTop, 8, (255,10,0),-1)
cv2.circle(img, exBottom, 8, (19,152,152),-1)


cv2.imshow("Original", img)
cv2.imshow("Gray", img1)
cv2.imshow("Threshold", thresh)

cv2.waitKey(0)
cv2.destroyAllWindows()
