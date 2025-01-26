import cv2 as cv
import numpy as np

"""
img1 = cv.resize(cv.imread("D:\\Images\\hero1.jpg"),(1024,650))
img2 = cv.resize(cv.imread("D:\\Images\\strom_breaker.jpg"),(600,650))

r,c,ch = img2.shape
print(r,c,ch)

roi = img1[0:r,0:c]

img_gry = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

_, mask = cv.threshold(img_gry, 50, 255, cv.THRESH_BINARY)

mask_inv = cv.bitwise_not(mask)

img1_bg = cv.bitwise_and(roi, roi,mask = mask_inv)

img2_fg = cv.bitwise_and(img2, img2, mask = mask)

res = cv.add(img1_bg,img2_fg)

# cv.imshow("Thor",img1)
cv.imshow("Stormbreaker",img2)
# cv.imshow("roi",roi)

cv.imshow("Step-1 gry==",img_gry)
cv.imshow("Step=2 Mask==",mask)
cv.imshow("Step=3 Mask_inv==",mask_inv)
cv.imshow("Step=4 Mask_bg==",img1_bg)
cv.imshow("Step-5 Mask_fg==",img2_fg)
cv.imshow("Step-6 Result==",res)

final = img1
final[0:r,0:c] = res
cv.imshow("Final",final)

cv.waitKey(0)
cv.destroyAllWindows()
"""
img1 = cv.resize(cv.imread("D:\\Images\\hero1.jpg"),(1024,650))
img2 = cv.resize(cv.imread("D:\\Images\\bro_thor.jpg"),(400,600))

r,c,ch = img2.shape
print(r,c,ch)

roi = img1[0:r,0:c]

img_gry = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

_, mask = cv.threshold(img_gry, 50, 255, cv.THRESH_BINARY)

mask_inv = cv.bitwise_not(mask)

img1_bg = cv.bitwise_and(roi, roi,mask = mask_inv)

img2_fg = cv.bitwise_and(img2, img2, mask = mask)

res = cv.add(img1_bg,img2_fg)

# cv.imshow("Thor",img1)
cv.imshow("Stormbreaker",img2)
# cv.imshow("roi",roi)

cv.imshow("Step-1 gry==",img_gry)
cv.imshow("Step=2 Mask==",mask)
cv.imshow("Step=3 Mask_inv==",mask_inv)
cv.imshow("Step=4 Mask_bg==",img1_bg)
cv.imshow("Step-5 Mask_fg==",img2_fg)
cv.imshow("Step-6 Result==",res)

final = img1
final[0:r,0:c] = res
cv.imshow("Final",final)

cv.waitKey(0)
cv.destroyAllWindows()

