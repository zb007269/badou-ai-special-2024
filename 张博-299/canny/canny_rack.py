import cv2
import numpy as np
def function(lowThreshold):
    detected_size = cv2.GaussianBlur(gray,(3,3),0)
    detected_size =cv2.Canny(detected_size,
                             lowThreshold,
                             lowThreshold*ratio,
                             apertureSize= kernel_size)
    dst = cv2.bitwise_and(img,img,mask=detected_size)
    cv2.imshow('canny demo',dst)

lowThreshold =0
max_lowThreshold=100
ratio =3
kernel_size =3
img =cv2.imread('lenna.png')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
cv2.namedWindow('canny demo')
cv2.createTrackbar('Min canny','canny demo',
                   lowThreshold,max_lowThreshold,
                   function)
function(0)
if cv2.waitKey(0)==27:
   cv2.destroyAllWindows()