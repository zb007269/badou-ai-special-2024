import cv2
import numpy as np
img =cv2.imread('lenna.png')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
sift = cv2.SIFT_create()
keypoints,descrptior = sift.detectAndCompute(gray,None)
cv2.drawKeypoints(image=img,outImage=img,keypoints=keypoints,
                  flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
                  color=(51,163,236))
cv2.imshow('sift_keypoints',img)
cv2.waitKey(0)
cv2.destroyAllWindows()