
import numpy as np
import random
from numpy import shape
import cv2
def fun(src,percetage):
    NoiseImg = src
    NoiseNum = int(percetage*src.shape[0]*src.shape[1])
    for i in range(NoiseNum):
        randX  = random.randint(0,src.shape[0]-1)
        randY = random.randint(0, src.shape[1] - 1)
        if random.random()<=0.5:
            NoiseImg[randX,randY]=0
        else:
            NoiseImg[randX,randY]=255

    return NoiseImg
img = cv2.imread('lenna.png')
img1 = fun(img,0.2)
img = cv2.imread('lenna.png')
img2 = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
cv2.imshow('A',img1)
cv2.imshow('soucre',img2)
cv2.waitKey()




