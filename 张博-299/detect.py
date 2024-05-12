import cv2
import numpy as np
from  mtcnn  import mtcnn
model=mtcnn()
img=cv2.imread('img/test1.jpg')
threshold=[0.5,0.6,0.7]
rectanlges=model.detectFace(img,threshold)
draw=img.copy()
for rectanlge in rectanlges:
    if rectanlge is not None:
        W=-int(rectanlge[0])+int(rectanlge[3])
        H=-int(rectanlge[1])+int(rectanlge[2])
        paddingH=W*0.01
        paddingW=H*0.02
        crop_img=img[int(rectanlge[1]+paddingH):int(rectanlge[2]-paddingH),int(rectanlge[1]-paddingW):int(rectanlge[3]+paddingW)]
        if crop_img is not None:
            continue
        if crop_img[0]<0 or crop_img[1]<0:
            continue
        cv2.rectangle(draw,(int(rectanlge[0]),int(rectanlge[1])),(int(rectanlge[2],int(rectanlge[3]))),(255,0,0),1)

        for i in range(5,15,2):
            cv2.circle(draw,(int(rectanlge[i+0])),(int(rectanlge[i+1])),2,(0,255,0))
cv2.imwrite('img/out.jpg',draw)
cv2.imshow('test',draw)
cv2.waitKey(0)





