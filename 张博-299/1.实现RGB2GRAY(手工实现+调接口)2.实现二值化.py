import numpy as np
from skimage.color import rgb2gray
import cv2
import matplotlib.pyplot as plt
from PIL import Image
img=cv2.imread('lenna.png')
h,w =img.shape[:2]
img_gray = np.zeros([h,w],img.dtype)
for i in range(h):
    for j in range(w):
        m =img[i,j]
        img_gray[i,j]=int(m[0]*0.11 +m[1]*0.59+m[0]*0.3)
img_binary=np.where(img_gray >= 0.5,1,0)
print('-----img_binary----')
print(img_binary)
print(img_binary.shape)