import cv2
import numpy as np
def aHash(img):
    img=cv2.resize(img,(8,8),interpolation=cv2.INPAINT_NS)
    gray = cv2.cvtColor(cv2.COLOR_BGR2GRAY)
    s=0
    hash_str=''
    for i in range(8):
        for j in range(8):
            s=s+gray[i,j]
    avg =s/64
    for i in range(8):
        for j in range(8):
            if gray[i,j]>avg:
                hash_str=hash_str+'1'
            else:
                hash_str=hash_str+'0'
    return hash_str
def dHash(img):
    img = cv2.resize(img, (8, 8), interpolation=cv2.INPAINT_NS)
    gray = cv2.cvtColor(cv2.COLOR_BGR2GRAY)
    hash_str=''
    for i in range(8):
        for j in range(8):
            if gray[i,j]>gray[i,j+1]:
                hash_str=hash_str+'1'
            else:
                hash_str = hash_str + '0'
    return hash_str
img1= cv2.imread('lenna.png')
img2= cv2.imread('lenna_noise.png')
hash1= aHash(img1)
hash2=dHash(img2)
print(hash1)
print(hash2)




