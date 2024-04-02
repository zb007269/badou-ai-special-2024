import cv2
import numpy as np
def function(img,out_dim):
    src_w,src_h,channel =img.shape
    dst_w,dst_h=out_dim[0],out_dim[1]
    print('src_w,src_h',src_w,src_h)
    print('dst_w,dst_h',dst_w,dst_h)
    if src_h==dst_h and src_w==dst_w:
        return img.copy()
    dst_img = np.zeros((dst_w,dst_h,3),dtype=np.uint8)
    scale_x,scale_y = float(src_w)/dst_w,float(src_h)/dst_h
    for i in range(3):
        for dst_x in range(dst_w):
            for dst_y in range(dst_h):
                src_x =(dst_x+0.5)*scale_x-0.5
                src_y =(dst_y + 0.5) * scale_y - 0.5
                src_x0 = int(np.floor(src_x))
                src_x1 = min(src_x0+1,src_w-1)
                src_y0 = int(np.floor(src_y))
                src_y1 = min(src_y0 + 1, src_h - 1)
                temp0 = (src_x1 - src_x) * img[src_y0, src_x0, i] + (src_x - src_x0) * img[src_y0, src_x1, i]
                temp1 = (src_x1 - src_x) * img[src_y1, src_x0, i] + (src_x - src_x0) * img[src_y1, src_x1, i]
                dst_img[dst_y, dst_x, i] = int((src_y1 - src_y) * temp0 + (src_y - src_y0) * temp1)
    return dst_img
if __name__ =='__main__':
    img =cv2.imread('lenna.png')
    dst =function(img,(700,700))
    cv2.imshow("A",dst)
    cv2.waitKey()






