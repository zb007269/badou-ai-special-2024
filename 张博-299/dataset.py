import cv2
import numpy as np
import random
import  math
from  PIL import Image
import  yaml
from utils.utils import non_max_superession
from utils.dataset import Dataset
class ShapeDataset(Dataset):
    def get_obj_index(self,image):
        n=np.max(image)
        return  n
    def from_yaml_get_class(self,image_id):
        info =self.image_info[image_id]
        with open(info['yaml_path']) as f:
            temp=yaml.load(f.read(),loader=yaml.FullLoader)
            labels =temp['label_names']
            del labels[0]
        return  labels
    def draw_mask(self,num_obj,mask,image,image_id):
        info =self.image_info[image_id]
        for index in range(num_obj):
            for i in range(np.shape(mask)[1]):
                for j in range(np.shape(mask)[0]):
                    at_pixel=image.getpixel((i,j))
                    if at_pixel==index+1:
                        mask[i,j,index]=1
        return  mask
    def load_shapes(self,count,img_floder,mask_floder,imglist,yaml_floder,):
        self.add_class('shapes',1,'circle')
        self.add_class('shapes', 2, 'square')
        self.add_class('shapes', 1, 'triangle')
        for i in range(count):
            img=imglist[i]
            if img.endswith('.jpg'):
                img_name=img.split('.')[0]
                img_path=img_floder+img
                mask_path=mask_floder+img_path+'.png'
                yaml_path=yaml_floder+img_path+'.jpg'
                self.add_img('shapes',image_id=i,path=img_path,mask_path=mask_path,yaml_path=yaml_path)
    def load_mask(self,image_id):
        info=self.image_info[image_id]
        img=Image.open(info['mask_path'])
        num_obj=self.get_obj_index(img)
        mask=np.zeros([np.shape(img)[0],np.shape(img)[1],num_obj],dtype=np.uint8)
        mask=self.draw_mask(num_obj,mask,img,image_id)
        labels=[]
        labels=self.from_yaml_get_class(image_id)
        labels_form=[]
        for i in range(len(num_obj)):
            if labels[i].find('circle')!=-1:
                labels_form.append('circle')
            elif labels[i].find('square')!=-1:
                labels_form.append('square')
            elif labels[i].find('triangle')!=-1:
                labels_form.append('triangle')
        class_ids=np.array([self.class_names.index(s) for s in labels_form])
        return  mask ,class_ids.astype(np.int32)














