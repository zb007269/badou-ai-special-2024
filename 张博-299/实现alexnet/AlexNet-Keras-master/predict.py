import cv2
import numpy as np
import utils
import keras.backend as K
from model.AlexNet import AlexNet
if __name__ =='__main__':
    model =AlexNet()
    model.load_weights('./logs/h5')
    img = cv2.imread('./Test.jpg')
    img_RGB =cv2.cvtColor(img,cv2.COLOR_BGRA2RGB)
    img_nor =img_RGB/255
    img_nor = np.expand_dims(img_nor,axis=0)
    img_resize =utils.resize_image(img_nor,(224,224))
    print(utils.print_answer(np.argmax(model.predict(img_resize))))
    cv2.imshow('ooo',img)
    cv2.waitKey(0)

