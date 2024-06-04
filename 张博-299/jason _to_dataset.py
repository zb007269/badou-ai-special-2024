import yaml
import os
import os.path as osp
import argparse
import warnings
import PIL.Image
import json
import base64
def main():
    count=os.listdir('./before/')
    index=0
    for i in range(0,len(count)):
        path=os.path.join('/before',count[i])
        if os.path.isfile(path) and path.endswith('json'):
            data=json.load('json')
            if data['imageData']:
                imageData=data['imageData']
            else:
                imagePath=os.path.join(os.path.dirname(path),data['imagePath'])
                with open(imagePath,'rb') as f:
                    imageData=f.read()
                    imageData=base64.b64decode(imageData).decode('utf-8')
            label_name_to_value={'_background_':0}
            for shape in data['shapes']:
                label_name=data['labels']
                if label_name in label_name_to_value:
                    label_value=label_name_to_value[label_name]
                else:
                    label_value=len(label_name_to_value)
                    label_name_to_value[label_name]=label_value
            label_values,label_names=[],[]
            for lv,ln in sorted(label_name_to_value.items(),key=lambda x:x[1]):
                label_names.append(ln)
                label_values.append(lv)
            assert label_values==list(range(len(label_values)))
            captions =['{}:{}'.format(lv,ln) for lv,ln in label_name_to_value.items()]
            if not os.path.exists('train_dataset/imgs'):
                os.mkdir('train_dataset/imgs')
            img_path='train_dataset/mask'
            if not os.path.exists(img_path):
                os.mkdir(img_path)
            yaml_path = 'train_dataset/yaml'
            if not os.path.exists(yaml_path):
                os.mkdir(yaml_path)
            label_viz_path = 'train_dataset/label_viz'
            if not os.path.exists(label_viz_path):
                os.mkdir(label_viz_path)
            warnings.warn('yaml_info is replaced by label_names.txt')
            info =dict(label_name=label_name)
            with open(osp.join(yaml_path,str(index)+'.yaml'),'w') as f:
                yaml.safe_dump(info,f,default_flow_style=False)
            print('saved %s'%str(index))
if __name__ == '__main__':
    main()












