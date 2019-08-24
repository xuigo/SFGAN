import os
import numpy as np
import tensorflow as tf
import PIL.Image
from utils.utils import post_process_generator_output
import cv2
import dnnlib
import dnnlib.tflib as tflib
import pickle
import numpy as np
import json
import sys

tflib.init_tf()
batch_size=30

def _train_data():
    seed=np.random.randint(1, 10000000)
    rnd = np.random.RandomState(seed)
    latents = rnd.randn(batch_size, Gs.input_shape[1])  
    z_data_=np.array(latents)
    return z_data_
           
url = 'https://drive.google.com/uc?id=1MEGjdvVpUsu1jB4zrXZN7Y4kBBOzizDQ'
with dnnlib.util.open_url(url, './model') as f:
    _G, _D, Gs = pickle.load(f)
        
def net():
    data_dir='./data'
    img_save_dir=os.path.join(data_dir,'Img')
    for path_ in [data_dir,img_save_dir]:
        if not os.path.exists(path_):
            os.makedirs(path_)    
    with open(os.path.join(data_dir,'images_lantent.json'),'w') as f:
        data={}
        for i in range(2000):               
            batch_z=_train_data()   
            
            fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True) 
            images_ = Gs.run(batch_z, None, truncation_psi=0.7, randomize_noise=True, output_transform=fmt)
            
            for index in range(len(images_)):       
                save_name='{}_{}.jpg'.format(i,index)
                out_fn = os.path.join(img_save_dir, save_name)              
                PIL.Image.fromarray(images_[index], 'RGB').save(out_fn)
                sys.stdout.writelines('%s: %d / %d  filename: %s'%('batch',i,2000,out_fn) + '\r')
                data[save_name]=batch_z[index].tolist()
        json.dump(data,f) 
                                                     
if __name__ == '__main__':
    net()
    


