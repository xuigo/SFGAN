import os
import glob
import sys
import numpy as np
import pickle
import tensorflow as tf
import PIL
import dnnlib
import dnnlib.tflib as tflib
import io
from PIL import Image
import attr_cnn.utils.feature_axis as feature_axis


from utils.feature_attr_organize import attr as attr_axis_

attr='gender'
save_Path='save_gender'
attr_axis=attr_axis_(attr)

if not os.path.exists(save_Path):
    os.makedirs(save_Path)

if attr not in ['age','gender','glass']:
    raise Exception('Attrbute erro...')
    
path_feature_direction = './attr_cnn/feature_direction'
pathfile_feature_direction =os.path.join(path_feature_direction, 'feature_direction_{}.pkl'.format(attr))

with open(pathfile_feature_direction, 'rb') as f:
    feature_direction_name = pickle.load(f)
    
feature_direction = feature_direction_name['direction']
feature_name = feature_direction_name['name']
num_feature = feature_direction.shape[1]
feature_name,feature_reverse=attr_axis.name_reverse()
feature_direction = feature_direction_name['direction']* feature_reverse[None, :]
tflib.init_tf()
url = 'https://drive.google.com/uc?id=1MEGjdvVpUsu1jB4zrXZN7Y4kBBOzizDQ' 
with dnnlib.util.open_url(url, './model') as f:
    _G, _D, Gs = pickle.load(f)
    
yn_CPU_only = False
if yn_CPU_only:
    config = tf.ConfigProto(device_count = {'CPU': 4}, allow_soft_placement=True)
else:
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    
dict_attr={'age':[1,2],'gender':[0,1],'glass':[0,1]}
index=dict_attr[attr]

def test_data():
    seed=np.random.randint(1, 10000000)
    rnd = np.random.RandomState(seed)
    latents = rnd.randn(1, Gs.input_shape[1])  
    z_data_=np.array(latents)
    return z_data_
    
def generate(x):
    z_sample_copy = test_data()
    
    for info in index:     
        z_sample=z_sample_copy.copy()
        
        id_feature=info
        feature_lock_status = np.zeros(num_feature).astype('bool')
        feature_lock_status[id_feature] = np.logical_not(feature_lock_status[id_feature])
        
        feature_directoion_disentangled = feature_axis.disentangle_feature_axis_by_idx(
            feature_direction, idx_base=np.flatnonzero(feature_lock_status))
        for i in range(5):                               
            z_sample += feature_directoion_disentangled[:, id_feature]*float(i/5.0)
            fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
            #print(z_sample.shape)
            images = Gs.run(z_sample, None, truncation_psi=0.7, randomize_noise=True, output_transform=fmt)    
            im = Image.fromarray(images[0])
            saveImg=os.path.join(save_Path,"{}_{}_{}.jpeg".format(x,feature_name[info],i))
            #print("asset_results/)
            im.save(saveImg)
if __name__=='__main__':
    #print(attrname)
    for i in range(10):
        print('Starting {} ...'.format(i))
        generate(i)           
    