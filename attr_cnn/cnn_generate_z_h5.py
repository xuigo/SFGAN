""" temporary script to transform samples from pkl to images """

import os
import glob
import pickle
import numpy as np
import PIL.Image
import h5py
import json


def get_z(generate_Img,json_file):
    #path_gan_sample_img = '../data/trained_img_128'
    file_pattern_x = '*.jpg'
    list_pathfile_x = glob.glob(os.path.join(generate_Img, file_pattern_x))
    list_pathfile_x.sort()
    list_z = []
    with open(json_file ,'r') as f:
        info= json.load(f)   
    for file in list_pathfile_x:        
        filepath,filename=os.path.split(file)
        z=info[filename]
        list_z.append(z)         
    list_z_=np.array(list_z)
    return list_z_