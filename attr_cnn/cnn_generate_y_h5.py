""" predict_feature labels of synthetic_images """

import os
import glob
import numpy as np
import PIL.Image
import h5py
import utils.cnn_face_attr as cnn_face
import sys

def get_y(generate_Img,weight_path):
    #path_gan_sample_img = '../data/trained_img_128'
    file_pattern_x = '*.jpg'
    list_pathfile_x = glob.glob(os.path.join(generate_Img, file_pattern_x))
    list_pathfile_x.sort()

    """ load model for prediction """
    model = cnn_face.create_cnn_model()
    model.load_weights(weight_path)
    list_y = []
    batch_size = 64
    list_img_batch = []
    list_pathfile_x_use = list_pathfile_x
    num_use = len(list_pathfile_x_use)
    
    save_every = 2048
    
    for i, pathfile_x in enumerate(list_pathfile_x_use):         
        img = np.asarray(PIL.Image.open(pathfile_x))
        list_img_batch.append(img)
        sys.stdout.write( "File transfer progress :[%3d] / [%d] complete!\r" % (i,num_use) )
        sys.stdout.flush()
    img_batch = np.stack(list_img_batch, axis=0)
    x = cnn_face.preprocess_input(img_batch)
    y = model.predict(x, batch_size=batch_size) 
    y_=np.array(y)
    list_y.append(y_)
    list_y_=np.array(list_y)    
    list_y_ = np.concatenate(list_y_, axis=0) 
    return list_y_      