import os
import glob
import numpy as np
import pickle
import h5py
import pandas as pd


import utils.misc as misc
import utils.feature_axis as feature_axis

from cnn_generate_y_h5 import get_y
from cnn_generate_z_h5 import get_z
##
""" get y and z from pre-generated files """
path_att = '/home/xushaohui/workspace_xsh/mydata/AttrData/FaceAttribute_128/gender/gender2.txt'
path_feature_direction = './feature_direction'
attr='gender'
generate_Img='../data/Img'
json_file='../data/images_lantent.json'
weight_path='./model/model_{}.h5'.format(attr)

if not os.path.exists(path_feature_direction):
    os.makedirs(path_feature_direction)
# read feature name
df_attr = pd.read_csv(path_att, sep='\s+', header=1, index_col=0)
y_name = df_attr.columns.values.tolist()

""" regression: use latent space z to predict features y """
z=get_z(generate_Img,json_file)
y=get_y(generate_Img,weight_path)


print('z-shape:{}  y-shape:{}'.format(z.shape,y.shape))
feature_slope = feature_axis.find_feature_axis(z, y, method='tanh')

##
""" normalize the feature vectors """
yn_normalize_feature_direction = True
if yn_normalize_feature_direction:
    feature_direction = feature_axis.normalize_feature_axis(feature_slope)
else:
    feature_direction = feature_slope

""" save_regression result to hard disk """
if not os.path.exists(path_feature_direction):
    os.mkdir(path_feature_direction)

pathfile_feature_direction = os.path.join(path_feature_direction, 'feature_direction_{}.pkl'.format(attr))
dict_to_save = {'direction': feature_direction, 'name': y_name}
with open(pathfile_feature_direction, 'wb') as f:
    pickle.dump(dict_to_save, f)


##
""" disentangle correlated feature axis """
pathfile_feature_direction = glob.glob(os.path.join(path_feature_direction, 'feature_direction_*.pkl'))[-1]

with open(pathfile_feature_direction, 'rb') as f:
    feature_direction_name = pickle.load(f)

feature_direction = feature_direction_name['direction']
feature_name = np.array(feature_direction_name['name'])

len_z, len_y = feature_direction.shape
print('len_z:{}  len_y:{}'.format(len_z,len_y))
