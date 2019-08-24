import os
import sys


def reshape_celebA(path_to_data,save_path):
    from scipy import misc
    import numpy as np
    from PIL import Image
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    files_read = []
    for root, subFolders, files in os.walk(path_to_data):
        print(root)
        print(subFolders)
        print(len(files))
        for f in files:
            if f.endswith('.jpg') or f.endswith('.png') or f.endswith('.jpeg'):
                files_read.append(os.path.join(root, f))
                # print(files_read[-1])
        print('one subdir done')
    # files = [f for f in os.listdir(path_to_data) if f.endswith('.jpg') or f.endswith('.png') or f.endswith('.jpeg')]
    print('Done listing files')
    images = []
    names=[]
    
    for f in files_read:
        try:
            # im = misc.imread(f)
            im = Image.open(f)
            im = np.array(im)
            # print(im)
        except IOError:
            print('Could not read: %s' % f)
        if len(im.shape) == 2:
            im = np.expand_dims(im, -1)
        filepath,filename=os.path.split(f) 
        num_c = im.shape[-1]
        im=misc.imresize(im, (128, 128, num_c))
        
        misc.imsave(os.path.join(save_path,filename), im)
        #names.append(filename)
        #images.append(im)
        
    print('Done reading files')
    '''num_c = images[0].shape[-1]
    for i in range(len(images)):
        images[i] = misc.imresize(images[i], (128, 128, num_c))
        
        misc.imsave(os.path.join(save_path,names[i]), images[i])
        # if len(images[i].shape) == 3:
        #     images[i] = np.expand_dims(images[i], 0)
        
    data = np.stack(images, axis=0).astype(np.float32)
    np.save(os.path.join(path_to_data, 'celeb_64.npy'), data)'''
    
if __name__=='__main__':
    reshape_celebA('/home/xushaohui/workspace_xsh/mydata/CelebA/img_align_celeba','/home/xushaohui/workspace_xsh/mydata/CelebA/img_align_celeba_128')
    