#-*- coding:utf-8 -*-
import numpy as np
import dicom
from PIL import Image
import PIL
import itertools
import random
import glob , os
import matplotlib.pyplot as plt
from collections import deque

def make_numpy(save_path, *root_paths):
    for i,root_path in enumerate(root_paths):
        paths=glob.glob(os.path.join(root_path, 'IMG*'))
        #print len(paths)
        size_set = []
        images = []
        for path in paths:
            image = np.asarray(dicom.read_file(path).pixel_array)
            #print np.shape(image)
            image = Image.fromarray(image)
            image = image.resize((23, 23), PIL.Image.ANTIALIAS)
            images.append(np.asarray(image))
            # size_set.append(np.shape(image))
        # print set(size_set)
        if i ==0:
             ret_images = np.asarray(images)  # c_image = cancer image
        else:
            ret_images=np.vstack((ret_images , np.asarray(images)))
    np.save(save_path , images)
    return images
"""
items = deque([1, 2])
    items.append(3)  # deque == [1, 2, 3]
    items.rotate(1)  # The deque is now: [3, 1, 2]
    items.rotate(-1)  # Returns deque to original state: [1, 2, 3]
    item = items.popleft()  # deque == [2, 3]
"""
if '__main__' == __name__ :
    """type3"""

    items = np.arange(1, 7)

    for i in range(5):
        make_numpy('./data/type3/n_images_' + str(i) + '_train.npy',
                   './AI_region/type3/0/' + str(items[1]) + '/DCM000',
                   './AI_region/type3/0/' + str(items[2]) + '/DCM000',
                   './AI_region/type3/0/' + str(items[3]) + '/DCM000',
                   './AI_region/type3/0/' + str(items[4]) + '/DCM000',
                    './AI_region/type3/0/' + str(items[5]) + '/DCM000')

        make_numpy('./data/type3/c_images_' + str(i) + '_train.npy',
                   './AI_region/type3/1/' + str(items[1]) + '/DCM000',
                   './AI_region/type3/1/' + str(items[2]) + '/DCM000',
                   './AI_region/type3/1/' + str(items[3]) + '/DCM000',
                   './AI_region/type3/1/' + str(items[4]) + '/DCM000',
                   './AI_region/type3/1/' + str(items[5]) + '/DCM000',)
        print items[1:]
        items = np.roll(items, -1)

    """type4"""
    items = np.arange(1, 10)
    for i in range(8):
        """make test data"""
        make_numpy('./data/type4/n_images_' + str(i) + 'test.npy','./AI_region/type4/0/' + str(i+1) + '/DCM000')
        make_numpy('./data/type4/c_images_' + str(i) + 'test.npy', './AI_region/type4/1/' + str(i + 1) + '/DCM000')
        """make train data"""
        make_numpy('./data/type4/n_images_' + str(i) + '_train.npy',

                   './AI_region/type4/0/' + str(items[1]) + '/DCM000',
                   './AI_region/type4/0/' + str(items[2]) + '/DCM000',
                   './AI_region/type4/0/' + str(items[3]) + '/DCM000',
                   './AI_region/type4/0/' + str(items[4]) + '/DCM000',
                   './AI_region/type4/0/' + str(items[5]) + '/DCM000',
                   './AI_region/type4/0/' + str(items[6]) + '/DCM000',
                   './AI_region/type4/0/' + str(items[7]) + '/DCM000',
                    './AI_region/type4/0/' + str(items[8]) + '/DCM000',)
        make_numpy('./data/type4/c_images_' + str(i) + '_train.npy',
                   './AI_region/type4/1/' + str(items[1]) + '/DCM000',
                   './AI_region/type4/1/' + str(items[2]) + '/DCM000',
                   './AI_region/type4/1/' + str(items[3]) + '/DCM000',
                   './AI_region/type4/1/' + str(items[4]) + '/DCM000',
                   './AI_region/type4/1/' + str(items[5]) + '/DCM000',
                   './AI_region/type4/1/' + str(items[6]) + '/DCM000',
                   './AI_region/type4/1/' + str(items[7]) + '/DCM000',
                   './AI_region/type4/1/' + str(items[8]) + '/DCM000',)
        print items[1:]
        items = np.roll(items, -1)






