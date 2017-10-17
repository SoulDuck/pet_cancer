#-*- coding:utf-8 -*-
import numpy as np
import dicom
from PIL import Image
import PIL
import itertools
import random
import glob , os
import matplotlib.pyplot as plt


def make_numpy(root_path , save_path):
    paths=glob.glob(os.path.join(root_path, 'IMG*'))
    print len(paths)
    size_set = []
    images = []
    for path in paths:
        image = np.asarray(dicom.read_file(path).pixel_array)
        image = Image.fromarray(image)
        image = image.resize((23, 23), PIL.Image.ANTIALIAS)
        images.append(np.asarray(image))
        # size_set.append(np.shape(image))
    # print set(size_set)
    images = np.asarray(images)  # c_image = cancer image
    np.save(save_path , images)
    return images

def get_cacner_image():
    path , sub_folders , files =os.walk('./AI_region/1').next()
    paths=map(lambda sub_folder : os.path.join(path , sub_folder) , sub_folders)
    paths=(map(lambda path : os.path.join(path , 'DCM000') , paths))
    file_paths=map(lambda path : glob.glob(os.path.join(path , 'IMG*')) , paths)
    paths=list(itertools.chain(*file_paths)) #[[1,2][3,4]] --> [1,2,3,4]

    size_set = []
    images=[]
    for path in paths:
        image=np.asarray(dicom.read_file(path).pixel_array)
        image=Image.fromarray(image)
        image=image.resize((23,23) , PIL.Image.ANTIALIAS)
        images.append(np.asarray(image))
        #size_set.append(np.shape(image))
    #print set(size_set)
    c_images=np.asarray(images)# c_image = cancer image
    return  c_images
def crop_image(image ,coord,crop_size):
    cropped_image=image[coord[0]:coord[0]+crop_size[0] , coord[1]:coord[1]+crop_size[1]]
    assert np.shape(cropped_image) == crop_size ,np.shape(cropped_image)
    return cropped_image


def get_normal_image():
    path , sub_folders , files =os.walk('./AI_region/0').next()
    paths=map(lambda sub_folder : os.path.join(path , sub_folder) , sub_folders)
    paths = (map(lambda path: os.path.join(path, 'DCM000'), paths))
    file_paths = map(lambda path: glob.glob(os.path.join(path, 'IMG*')), paths)
    paths = list(itertools.chain(*file_paths))  # [[1,2][3,4]] --> [1,2,3,4]

    print '## normal image info ##'
    print '# images : ',len(paths)
    size_set = []
    images = []
    crop_size=(23,23)
    for path in paths:
        try:
            image = np.asarray(dicom.read_file(path).pixel_array)
            image = Image.fromarray(image)
            image = image.resize((23, 23), PIL.Image.ANTIALIAS)
            images.append(np.asarray(image))

            """
            #print np.shape(image)
            h,w=np.shape(image)
            h_ind=random.randint(0,h-crop_size[0]-1)
            w_ind = random.randint(0,w - crop_size[1]-1)
            #print h_ind , w_ind
            assert h_ind < h-crop_size[0] or w_ind < w - crop_size[1]
            image=crop_image(image ,(h_ind , w_ind) , (23,23))
            images.append(np.asarray(image))
            """
        except ValueError as ve:
            print ve
            print path
    images=np.asarray(images)
    return images



def get_images():
    c_images=get_cacner_image()
    n_images=get_normal_image()
    print np.shape(n_images)
    count=0
    n_filename='./data/n_images_'+str(count)+'.npy'
    c_filename = './data/c_images_' + str(count) + '.npy'
    while os.path.isfile(n_filename):
        count+=1
        n_filename='./data/n_images_'+str(count)+'.npy'
        c_filename = './data/c_images_' + str(count) + '.npy'
        print n_filename
        print c_filename
    np.save(n_filename, n_images)
    np.save(c_filename ,c_images)
    return c_images , n_images
#c_images = np.asarray(im
#def get_nomal_images():



def get_train_test_images(c_images , n_images,c_test=23 ,n_test=22 , random_shuffle=False):
    #assert len(c_images) < len(n_images) #이걸 왜 만들어 놓은거지?? 어떤 목적으로
    #c_images, n_images = get_images()
    n = len(c_images)
    if random_shuffle:
        random.seed(12)
        c_indices = random.sample(range(len(c_images)), n)
        n_indices = random.sample(range(len(n_images)), n)
    else:
        c_indices = range(n)
        n_indices = range(len(n_images))
    c_test_imgs = c_images[c_indices[:c_test]]
    c_train_imgs = c_images[c_indices[c_test:]]
    c_test_cls=np.zeros(len(c_test_imgs))
    c_train_cls=np.zeros(len(c_train_imgs))

    n_test_imgs = n_images[n_indices[:n_test]]
    n_train_imgs = n_images[n_indices[n_test:]]
    n_test_cls=np.ones(len(n_test_imgs))
    n_train_cls=np.ones(len(n_train_imgs))



    train_imgs=np.concatenate((c_train_imgs , n_train_imgs ), axis=0)
    train_cls = np.concatenate((c_train_cls, n_train_cls), axis=0)
    test_imgs = np.concatenate((c_test_imgs, n_test_imgs), axis=0)
    test_cls = np.concatenate((c_test_cls, n_test_cls), axis=0)


    train_imgs=train_imgs.reshape((-1,23,23,1))
    test_imgs = test_imgs.reshape((-1, 23, 23, 1))
    print np.shape(train_imgs)
    print np.shape(test_imgs)
    assert len(train_imgs) == len(train_cls) and len(test_imgs) == len(test_cls)
    return train_imgs , train_cls, test_imgs, test_cls






def get_type0_image():
    #[22,48,62,33,20]
    path , sub_folders , files =os.walk('./AI_region/0').next()
    print sub_folders
    paths=map(lambda sub_folder : os.path.join(path , sub_folder) , sub_folders)
    paths=(map(lambda path : os.path.join(path , 'DCM000') , paths))
    file_paths=map(lambda path : glob.glob(os.path.join(path , 'IMG*')) , paths)
    paths=list(itertools.chain(*file_paths)) #[[1,2][3,4]] --> [1,2,3,4]

    size_set = []
    images=[]
    for path in paths:
        image=np.asarray(dicom.read_file(path).pixel_array)
        image=Image.fromarray(image)
        image=image.resize((23,23) , PIL.Image.ANTIALIAS)
        images.append(np.asarray(image))
    images = np.asarray(images)
    return images

def get_type1_image():
    #[23,26,27,26,14]
    path , sub_folders , files =os.walk('./AI_region/1').next()
    print sub_folders
    paths=map(lambda sub_folder : os.path.join(path , sub_folder) , sub_folders)
    paths=(map(lambda path : os.path.join(path , 'DCM000') , paths))
    file_paths=map(lambda path : glob.glob(os.path.join(path , 'IMG*')) , paths)
    paths=list(itertools.chain(*file_paths)) #[[1,2][3,4]] --> [1,2,3,4]

    size_set = []
    images=[]
    for path in paths:
        image=np.asarray(dicom.read_file(path).pixel_array)
        image=Image.fromarray(image)
        image=image.resize((23,23) , PIL.Image.ANTIALIAS)
        images.append(np.asarray(image))
    images=np.asarray(images)
    return images




if '__main__' == __name__ :

    make_numpy('./AI_region/0/1/DCM000' , './data/n_images_0_test.npy')
    make_numpy('./AI_region/0/2/DCM000', './data/n_images_1_test.npy')
    make_numpy('./AI_region/0/3/DCM000', './data/n_images_2_test.npy')
    make_numpy('./AI_region/0/4/DCM000', './data/n_images_3_test.npy')
    make_numpy('./AI_region/0/5/DCM000', './data/n_images_4_test.npy')
    make_numpy('./AI_region/0/none1/DCM000', './data/n_images_5_test.npy')
    make_numpy('./AI_region/0/none2/DCM000', './data/n_images_6_test.npy')
    make_numpy('./AI_region/0/none3/DCM000', './data/n_images_7_test.npy')
    make_numpy('./AI_region/0/none4/DCM000', './data/n_images_8_test.npy')
    make_numpy('./AI_region/0/none5/DCM000', './data/n_images_9_test.npy')

    make_numpy('./AI_region/1/1/DCM000', './data/c_images_0_test.npy')
    make_numpy('./AI_region/1/2/DCM000', './data/c_images_1_test.npy')
    make_numpy('./AI_region/1/3/DCM000', './data/c_images_2_test.npy')
    make_numpy('./AI_region/1/4/DCM000', './data/c_images_3_test.npy')
    make_numpy('./AI_region/1/5/DCM000', './data/c_images_4_test.npy')
    make_numpy('./AI_region/1/meta1/DCM000', './data/c_images_5_test.npy')
    make_numpy('./AI_region/1/meta2/DCM000', './data/c_images_6_test.npy')
    make_numpy('./AI_region/1/meta3/DCM000', './data/c_images_7_test.npy')
    make_numpy('./AI_region/1/meta4/DCM000', './data/c_images_8_test.npy')
    make_numpy('./AI_region/1/meta5/DCM000', './data/c_images_9_test.npy')

