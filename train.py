# -*- coding: utf-8 -*-
import tensorflow as tf
from cnn import convolution2d, max_pool, algorithm, affine, batch_norm_layer, gap
import inception_v4
import input
import numpy as np
import utils
from inception_v4 import stem, stem_1, stem_2, reductionA, reductionB, blockA, blockB, blockC
import random
import argparse
import os
import time
import aug
import shutil
n_test=30
#c_images , n_images=input.get_images()
def train(max_iter ,learning_rate , structure, optimizer,src_root_dir , file_idx  ,restored_model_folder_path , save_log_path):



    c_images_train = np.load(os.path.join(src_root_dir , 'c_images_'+file_idx+'_train.npy'))
    n_images_train = np.load(os.path.join(src_root_dir , 'n_images_'+file_idx+'_train.npy'))
    c_images_test = np.load(os.path.join(src_root_dir , 'c_images_'+file_idx+'_test.npy'))
    n_images_test = np.load(os.path.join(src_root_dir , 'n_images_'+file_idx+'_test.npy'))
    print 'cacner (label :1 )train images shape:',np.shape(c_images_train)
    print 'normal (label :0 )train images shape:',np.shape(n_images_train)
    print 'cacner (label :1 )test images shape:',np.shape(c_images_test)
    print 'normal (label :0 )test images shape:',np.shape(n_images_test)
    n_cls_train=np.zeros(len(n_images_train))
    c_cls_train=np.ones(len(c_images_train))
    n_cls_test = np.zeros(len(n_images_test))
    c_cls_test = np.ones(len(c_images_test))

    h = 23
    w = 23
    ch = 1
    n_classes = 2

    train_imgs = np.vstack((c_images_train, n_images_train))
    test_imgs = np.vstack((c_images_test, n_images_test))
    train_cls = np.hstack((c_cls_train , n_cls_train))
    test_cls = np.hstack((c_cls_test, n_cls_test))
    if train_imgs.ndim == 3:
        train_imgs=train_imgs.reshape([-1 , h,w,ch])
    if test_imgs.ndim == 3:
        test_imgs = test_imgs.reshape([-1 , h,w,ch])
    print "## input data info ##"
    print 'train images : ' , np.shape(train_imgs)
    print 'train cls : ', np.shape(train_cls)
    print 'test images : ' , np.shape(test_imgs)
    print 'test cls : ', np.shape(test_cls)



    x_ = tf.placeholder(dtype=tf.float32, shape=[None, h, w, ch], name='x_')
    y_cls = tf.placeholder(dtype=tf.int32, shape=[None], name='y_')
    y_ = tf.one_hot(y_cls , 2)
    phase_train = tf.placeholder(dtype=tf.bool, name='phase_train')
    ##########################structure##########################
    if structure == 'inception_A':
        top_conv=inception_v4.structure_A(x_)
    elif structure == 'inception_B':
        top_conv = inception_v4.structure_B(x_ , phase_train)
    elif structure == 'simple_cnn':
        top_conv = inception_v4.simple_cnn(x_, phase_train)
    y_conv = gap('gap', top_conv, n_classes)
    #cam_ = cam.get_class_map('gap', top_conv, 0, image_height)
    #################fully connected#############################
    """
    layer=tf.contrib.layers.flatten(layer)
    print layer.get_shape()
    layer = affine('fully_connect', layer, 1024 ,keep_prob=0.5)
    y_conv=affine('end_layer' , layer , n_classes , keep_prob=1.0)
    """
    #############################################################
    # cam = get_class_map('gap', top_conv, 0, im_width=image_width)
    print 'y_ shape : ',y_.get_shape()
    print 'y_conv shape : ',y_conv.get_shape()
    pred, pred_cls, cost, train_op, correct_pred, accuracy = algorithm(y_conv, y_, learning_rate , optimizer)
    saver = tf.train.Saver()
    config = tf.ConfigProto(
        device_count={'GPU': 1},
        log_device_placement=True
    )
    #config.gpu_options.allow_growth = True
    if not os.path.isdir(save_log_path):
        os.mkdir(save_log_path)
    summary_writer = tf.summary.FileWriter(save_log_path) # train_acc , train_loss, test_loss , test_acc 을 저장
    summary_writer.add_graph(tf.get_default_graph())
    sess = tf.Session()
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    """
    try:
        saver.restore(sess, restored_model_folder_path + 'best_acc.ckpt')
        print restored_model_folder_path+'model was restored!'
    except tf.errors.NotFoundError:
        print 'there was no model , make new model'
    ########################training##############################
    """
    check_point=100
    max_val = 0
    train_acc = 0;
    train_loss = 0;
    f=open('./log.txt' , 'w')
    batch_size =30
    start_time=time.time()
    try:
        for step in range(max_iter):
            #utils.show_progress(step, max_iter)
            if step % check_point == 0:
                end_time=time.time()
                #validation
                test_acc, test_loss = sess.run([accuracy, cost],feed_dict={x_: test_imgs, y_cls: test_cls, phase_train: False})
                utils.write_acc_loss(f, train_acc, train_loss, test_acc, test_loss)
                print '\n', step , test_acc, test_loss
                if test_acc > max_val or test_acc == max_val:
                    if test_acc == max_val and test_loss < min_loss:
                        saver.save(sess, restored_model_folder_path + '/best_acc.ckpt')
                        print 'model was saved!'
                        min_loss = test_loss
                        continue;
                    elif test_acc > max_val:
                        saver.save(sess, restored_model_folder_path + '/best_acc.ckpt')
                        print 'model was saved!'
                        max_val = test_acc
                        min_loss= test_loss
                #print 'time was consumed : ',end_time- start_time
                start_time=time.time()
                summary = tf.Summary(value=[tf.Summary.Value(tag='loss_test', simple_value=float(test_loss)), \
                                        tf.Summary.Value(tag='accuracy_test', simple_value=float(test_acc))])
                summary_writer.add_summary(summary , step)

            # names = ['cataract', 'glaucoma', 'retina', 'retina_glaucoma','retina_cataract', 'cataract_glaucoma', 'normal']
            batch_xs , batch_ys=utils.next_batch(train_imgs , train_cls , batch_size =30)
            batch_xs = aug.aug_level_1(batch_xs)
            train_acc, train_loss, _ = sess.run([accuracy, cost, train_op],
                                                feed_dict={x_: batch_xs, y_cls: batch_ys, phase_train: True})
            summary=tf.Summary(value=[tf.Summary.Value(tag='loss_train' , simple_value=float(train_loss)) ,\
                              tf.Summary.Value(tag='accuracy_train' , simple_value=float(train_acc))] )
            summary_writer.add_summary(summary , step)
            f.flush()
        f.close()
    except KeyboardInterrupt as kbi:
        print 'keyboard Interrupted all log was saved and tensorflow session closed'
        f.close()
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--max_iter", help='iteration',type=int)
    parser.add_argument("--learning_rate" , help='learning rate ',type=float)
    parser.add_argument("--structure" , help = 'what structrue you need')
    parser.add_argument("--optimizer",help='')
    args = parser.parse_args()
    args.max_iter=60000
    args.learning_rate=0.001
    """
    for i in range(5):
        train(args.max_iter, args.learning_rate,'simple_cnn', 'AdamOptimizer' ,'./data/type3' ,str(i), \
              'model/type3/'+str(i) , './logs/type3/'+str(i))
        tf.reset_default_graph()
    """
    for i in range(9):
        train(args.max_iter, args.learning_rate, 'simple_cnn', 'AdamOptimizer', './data/type4', str(i),\
              'model/type4/' + str(i) , './logs/type4/'+str(i))
        tf.reset_default_graph()


