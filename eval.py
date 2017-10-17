import tensorflow as tf
import numpy as np

import input


def eval(model_folder_path , images, labels=None):
    if not model_folder_path.endswith('/'):
        model_folder_path=model_folder_path+'/'
    sess = tf.Session()
    try:
        saver = tf.train.import_meta_graph(model_folder_path+'best_acc.ckpt.meta')
        saver.restore(sess, model_folder_path+'best_acc.ckpt')
    except IOError as ioe:
        print 'in model folder path , there is no best_acc.ckpt or best_acc.ckpt.meta files'
        return
    tf.get_default_graph()
    accuray = tf.get_default_graph().get_tensor_by_name('accuracy:0')
    prediction = tf.get_default_graph().get_tensor_by_name('softmax:0')

    x_ = tf.get_default_graph().get_tensor_by_name('x_:0')
    y_ = tf.get_default_graph().get_tensor_by_name('y_:0')

    if type(labels).__module__ == np.__name__ :
        'label data type : numpy '
        acc,pred=sess.run([accuray , prediction] , feed_dict={x_:images ,y_ : labels })
        return acc,pred
    else:
        print 'label data not assin '
        pred=sess.run([prediction] , feed_dict={x_:images })
        return pred


#train_imgs, train_cls, test_imgs, test_cls = input.get_train_test_images(input.get_type1_image(),
#                                                                             input.get_type0_image())

test_imgs=np.load('./data/c_images_0_test.npy')
if np.ndim(test_imgs) ==3 :
    test_imgs=test_imgs.reshape([-1,23,23,1])

print np.shape(test_imgs)
pred=eval('./model/4' , test_imgs)

print pred

print np.argmax([0.49 , 0.51])
print np.argmax(pred , axis=2)

