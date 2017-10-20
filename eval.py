import tensorflow as tf
import numpy as np
import utils
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

c_test_imgs=np.load('./data/type1/c_images_4_test.npy')
n_test_imgs=np.load('./data/type1/n_images_4_test.npy')
if np.ndim(c_test_imgs) ==3 :
    c_test_imgs=c_test_imgs.reshape([-1,23,23,1])
    n_test_imgs = n_test_imgs.reshape([-1, 23, 23, 1])

print np.shape(c_test_imgs)
print np.shape(n_test_imgs)
pred=eval('./model/type1/0' , c_test_imgs)
tf.reset_default_graph()
print pred
print np.argmax(pred , axis=2)

pred=eval('./model/type1/0' , n_test_imgs)
tf.reset_default_graph()

print pred
print np.argmax(pred , axis=2)
c_test_imgs=c_test_imgs.reshape([-1,23,23])
n_test_imgs=n_test_imgs.reshape([-1,23,23])
print np.shape(n_test_imgs)
utils.plot_images(c_test_imgs)
utils.plot_images(n_test_imgs)

