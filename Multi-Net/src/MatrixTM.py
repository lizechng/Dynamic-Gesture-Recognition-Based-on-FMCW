# app.py

import tensorflow as tf
import vgg
import tfRecord
import utils
import tools
import time
import warnings
import numpy as np
warnings.filterwarnings('ignore')
def evalDTM():
    '''
    Train the model defined
    '''

    with tf.name_scope('input'):
        # train_image_batch, train_label_batch = tfRecord.createShuffleBatch('../tfRecords/train/train.DTM.tfrecords', utils.batch_size)
        val_image_batch, val_label_batch = tfRecord.createBatch('../tfRecords/test/test.DTM.tfrecords', 101)
    x = tf.placeholder(tf.float32, shape=[None, utils.img_width, utils.img_height, utils.img_channels])
    y_ = tf.placeholder(tf.int16, shape = [None, utils.num_class])

    logits = vgg.vgg16('inputx',x, utils.num_class, False)
    accuracy = tools.accuracy(logits, y_)

    my_global_step = tf.Variable(0, name='global_step', trainable=False)

    saver = tf.train.Saver()

    with tf.Session() as sess:

        saver.restore(sess, '../model/TrainDTM/model-DTM.ckpt-200')
        print('** my global step:{}'.format(sess.run(my_global_step)))
        
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess,coord=coord)
        #time_start = time.time()
        try:
            for i in range(4):
                val_images, val_labels = sess.run([val_image_batch, val_label_batch])
                val_logits = sess.run(logits, feed_dict={x:val_images})
                print('\n -------')
                # val_logits = val_logits.tolist()
                print('type(val_logits) = ',type(val_logits))
                print('val_logits:\n', np.argmax(val_logits,1))
                # val_labels = val_labels.tolist()
                print('type(val_labels) = ',type(val_labels))
                print('val_labels:\n', np.argmax(val_labels,1))
                # print('** val accuracy = %.2f%% **'%(val_acc))
        except tf.errors.OutOfRangeError:
            print('Done training -- epoch limit reached.')
        finally:
            print('In finally')
            coord.request_stop()

        coord.join(threads)
def evalDTM():
    '''
    Train the model defined
    '''

    with tf.name_scope('input'):
        # train_image_batch, train_label_batch = tfRecord.createShuffleBatch('../tfRecords/train/train.DTM.tfrecords', utils.batch_size)
        val_image_batch, val_label_batch = tfRecord.createBatch('../tfRecords/test/test.DTM.tfrecords', 101)
    x = tf.placeholder(tf.float32, shape=[None, utils.img_width, utils.img_height, utils.img_channels])
    y_ = tf.placeholder(tf.int16, shape = [None, utils.num_class])

    logits = vgg.vgg16('inputx',x, utils.num_class, False)
    accuracy = tools.accuracy(logits, y_)

    my_global_step = tf.Variable(0, name='global_step', trainable=False)

    saver = tf.train.Saver()

    with tf.Session() as sess:

        saver.restore(sess, '../model/TrainDTM/model-DTM.ckpt-200')
        print('** my global step:{}'.format(sess.run(my_global_step)))
        
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess,coord=coord)
        #time_start = time.time()
        try:
            for i in range(4):
                val_images, val_labels = sess.run([val_image_batch, val_label_batch])
                val_logits = sess.run(logits, feed_dict={x:val_images})
                print('\n -------')
                # val_logits = val_logits.tolist()
                print('type(val_logits) = ',type(val_logits))
                print('val_logits:\n', np.argmax(val_logits,1))
                # val_labels = val_labels.tolist()
                print('type(val_labels) = ',type(val_labels))
                print('val_labels:\n', np.argmax(val_labels,1))
                # print('** val accuracy = %.2f%% **'%(val_acc))
        except tf.errors.OutOfRangeError:
            print('Done training -- epoch limit reached.')
        finally:
            print('In finally')
            coord.request_stop()

        coord.join(threads)
def evalDTM():
    '''
    Train the model defined
    '''

    with tf.name_scope('input'):
        # train_image_batch, train_label_batch = tfRecord.createShuffleBatch('../tfRecords/train/train.DTM.tfrecords', utils.batch_size)
        val_image_batch, val_label_batch = tfRecord.createBatch('../tfRecords/test/test.DTM.tfrecords', 101)
    x = tf.placeholder(tf.float32, shape=[None, utils.img_width, utils.img_height, utils.img_channels])
    y_ = tf.placeholder(tf.int16, shape = [None, utils.num_class])

    logits = vgg.vgg16('inputx',x, utils.num_class, False)
    accuracy = tools.accuracy(logits, y_)

    my_global_step = tf.Variable(0, name='global_step', trainable=False)

    saver = tf.train.Saver()

    with tf.Session() as sess:

        saver.restore(sess, '../model/TrainDTM/model-DTM.ckpt-200')
        print('** my global step:{}'.format(sess.run(my_global_step)))
        
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess,coord=coord)
        #time_start = time.time()
        try:
            for i in range(4):
                val_images, val_labels = sess.run([val_image_batch, val_label_batch])
                val_logits = sess.run(logits, feed_dict={x:val_images})
                print('\n -------')
                # val_logits = val_logits.tolist()
                print('type(val_logits) = ',type(val_logits))
                print('val_logits:\n', np.argmax(val_logits,1))
                # val_labels = val_labels.tolist()
                print('type(val_labels) = ',type(val_labels))
                print('val_labels:\n', np.argmax(val_labels,1))
                # print('** val accuracy = %.2f%% **'%(val_acc))
        except tf.errors.OutOfRangeError:
            print('Done training -- epoch limit reached.')
        finally:
            print('In finally')
            coord.request_stop()

        coord.join(threads)

if __name__ == '__main__':
    evalDTM()
