# app.py

import tensorflow as tf
import vgg
import tfRecord
import utils
import tools
import time
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '2'

def eval():
    '''
    Train the model defined
    '''

    with tf.name_scope('input'):
        # train_image_batch, train_label_batch = tfRecord.createShuffleBatch('../tfRecords/train/train.DTM.tfrecords', utils.batch_size)
        val_image_batch, val_label_batch = tfRecord.createBatch('../tfRecords/test/Test.ATM.tfrecords', 404)
    x = tf.placeholder(tf.float32, shape=[None, utils.img_width, utils.img_height, utils.img_channels])
    y_ = tf.placeholder(tf.int16, shape = [None, utils.num_class])

    logits = vgg.vgg16('inputz',x, utils.num_class, False)
    accuracy = tools.accuracy(logits, y_)

    my_global_step = tf.Variable(0, name='global_step', trainable=False)

    saver = tf.train.Saver()

    with tf.Session() as sess:

        saver.restore(sess, '../model/TrainATM/model-ATM.ckpt-2400')
        print('** my global step:{}'.format(sess.run(my_global_step)))
        
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess,coord=coord)
        #time_start = time.time()
        try:
            for i in range(1):
                val_images, val_labels = sess.run([val_image_batch, val_label_batch])
                val_acc = sess.run(accuracy, feed_dict={x:val_images, y_:val_labels})
                print('\n -------')
                print('** val accuracy = %.2f%% **'%(val_acc))
        except tf.errors.OutOfRangeError:
            print('Done training -- epoch limit reached.')
        finally:
            print('In finally')
            coord.request_stop()

        coord.join(threads)

eval()
