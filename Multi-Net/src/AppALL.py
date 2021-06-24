# app.py

import tensorflow as tf
import vgg
import tfRecord
import utils
import tools
import multi_net
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '2'

def eval():
    '''
    Train the model defined
    '''

    with tf.name_scope('input'):
        # train_image_batch, train_label_batch = tfRecord.createShuffleBatch('../tfRecords/train/train.DTM.tfrecords', utils.batch_size)
        val_image_x_batch, val_image_y_batch, val_image_z_batch, val_label_batch = tfRecord.createAllBatch('../tfRecords/test/test.tfrecords', 404)
    
    x = tf.placeholder(tf.float32, shape=[None, utils.img_width, utils.img_height, utils.img_channels])
    y = tf.placeholder(tf.float32, shape=[None, utils.img_width, utils.img_height, utils.img_channels])
    z = tf.placeholder(tf.float32, shape=[None, utils.img_width, utils.img_height, utils.img_channels])
    y_ = tf.placeholder(tf.int16, shape = [None, utils.num_class])
    
    logits = multi_net.multi_net(x, y, z, utils.num_class, False)
    accuracy = tools.accuracy(logits, y_)

    my_global_step = tf.Variable(0, name='global_step', trainable=False)

    saver = tf.train.Saver()

    with tf.Session() as sess:

        saver.restore(sess, '../model/TrainALL/model_multi_net.ckpt-800')
        print('** my global step:{}'.format(sess.run(my_global_step)))
        
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess,coord=coord)
        #time_start = time.time()
        try:
            for i in range(1):
                val_images_x, val_images_y, val_images_z, val_labels = sess.run([val_image_x_batch, val_image_y_batch, val_image_z_batch, val_label_batch])
                val_acc = sess.run(accuracy, feed_dict={x:val_images_x, y:val_images_y, z:val_images_z, y_:val_labels})
                print('\n -------')
                print('** val accuracy = %.2f%% **'%(val_acc))
        except tf.errors.OutOfRangeError:
            print('Done training -- epoch limit reached.')
        finally:
            print('In finally')
            coord.request_stop()

        coord.join(threads)

eval()
