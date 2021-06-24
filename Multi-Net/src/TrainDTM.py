import os
import os.path
import math

import numpy as np
import tensorflow as tf
import tfRecord
import vgg
import tools
import utils

import matplotlib.pyplot as plt

import time

os.environ['CUDA_VISIBLE_DEVICES'] = '2'

def train_x():
    '''
    Train the model defined
    '''

    with tf.name_scope('input'):
        train_image_batch, train_label_batch = tfRecord.createShuffleBatch('../tfRecords/train/train.DTM.tfrecords', utils.batch_size)
        val_image_batch, val_label_batch = tfRecord.createShuffleBatch('../tfRecords/test/test.DTM.tfrecords', utils.eval_batch_size)
    x = tf.placeholder(tf.float32, shape=[None, utils.img_width, utils.img_height, utils.img_channels])
    y_ = tf.placeholder(tf.int16, shape = [None, utils.num_class])

    logits = vgg.vgg16('inputx',x, utils.num_class, utils.IS_PRETRAIN)
    loss = tools.loss(logits, y_)
    accuracy = tools.accuracy(logits, y_)
    
    my_global_step = tf.Variable(0, name='global_step', trainable=False)
    
    lr = tf.train.exponential_decay(learning_rate=utils.learning_rate, global_step=my_global_step, decay_steps=utils.decay_step, decay_rate=0.99)
    
    train_op = tools.optimize(loss, lr, my_global_step)

    saver = tf.train.Saver(tf.global_variables(), max_to_keep=100)

    init = tf.global_variables_initializer()

    merged = tf.summary.merge_all()

    with tf.Session() as sess:
        
        writer = tf.summary.FileWriter('../log/LogDTM', sess.graph)
        print('** sess.graph write sucessfully.')
        
        sess.run(init)
        
        # Train Con
        # saver.restore(sess, '../model/TrainRTM/model-RTM.ckpt-600')
        
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess,coord=coord)
        time_start = time.time()
        
        try:
            while True:
                step = sess.run(my_global_step)
                if step <= utils.max_step:
                    if coord.should_stop():
                        print('coord.should_stop.')
                        break
                    train_images, train_labels = sess.run([train_image_batch, train_label_batch])
                    # plt.imshow(train_images[0])
                    # plt.show()
                    # print('\n ---------')
                    # print('** run_labels:{}'.format(sess.run(logits, feed_dict={x:train_images})))
                    # print('** train_labels:{}'.format(train_labels))
                    _, train_loss, train_acc, summery = sess.run([train_op, loss, accuracy, merged], feed_dict={x:train_images,y_:train_labels})
                    writer.add_summary(summery, step)
                    print('** step{}: train_loss = {:.3f}, train_acc = {:.3f} **'.format(step, train_loss, train_acc))
                    # step by 50
                    # if step%10 == 0 or (step+1) == utils.max_step:
                        # time_end = time.time()
                        # val_images, val_labels = sess.run([val_image_batch, val_label_batch])
                        # val_loss, val_acc = sess.run([loss, accuracy], feed_dict={x:val_images, y_:val_labels})
                        # print('\n -------')
                        # print('** Step %d, val loss = %.2f, val accuracy = %.2f%% **'%(step, val_loss, val_acc))
                        # print('** Time cost: {} **'.format(time_end-time_start))
                    if step%200 == 0 or step == utils.max_step:
                        checkpoint_path = '../model/TrainDTM' + '/model-DTM.ckpt'
                        saver.save(sess, checkpoint_path, global_step=step)
                else:
                    break
        except tf.errors.OutOfRangeError:
            print('Done training -- epoch limit reached.')
        finally:
            print('In finally')
            coord.request_stop()

        coord.join(threads)

if __name__ == '__main__':
    train_x()
    print('** DTM train successfully.')
