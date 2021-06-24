import tensorflow as tf
import multi_net
import tfRecord
import tools
import numpy as np

import os.path
import math
import utils

os.environ['CUDA_VISIBLE_DEVICES'] = '2'

def multi_net_train():
    '''
    Train the multi-branch neural network model
    '''
    checkpoint_x_path = '../model/TrainDTM/model-DTM.ckpt-2400'
    checkpoint_y_path = '../model/TrainRTM/model-RTM.ckpt-2400'
    checkpoint_z_path = '../model/TrainATM/model-ATM.ckpt-2400'

    # train_log_dir = './model'

    train_image_x_batch, train_image_y_batch, train_image_z_batch, train_label_batch = tfRecord.createAllShuffleBatch('../tfRecords/Train/train.tfrecords', utils.batch_size)
    val_image_x_batch, val_image_y_batch, val_image_z_batch, val_label_batch = tfRecord.createAllShuffleBatch('../tfRecords/Test/test.tfrecords', utils.eval_batch_size)

    x = tf.placeholder(tf.float32, shape=[None, utils.img_width, utils.img_height, utils.img_channels])
    y = tf.placeholder(tf.float32, shape=[None, utils.img_width, utils.img_height, utils.img_channels])
    z = tf.placeholder(tf.float32, shape=[None, utils.img_width, utils.img_height, utils.img_channels])
    y_ = tf.placeholder(tf.int16, shape = [None, utils.num_class])

    logits = multi_net.multi_net(x, y, z, utils.num_class, utils.IS_PRETRAIN)
    loss = tools.loss(logits, y_)
    accuracy = tools.accuracy(logits, y_)

    my_global_step = tf.Variable(0, name='global_step', trainable=False)
    lr = tf.train.exponential_decay(learning_rate=utils.learning_rate, global_step=my_global_step, decay_steps=utils.decay_step, decay_rate=0.99)
    
    train_op = tools.optimize(loss, lr, my_global_step)

    variables = tf.global_variables()

    variables_to_restore_x = [v for v in variables if v.name.split('/')[0] == 'inputx_conv_block']
    variables_to_restore_y = [v for v in variables if v.name.split('/')[0] == 'inputy_conv_block']
    variables_to_restore_z = [v for v in variables if v.name.split('/')[0] == 'inputz_conv_block']

    saver_x = tf.train.Saver(variables_to_restore_x)
    saver_y = tf.train.Saver(variables_to_restore_y)
    saver_z = tf.train.Saver(variables_to_restore_z)
    
    saver = tf.train.Saver(variables)

    init = tf.global_variables_initializer()
    merged = tf.summary.merge_all()
    with tf.Session() as sess:

        writer = tf.summary.FileWriter('../log/LogALL', sess.graph)
        print('** sess.graph write sucessfully.')

        sess.run(init)

        saver_x.restore(sess, checkpoint_x_path)
        saver_y.restore(sess, checkpoint_y_path)
        saver_z.restore(sess, checkpoint_z_path)
        
        # Train Con
        # saver.restore(sess, '../model/TrainALL/model_multi_net.ckpt-600')
        
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess,coord=coord)

        try:
            while True:
                step = sess.run(my_global_step)
                if step <= utils.max_step:
                    if coord.should_stop():
                        print('coord.should_stop.')
                        break
                    train_images_x, train_images_y, train_images_z, train_labels = sess.run([train_image_x_batch, train_image_y_batch, train_image_z_batch, train_label_batch])

                    _, train_loss, train_acc, summery = sess.run([train_op, loss, accuracy, merged], feed_dict={x:train_images_x, y:train_images_y, z:train_images_z, y_:train_labels})
                    writer.add_summary(summery, step)
                    print('** Step %d, train loss = %.2f, train accuracy = %.2f%% **'%(step, train_loss, train_acc))
                    # step by 50
                    # if step%1 == 0 or (step+1) == utils.max_step:
                        # val_images_x, val_images_y, val_labels = sess.run([val_image_x_batch, val_image_y_batch, val_label_batch])
                        # val_loss, val_acc = sess.run([loss, accuracy], feed_dict={x:val_images_x, y:val_images_y, y_:val_labels})
                        # print('** Step %d, val loss = %.2f, val accuracy = %.2f%% **'%(step, val_loss, val_acc))

                    if step%200 == 0 or step == utils.max_step:
                        # checkpoint_path = os.path.join(train_log_dir, 'model.ckpt')
                        checkpoint_path = '../model/TrainALL' + '/model_multi_net.ckpt'
                        saver.save(sess, checkpoint_path, global_step=step)
                else:
                    break
        except tf.errors.OutOfRangeError:
            print('Done training -- epoch limit reached.')
        finally:
            coord.request_stop()
        coord.join(threads)

if __name__ == '__main__':
    multi_net_train()
