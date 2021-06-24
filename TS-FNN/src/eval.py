import tensorflow as tf
import tools
import TS_FNN
import utils
import tfRecord
import numpy as np
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '2'

def eval():

    with tf.name_scope('input'):
        val_image_RDM_batch, val_image_ATM_batch, val_label_batch = tfRecord.createShuffleBatch('../tfRecord/Test/test.tfrecords',utils.batch_size)
        # train_image_y_batch, train_label_y_batch = tfRecord.createBatch('../tfRecord/train/train2D.tfrecords','2D')

    x = tf.placeholder(tf.float32, shape=[None, utils.img_depth, utils.img_height, utils.img_width, utils.img_channels])
    y = tf.placeholder(tf.float32, shape=[None, utils.img_height, utils.img_width, utils.img_channels])
    y_ = tf.placeholder(tf.float32, shape=[None, utils.n_class])

    logits = TS_FNN.TS_FNN(x,y,is_pretrain=True)
    loss = tools.loss(logits, y_)
    accuracy = tools.accuracy(logits, y_)

    my_global_step = tf.Variable(0, name='global_step', trainable=False)
  
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=100)
    
    with tf.Session() as sess:
    
        saver.restore(sess, '../model/tf.fnn.ckpt-2000')
        print('** my global step:{}'.format(sess.run(my_global_step)))

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess,coord=coord)

        try:
            for step in np.arange(utils.max_step):
                if coord.should_stop():
                    break
                val_images_RDM, val_labels_ATM, val_labels = sess.run([val_image_RDM_batch, val_image_ATM_batch, val_label_batch])
                val_acc = sess.run(accuracy, feed_dict={x:val_images_RDM,y:val_images_ATM,y_:val_labels})
                print('** val accuracy = %.2f%% **'%(val_acc))
        except tf.errors.OutOfRangeError:
            print('Done training -- epoch limit reached.')
        finally:
            coord.request_stop()

        coord.join(threads)



if __name__ == '__main__':
    train()
