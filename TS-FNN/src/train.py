import tensorflow as tf
import tools
import TS_FNN
import utils
import tfRecord
import numpy as np
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '2'

def train():

    with tf.name_scope('input'):
        train_image_RDM_batch, train_image_ATM_batch, train_label_batch = tfRecord.createShuffleBatch('../tfRecords/Train/train.tfrecords',utils.batch_size)
        # train_image_y_batch, train_label_y_batch = tfRecord.createBatch('../tfRecord/train/train2D.tfrecords','2D')

    x = tf.placeholder(tf.float32, shape=[None, utils.img_depth, utils.img_height, utils.img_width, utils.img_channels])
    y = tf.placeholder(tf.float32, shape=[None, utils.img_height, utils.img_width, utils.img_channels])
    y_ = tf.placeholder(tf.float32, shape=[None, utils.n_class])

    logits = TS_FNN.TS_FNN(x,y,is_pretrain=True)
    loss = tools.loss(logits, y_)
    accuracy = tools.accuracy(logits, y_)

    my_global_step = tf.Variable(0, name='global_step', trainable=False)
    
    lr = tf.train.exponential_decay(learning_rate=0.003, global_step=my_global_step, decay_steps=30, decay_rate=0.99)
    
    train_op = tools.optimize(loss, lr, my_global_step)

    saver = tf.train.Saver(tf.global_variables(), max_to_keep=100)

    init = tf.global_variables_initializer()

    merged = tf.summary.merge_all()
    
    with tf.Session() as sess:
    
        writer = tf.summary.FileWriter('../log', sess.graph)
        print('** sess.graph write sucessfully.')

        sess.run(init)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess,coord=coord)

        try:
            for step in np.arange(utils.max_step):
                if coord.should_stop():
                    break
                train_images_RDM, train_images_ATM, train_labels = sess.run([train_image_RDM_batch, train_image_ATM_batch, train_label_batch])
                _, train_loss, train_acc, summery = sess.run([train_op, loss, accuracy, merged], feed_dict={x:train_images_RDM,y:train_images_ATM,y_:train_labels})
                writer.add_summary(summery, step)
                    
                # step by 50
                # if step%1 == 0 or (step+1) == utils.max_step:
                    # val_images_x, val_labels_x = sess.run([val_image_x_batch, val_label_x_batch])
                    # val_images_y, val_labels_y = sess.run([val_image_y_batch, val_label_y_batch])
                    # val_loss, val_acc = sess.run([loss, accuracy], feed_dict={x:val_images_x,y:val_images_y, y_:val_labels_y})
                    # print('** Step %d, val loss = %.2f, val accuracy = %.2f%% **'%(step, val_loss, val_acc))

                if step%200 == 0 or (step+1) == utils.max_step:
                    checkpoint_path = '../model/' + 'ts.fnn.model.ckpt'
                    saver.save(sess, checkpoint_path, global_step=step)
        except tf.errors.OutOfRangeError:
            print('Done training -- epoch limit reached.')
        finally:
            coord.request_stop()

        coord.join(threads)



if __name__ == '__main__':
    train()
