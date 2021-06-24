import tensorflow as tf
import numpy as np

def conv(layer_name, x, out_channels, kernel_size=[3,3], stride=[1,1,1,1], is_pretrain=True):
    '''
    Convolution op wrapper, use RELU activation after convolution
    Args:
        layer_name:
        x: input tensor
    Returns:
        4D tensor
    '''
    # x.get_shape()[-1] : Dimension(3)
    # x.get_shape()[-1].value : 3
    in_channels = x.get_shape()[-1].value
    with tf.variable_scope(layer_name):
        w = tf.get_variable(name='weights',
                            trainable=is_pretrain,
                            shape=[kernel_size[0],kernel_size[1],in_channels,out_channels],
                            initializer=tf.contrib.layers.xavier_initializer(seed=12345))
        b = tf.get_variable(name='bias',
                            trainable=is_pretrain,
                            shape=[out_channels],
                            initializer=tf.constant_initializer(0.0))
        x = tf.nn.conv2d(x,w,stride,padding='SAME',name='conv')
        x = tf.nn.bias_add(x,b,name='bias_add')
        x = tf.nn.leaky_relu(x,name='relu')

        return x

def pool(layer_name, x, kernel=[1,2,2,1], stride=[1,2,2,1], is_max_pool=True):
    '''
    Pooling op
    Args:

    Returns:

    '''
    if is_max_pool:
        # May Name Conflict
        x = tf.nn.max_pool(x,kernel,strides=stride,padding='SAME',name=layer_name)
    else:
        x = tf.nn.avg_pool(x,kernel,strides=stride,padding='SAME',name=layer_name)
    return x

def batch_norm(x):
    '''
    Batch normlization(without the offset and scale)
    '''
    epsilon = 1e-3
    batch_mean, batch_var = tf.nn.moments(x,[0])
    x = tf.nn.batch_normalization(x,
                                  mean=batch_mean,
                                  variance=batch_var,
                                  offset=None,
                                  scale=None,
                                  variance_epsilon=epsilon)
    return x

def fc_layer(layer_name, x, out_nodes):
    '''
    Wrapper for fully connected layers with RELU activation as default
    Args:

    Returns:
        2D tensor
    '''
    shape = x.get_shape()
    if len(shape) == 4:
        size = shape[1].value*shape[2].value*shape[3].value
    else:
        size = shape[-1].value
    # attention the initializer
    with tf.variable_scope(layer_name):
        w = tf.get_variable(name='weight',
                            shape=[size, out_nodes],
                            initializer=tf.contrib.layers.xavier_initializer(seed=12345))
        b = tf.get_variable(name='bias',
                            shape=[out_nodes],
                            initializer=tf.constant_initializer(0.0))
        # -1 for the possibility of batch_size
        flat_x = tf.reshape(x, [-1, size])

        x = tf.nn.bias_add(tf.matmul(flat_x,w), b)
        # x = tf.nn.leaky_relu(x)

    return x

def loss(logits, labels):
    '''
    Compute loss
    Args:

    '''
    # attention. name_scope.
    with tf.name_scope('loss') as scope:
        # cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels, name='cross-entropy')
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels, name='cross-entropy')
        # without regularizer
        loss = tf.reduce_mean(cross_entropy, name='loss')
        # attention. what the scope
        tf.summary.scalar(scope+'/loss',loss)

    return loss

def accuracy(logits, labels):
    '''
    Evaluate the quality of the logits at predicting the label.
    Args:
    '''
    with tf.name_scope('accuracy') as scope:
        correct = tf.equal(tf.arg_max(logits,1), tf.arg_max(labels,1))
        correct = tf.cast(correct, tf.float32)
        accuracy = tf.reduce_mean(correct)*100.0
        tf.summary.scalar(scope+'/accuracy',accuracy)

    return accuracy

def num_correct_prediction(logits, labels):
    '''
    Evaluate the quality fo the logits at predicting the label.
    Args:
    '''
    correct = tf.equal(tf.arg_max(logits,1), tf.arg_max(labels,1))
    correct = tf.cast(correct, tf.int32)
    n_correct = tf.reduce_sum(correct)

    return n_correct

def optimize(loss, learning_rate, global_step):
    '''
    Optimization, use Gradient Descent as default
    '''
    with tf.name_scope('optimizer'):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        train_op = optimizer.minimize(loss, global_step=global_step)

    return train_op

def print_all_variables(train_only=True):
    '''
    Print all trainable and non-trainable variables
    '''
    if train_only:
        t_vars = tf.trainable_variables()
        print('[*] printing trainable variables')
    else:
        try:
            t_vars = tf.global_variables()
        except:
            t_vars = tf.all_variables()
        print('[*] printing global variables')

    for idx, v in enumerate(t_vars):
        print('  var {:3}: {:15}   {}'.format(idx, str(v.get_shape()), v.name))
