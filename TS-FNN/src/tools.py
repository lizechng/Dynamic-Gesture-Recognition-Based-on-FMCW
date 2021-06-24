import tensorflow as tf

def conv3d(layer_name, x, out_channels, kernel_size=[1,3,3], strides=[1,1,1,1,1], data_format='NDHWC', is_pretrain=True):
    '''
    Convolution 3D op wrapper, use RELU activation after convolution
    '''
    in_channels = x.get_shape()[-1].value
    with tf.variable_scope(layer_name):
        w = tf.get_variable(name='weight',
                            trainable=is_pretrain,
                            shape=[kernel_size[0],kernel_size[1],kernel_size[2],in_channels,out_channels],
                            initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable(name='bias',
                            trainable=is_pretrain,
                            shape=[out_channels],
                            initializer=tf.contrib.layers.xavier_initializer())
        x = tf.nn.conv3d(x, w, strides=strides, padding='SAME', data_format=data_format, name='conv3d')
        x = tf.nn.bias_add(x, b, name='bias_add')
        x = tf.nn.relu(x, name='relu')

    return x

def conv(layer_name, x, out_channels, kernel_size=[3,3], strides=[1,1,1,1], is_pretrain=True):
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
                            initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable(name='bias',
                            trainable=is_pretrain,
                            shape=[out_channels],
                            initializer=tf.constant_initializer(0.0))
        x = tf.nn.conv2d(x,w,strides,padding='SAME',name='conv')
        x = tf.nn.bias_add(x,b,name='bias_add')
        x = tf.nn.relu(x,name='relu')

        return x

def pool(layer_name, x, kernel_size=[1,2,2,1], strides=[1,2,2,1], is_max_pool=True):
    '''
    Pooling op
    Args:

    Returns:

    '''
    if is_max_pool:
        # May Name Conflict
        x = tf.nn.max_pool(x,kernel_size,strides=strides,padding='SAME',name=layer_name)
    else:
        x = tf.nn.avg_pool(x,kernel_size,strides=strides,padding='SAME',name=layer_name)
    return x

def pool3d(layer_name, x, kernel_size=[1,1,2,2,1], strides=[1,1,2,2,1], is_max_pool=True):
    '''
    Pooling 3D op
    '''
    if is_max_pool:
        x = tf.nn.max_pool3d(x, ksize=kernel_size, strides=strides, padding='VALID', name=layer_name)
    else:
        x = tf.nn.avg_pool3d(x, ksize=kernel_size, strides=strides, padding='VALID', name=layer_name)
    return x

def batch_norm(x):
    '''
    Batch normlization (w/o the offset and scale)
    '''
    pass

def fc_layer(layer_name, x, out_nodes):
    '''
    Wrapper for fully connected layers with RELU activation as default
    '''
    shape = x.get_shape()
    if len(shape) == 5: # FC 3D
        size = shape[1].value*shape[2].value*shape[3].value*shape[4].value
    elif len(shape) == 4:
        size = shape[1].value*shape[2].value*shape[3].value
    else:
        size = shape[-1].value

    with tf.variable_scope(layer_name):
        w = tf.get_variable(name='weight',
                            shape=[size, out_nodes],
                            initializer=tf.constant_initializer(0.0))
        b = tf.get_variable(name='bias',
                            shape=[out_nodes],
                            initializer=tf.constant_initializer(0.0))
        # batch?
        flat_x = tf.reshape(x, [-1,size])

        x = tf.nn.bias_add(tf.matmul(flat_x,w), b)
        x = tf.nn.relu(x)

    return x

def lstm():
    '''
    Build LSTM cell
    '''
    pass
def loss(logits, labels):
    '''
    Compute loss
    '''
    with tf.name_scope('loss') as scope:
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels, name='cross-entropy')
        loss = tf.reduce_mean(cross_entropy, name='loss')
        tf.summary.scalar(scope+'/loss', loss)

    return loss

def accuracy(logits, labels):
    '''
    Evaluate the quality of the logits at predicting the label
    '''
    # for summary
    with tf.name_scope('accuracy') as scope:
        correct = tf.equal(tf.arg_max(logits,1), tf.arg_max(labels,1))
        correct = tf.cast(correct, tf.float32)
        accuracy = tf.reduce_mean(correct)*100.0
        tf.summary.scalar(scope+'accuracy',accuracy)

    return accuracy

def num_correct_prediction(logits, labels):
    '''
    Evaluate the quality of the logits at predicting the label
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
