import tensorflow as tf
import tools
import utils

def TS_FNN(x, y, is_pretrain):
    '''
    Structure of TS FNN
    : x for 3D CNN input tensor
    : y for 2D CNN input tensor
    '''
    with tf.variable_scope('3D-CNN'):
        x = tools.conv3d('conv3d_1',x,64,kernel_size=[3,3,3],strides=[1,1,1,1,1],data_format='NDHWC',is_pretrain=is_pretrain)
        x = tools.pool3d('pool3d_1',x,kernel_size=[1,1,2,2,1],strides=[1,1,2,2,1],is_max_pool=True)

        x = tools.conv3d('conv3d_2',x,128,kernel_size=[3,3,3],strides=[1,1,1,1,1],data_format='NDHWC',is_pretrain=is_pretrain)
        x = tools.pool3d('pool3d_2',x,kernel_size=[1,2,2,2,1],strides=[1,2,2,2,1],is_max_pool=True)

        x = tools.conv3d('conv3d_3_1',x,256,kernel_size=[3,3,3],strides=[1,1,1,1,1],data_format='NDHWC',is_pretrain=is_pretrain)
        x = tools.conv3d('conv3d_3_2',x,256,kernel_size=[3,3,3],strides=[1,1,1,1,1],data_format='NDHWC',is_pretrain=is_pretrain)
        x = tools.pool3d('pool3d_3',x,kernel_size=[1,2,2,2,1],strides=[1,2,2,2,1],is_max_pool=True)

        x = tools.conv3d('conv3d_4_1',x,512,kernel_size=[3,3,3],strides=[1,1,1,1,1],data_format='NDHWC',is_pretrain=is_pretrain)
        x = tools.conv3d('conv3d_4_2',x,512,kernel_size=[3,3,3],strides=[1,1,1,1,1],data_format='NDHWC',is_pretrain=is_pretrain)
        x = tools.pool3d('pool3d_4',x,kernel_size=[1,2,2,2,1],strides=[1,2,2,2,1],is_max_pool=True)

        x = tools.conv3d('conv3d_5_1',x,512,kernel_size=[3,3,3],strides=[1,1,1,1,1],data_format='NDHWC',is_pretrain=is_pretrain)
        x = tools.conv3d('conv3d_5_2',x,512,kernel_size=[3,3,3],strides=[1,1,1,1,1],data_format='NDHWC',is_pretrain=is_pretrain)
        x = tools.pool3d('pool3d_5',x,kernel_size=[1,2,2,2,1],strides=[1,2,2,2,1],is_max_pool=True)

        x = tools.fc_layer('fc',x,out_nodes=utils.num_steps)

        x = tf.reshape(x, [utils.batch_size,utils.num_steps,utils.input_size_x])

    with tf.variable_scope('2D-CNN'):
        y = tools.conv('conv_1',y,64,kernel_size=[3,3],strides=[1,1,1,1],is_pretrain=is_pretrain)
        y = tools.pool('pool_1',y,kernel_size=[1,2,2,1],strides=[1,2,2,1],is_max_pool=True)

        y = tools.conv('conv_2',y,128,kernel_size=[3,3],strides=[1,1,1,1],is_pretrain=is_pretrain)
        y = tools.pool('pool_2',y,kernel_size=[1,2,2,1],strides=[1,2,2,1],is_max_pool=True)

        y = tools.conv('conv_3',y,256,kernel_size=[3,3],strides=[1,1,1,1],is_pretrain=is_pretrain)
        y = tools.pool('pool_3',y,kernel_size=[1,2,2,1],strides=[1,2,2,1],is_max_pool=True)

        y = tools.conv('conv_4',y,512,kernel_size=[3,3],strides=[1,1,1,1],is_pretrain=is_pretrain)
        y = tools.pool('pool_4',y,kernel_size=[1,2,2,1],strides=[1,2,2,1],is_max_pool=True)

        y = tools.fc_layer('fc',y,out_nodes=utils.num_steps)

        y = tf.reshape(y, [utils.batch_size,utils.num_steps,utils.input_size_y])

    with tf.variable_scope('LSTM'):

        z = tf.concat([x, y], axis = 2)

        with tf.variable_scope('lstm'):
            seq_len = tf.fill([utils.batch_size], utils.num_steps)

            cell = tf.nn.rnn_cell.LSTMCell(utils.num_hidden, state_is_tuple=True)

            cell1 = tf.nn.rnn_cell.LSTMCell(utils.num_hidden, state_is_tuple=True)

            stack = tf.nn.rnn_cell.MultiRNNCell([cell, cell1], state_is_tuple=True)
            initial_state = stack.zero_state(utils.batch_size, dtype=tf.float32)

            outputs, _ = tf.nn.dynamic_rnn(
                cell=stack,
                inputs=z,
                sequence_length=seq_len,
                initial_state=initial_state,
                dtype=tf.float32,
                time_major=False
            )

            outputs = outputs[:,-1,:]
            W = tf.get_variable(name='W_out',shape=[utils.num_hidden,utils.n_class],dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer())
            b = tf.get_variable(name='b_out',shape=[utils.n_class],dtype=tf.float32,initializer=tf.constant_initializer())

            result = tf.matmul(outputs, W) + b

    return result
