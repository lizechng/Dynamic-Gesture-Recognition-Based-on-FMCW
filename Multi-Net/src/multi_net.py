import tensorflow as tf
import tools
import utils

def multi_net(x, y, z, n_class, is_pretrain):

    with tf.variable_scope('inputx_conv_block'):
        x = tools.conv('conv1_1',x,32,kernel_size=[3,3],stride=[1,1,1,1],is_pretrain=False)
        x = tools.conv('conv1_2',x,32,kernel_size=[3,3],stride=[1,1,1,1],is_pretrain=False)
        x = tools.pool('pool1',x,kernel=[1,2,2,1],stride=[1,2,2,1],is_max_pool=True)

        x = tools.conv('conv2_1',x,64,kernel_size=[3,3],stride=[1,1,1,1],is_pretrain=False)
        x = tools.conv('conv2_2',x,64,kernel_size=[3,3],stride=[1,1,1,1],is_pretrain=False)
        x = tools.pool('pool2',x,kernel=[1,2,2,1],stride=[1,2,2,1],is_max_pool=True)

        x = tools.conv('conv3_1',x,128,kernel_size=[3,1],stride=[1,1,1,1],is_pretrain=False)
        x = tools.conv('conv3_2',x,128,kernel_size=[1,3],stride=[1,1,1,1],is_pretrain=False)
        x = tools.conv('conv3_3',x,128,kernel_size=[3,3],stride=[1,1,1,1],is_pretrain=False)
        x = tools.pool('pool3',x,kernel=[1,2,2,1],stride=[1,2,2,1],is_max_pool=True)

        x = tools.conv('conv4_1',x,256,kernel_size=[3,1],stride=[1,1,1,1],is_pretrain=False)
        x = tools.conv('conv4_2',x,256,kernel_size=[1,3],stride=[1,1,1,1],is_pretrain=False)
        x = tools.conv('conv4_3',x,256,kernel_size=[3,3],stride=[1,1,1,1],is_pretrain=False)
        x = tools.pool('pool4',x,kernel=[1,2,2,1],stride=[1,2,2,1],is_max_pool=True)

        x = tools.conv('conv5_1',x,256,kernel_size=[3,1],stride=[1,1,1,1],is_pretrain=False)
        x = tools.conv('conv5_2',x,256,kernel_size=[1,3],stride=[1,1,1,1],is_pretrain=False)
        x = tools.conv('conv5_3',x,256,kernel_size=[3,3],stride=[1,1,1,1],is_pretrain=False)
        x = tools.pool('pool5',x,kernel=[1,2,2,1],stride=[1,2,2,1],is_max_pool=True)

        x = tools.fc_layer('fc6',x,out_nodes=utils.fc6_nodes)
        x = tf.nn.relu(x)
    with tf.variable_scope('inputy_conv_block'):
        y = tools.conv('conv1_1',y,32,kernel_size=[3,3],stride=[1,1,1,1],is_pretrain=False)
        y = tools.conv('conv1_2',y,32,kernel_size=[3,3],stride=[1,1,1,1],is_pretrain=False)
        y = tools.pool('pool1',y,kernel=[1,2,2,1],stride=[1,2,2,1],is_max_pool=True)

        y = tools.conv('conv2_1',y,64,kernel_size=[3,3],stride=[1,1,1,1],is_pretrain=False)
        y = tools.conv('conv2_2',y,64,kernel_size=[3,3],stride=[1,1,1,1],is_pretrain=False)
        y = tools.pool('pool2',y,kernel=[1,2,2,1],stride=[1,2,2,1],is_max_pool=True)

        y = tools.conv('conv3_1',y,128,kernel_size=[3,1],stride=[1,1,1,1],is_pretrain=False)
        y = tools.conv('conv3_2',y,128,kernel_size=[1,3],stride=[1,1,1,1],is_pretrain=False)
        y = tools.conv('conv3_3',y,128,kernel_size=[3,3],stride=[1,1,1,1],is_pretrain=False)
        y = tools.pool('pool3',y,kernel=[1,2,2,1],stride=[1,2,2,1],is_max_pool=True)

        y = tools.conv('conv4_1',y,256,kernel_size=[3,1],stride=[1,1,1,1],is_pretrain=False)
        y = tools.conv('conv4_2',y,256,kernel_size=[1,3],stride=[1,1,1,1],is_pretrain=False)
        y = tools.conv('conv4_3',y,256,kernel_size=[3,3],stride=[1,1,1,1],is_pretrain=False)
        y = tools.pool('pool4',y,kernel=[1,2,2,1],stride=[1,2,2,1],is_max_pool=True)

        y = tools.conv('conv5_1',y,256,kernel_size=[3,1],stride=[1,1,1,1],is_pretrain=False)
        y = tools.conv('conv5_2',y,256,kernel_size=[1,3],stride=[1,1,1,1],is_pretrain=False)
        y = tools.conv('conv5_3',y,256,kernel_size=[3,3],stride=[1,1,1,1],is_pretrain=False)
        y = tools.pool('pool5',y,kernel=[1,2,2,1],stride=[1,2,2,1],is_max_pool=True)

        y = tools.fc_layer('fc6',y,out_nodes=utils.fc6_nodes)
        y = tf.nn.relu(y)
        
    with tf.variable_scope('inputz_conv_block'):
        z = tools.conv('conv1_1',z,32,kernel_size=[3,3],stride=[1,1,1,1],is_pretrain=False)
        z = tools.conv('conv1_2',z,32,kernel_size=[3,3],stride=[1,1,1,1],is_pretrain=False)
        z = tools.pool('pool1',z,kernel=[1,2,2,1],stride=[1,2,2,1],is_max_pool=True)

        z = tools.conv('conv2_1',z,64,kernel_size=[3,3],stride=[1,1,1,1],is_pretrain=False)
        z = tools.conv('conv2_2',z,64,kernel_size=[3,3],stride=[1,1,1,1],is_pretrain=False)
        z = tools.pool('pool2',z,kernel=[1,2,2,1],stride=[1,2,2,1],is_max_pool=True)

        z = tools.conv('conv3_1',z,128,kernel_size=[3,1],stride=[1,1,1,1],is_pretrain=False)
        z = tools.conv('conv3_2',z,128,kernel_size=[1,3],stride=[1,1,1,1],is_pretrain=False)
        z = tools.conv('conv3_3',z,128,kernel_size=[3,3],stride=[1,1,1,1],is_pretrain=False)
        z = tools.pool('pool3',z,kernel=[1,2,2,1],stride=[1,2,2,1],is_max_pool=True)

        z = tools.conv('conv4_1',z,256,kernel_size=[3,1],stride=[1,1,1,1],is_pretrain=False)
        z = tools.conv('conv4_2',z,256,kernel_size=[1,3],stride=[1,1,1,1],is_pretrain=False)
        z = tools.conv('conv4_3',z,256,kernel_size=[3,3],stride=[1,1,1,1],is_pretrain=False)
        z = tools.pool('pool4',z,kernel=[1,2,2,1],stride=[1,2,2,1],is_max_pool=True)

        z = tools.conv('conv5_1',z,256,kernel_size=[3,1],stride=[1,1,1,1],is_pretrain=False)
        z = tools.conv('conv5_2',z,256,kernel_size=[1,3],stride=[1,1,1,1],is_pretrain=False)
        z = tools.conv('conv5_3',z,256,kernel_size=[3,3],stride=[1,1,1,1],is_pretrain=False)
        z = tools.pool('pool5',z,kernel=[1,2,2,1],stride=[1,2,2,1],is_max_pool=True)

        z = tools.fc_layer('fc6',z,out_nodes=utils.fc6_nodes)
        z = tf.nn.relu(z)
    with tf.variable_scope('predict'):
        total_fc6 = tf.concat([x, y, z], axis = 1)
        fc7 = tools.fc_layer('fc7',total_fc6,out_nodes=utils.fc7_nodes)
        fc7 = tf.nn.relu(fc7)
        result = tools.fc_layer('softmax',fc7,out_nodes=n_class)

    return result
