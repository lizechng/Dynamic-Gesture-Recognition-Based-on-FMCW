import tensorflow as tf
import tools
import utils

def vgg16(net_name,x, n_class, is_pretrain=True):
    with tf.variable_scope(net_name + '_conv_block'):
        x = tools.conv('conv1_1',x,32,kernel_size=[3,3],stride=[1,1,1,1],is_pretrain=is_pretrain)
        x = tools.conv('conv1_2',x,32,kernel_size=[3,3],stride=[1,1,1,1],is_pretrain=is_pretrain)
        x = tools.pool('pool1',x,kernel=[1,2,2,1],stride=[1,2,2,1],is_max_pool=True)

        x = tools.conv('conv2_1',x,64,kernel_size=[3,3],stride=[1,1,1,1],is_pretrain=is_pretrain)
        x = tools.conv('conv2_2',x,64,kernel_size=[3,3],stride=[1,1,1,1],is_pretrain=is_pretrain)
        x = tools.pool('pool2',x,kernel=[1,2,2,1],stride=[1,2,2,1],is_max_pool=True)

        x = tools.conv('conv3_1',x,128,kernel_size=[3,1],stride=[1,1,1,1],is_pretrain=is_pretrain)
        x = tools.conv('conv3_2',x,128,kernel_size=[1,3],stride=[1,1,1,1],is_pretrain=is_pretrain)
        x = tools.conv('conv3_3',x,128,kernel_size=[3,3],stride=[1,1,1,1],is_pretrain=is_pretrain)
        x = tools.pool('pool3',x,kernel=[1,2,2,1],stride=[1,2,2,1],is_max_pool=True)

        x = tools.conv('conv4_1',x,256,kernel_size=[3,1],stride=[1,1,1,1],is_pretrain=is_pretrain)
        x = tools.conv('conv4_2',x,256,kernel_size=[1,3],stride=[1,1,1,1],is_pretrain=is_pretrain)
        x = tools.conv('conv4_3',x,256,kernel_size=[3,3],stride=[1,1,1,1],is_pretrain=is_pretrain)
        x = tools.pool('pool4',x,kernel=[1,2,2,1],stride=[1,2,2,1],is_max_pool=True)

        x = tools.conv('conv5_1',x,256,kernel_size=[3,1],stride=[1,1,1,1],is_pretrain=is_pretrain)
        x = tools.conv('conv5_2',x,256,kernel_size=[1,3],stride=[1,1,1,1],is_pretrain=is_pretrain)
        x = tools.conv('conv5_3',x,256,kernel_size=[3,3],stride=[1,1,1,1],is_pretrain=is_pretrain)
        x = tools.pool('pool5',x,kernel=[1,2,2,1],stride=[1,2,2,1],is_max_pool=True)

        x = tools.fc_layer('fc6',x,out_nodes=utils.fc6_nodes)
        x = tf.nn.relu(x)
        
    with tf.name_scope(net_name + '_fc_block'):
        # x = tools.fc_layer('fc7',x,out_nodes=4096)
        # x = tf.nn.relu(x)
        x = tools.fc_layer('softmax',x,out_nodes=n_class)

    return x


def vgg16N(x, n_class, is_pretrain=True):
    with tf.name_scope('VGG16'):
        x = tools.conv('conv1_1', x, 64, kernel_size=[3,3], stride=[1,1,1,1], is_pretrain=is_pretrain)
        x = tools.conv('conv1_2', x, 64, kernel_size=[3,3], stride=[1,1,1,1], is_pretrain=is_pretrain)
        with tf.name_scope('pool1'):
            x = tools.pool('pool1', x, kernel=[1,2,2,1], stride=[1,2,2,1], is_max_pool=True)

        x = tools.conv('conv2_1', x, 128, kernel_size=[3,3], stride=[1,1,1,1], is_pretrain=is_pretrain)
        x = tools.conv('conv2_2', x, 128, kernel_size=[3,3], stride=[1,1,1,1], is_pretrain=is_pretrain)
        with tf.name_scope('pool2'):
            x = tools.pool('pool2', x, kernel=[1,2,2,1], stride=[1,2,2,1], is_max_pool=True)



        x = tools.conv('conv3_1', x, 256, kernel_size=[3,3], stride=[1,1,1,1], is_pretrain=is_pretrain)
        x = tools.conv('conv3_2', x, 256, kernel_size=[3,3], stride=[1,1,1,1], is_pretrain=is_pretrain)
        x = tools.conv('conv3_3', x, 256, kernel_size=[3,3], stride=[1,1,1,1], is_pretrain=is_pretrain)
        with tf.name_scope('pool3'):
            x = tools.pool('pool3', x, kernel=[1,2,2,1], stride=[1,2,2,1], is_max_pool=True)


        x = tools.conv('conv4_1', x, 512, kernel_size=[3,3], stride=[1,1,1,1], is_pretrain=is_pretrain)
        x = tools.conv('conv4_2', x, 512, kernel_size=[3,3], stride=[1,1,1,1], is_pretrain=is_pretrain)
        x = tools.conv('conv4_3', x, 512, kernel_size=[3,3], stride=[1,1,1,1], is_pretrain=is_pretrain)
        with tf.name_scope('pool4'):
            x = tools.pool('pool4', x, kernel=[1,2,2,1], stride=[1,2,2,1], is_max_pool=True)


        x = tools.conv('conv5_1', x, 512, kernel_size=[3,3], stride=[1,1,1,1], is_pretrain=is_pretrain)
        x = tools.conv('conv5_2', x, 512, kernel_size=[3,3], stride=[1,1,1,1], is_pretrain=is_pretrain)
        x = tools.conv('conv5_3', x, 512, kernel_size=[3,3], stride=[1,1,1,1], is_pretrain=is_pretrain)
        with tf.name_scope('pool5'):
            x = tools.pool('pool5', x, kernel=[1,2,2,1], stride=[1,2,2,1], is_max_pool=True)


        x = tools.fc_layer('fc6', x, out_nodes=4096)
        #with tf.name_scope('batch_norm1'):
            #x = tools.batch_norm(x)
        x = tools.fc_layer('fc7', x, out_nodes=4096)
        #with tf.name_scope('batch_norm2'):
            #x = tools.batch_norm(x)
        x = tools.fc_layer('fc8', x, out_nodes=n_class)

    return x
