from PIL import Image
import tensorflow as tf
import os
import matplotlib.pyplot as plt
import utils
import numpy as np

def _int64_feature(value):
    '''
    Convert to int64.
    '''
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    '''
    Convert to bytes.
    '''
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def createTrainTFRecord(filename, mapfile):
    '''
    Images to tfrecords file.
    : tfrecords store path.
    : path of map of idx and class name.
    '''
    class_map = {}
    Infor = ['ATM/', 'RDM/']
    classes = ['QH/', 'SX/', 'XZ/', 'ZY/']
    name = ['qh-','sx-','xz-','zy-']
    CH = 'Train/'
    writer = tf.python_io.TFRecordWriter(filename)

    for i in range(4):
        for j in range(1):
            ATM_path = Infor[0] + classes[i] + CH
            RDM_path = Infor[1] + classes[i] + CH
            class_map[i] = classes[i]
            for cnt in range(80):
                RDM_list = []
                for k in range(32):
                    img_name = name[i] + str(1+cnt*8+k) + '.png'
                    img_path = RDM_path + img_name
                    img = Image.open(img_path)
                    img = img.convert("RGB")
                    img = img.resize((utils.img_width,utils.img_height))
                    img = np.array(img)
                    RDM_list.append(img)
                im = np.array(RDM_list)
                RDM_raw = im.tobytes()
                path_ATM = ATM_path + str(cnt) + '.png'
                img_ATM = Image.open(path_ATM)
                img_ATM = img_ATM.convert("RGB")
                img_ATM = img_ATM.resize((utils.img_width,utils.img_height))
                ATM_raw = img_ATM.tobytes()
                example = tf.train.Example(features=tf.train.Features(feature={
                    'label': _int64_feature(i),
                    'RDM_raw': _bytes_feature(RDM_raw),
                    'ATM_raw': _bytes_feature(ATM_raw)
                }))
                writer.write(example.SerializeToString())
    writer.close()
    print('tfRecord file write successfully.')

    txtfile = open(mapfile, 'w+')
    for key in class_map.keys():
        txtfile.writelines(str(key) + ' : ' + class_map[key] + '\n')
    txtfile.close()
    print('Class map file write successfully.')

def createTestTFRecord(filename, mapfile):
    '''
    Images to tfrecords file.
    : tfrecords store path.
    : path of map of idx and class name.
    '''
    class_map = {}
    Infor = ['ATM/', 'RDM/']
    classes = ['QH/', 'SX/', 'XZ/', 'ZY/']
    name = ['qh-','sx-','xz-','zy-']
    CH = 'Test/'
    writer = tf.python_io.TFRecordWriter(filename)

    for i in range(4):
        for j in range(1):
            ATM_path = Infor[0] + classes[i] + CH
            RDM_path = Infor[1] + classes[i] + CH
            class_map[i] = classes[i]
            for cnt in range(17):
                RDM_list = []
                for k in range(32):
                    img_name = name[i] + str(1+cnt*8+k+640) + '.png'
                    img_path = RDM_path + img_name
                    img = Image.open(img_path)
                    img = img.convert("RGB")
                    img = img.resize((utils.img_width,utils.img_height))
                    img = np.array(img)
                    RDM_list.append(img)
                im = np.array(RDM_list)
                RDM_raw = im.tobytes()
                path_ATM = ATM_path + str(cnt) + '.png'
                img_ATM = Image.open(path_ATM)
                # img_ATM.show()
                img_ATM = img_ATM.convert("RGB")
                img_ATM = img_ATM.resize((utils.img_width,utils.img_height))
                ATM_raw = img_ATM.tobytes()
                example = tf.train.Example(features=tf.train.Features(feature={
                    'label': _int64_feature(i),
                    'RDM_raw': _bytes_feature(RDM_raw),
                    'ATM_raw': _bytes_feature(ATM_raw)
                }))
                writer.write(example.SerializeToString())
    writer.close()
    print('tfRecord file write successfully.')

    txtfile = open(mapfile, 'w+')
    for key in class_map.keys():
        txtfile.writelines(str(key) + ' : ' + class_map[key] + '\n')
    txtfile.close()
    print('Class map file write successfully.')

def readTFRecord(filename):
    '''
    Read the tfRecord file created.
    : the tfRecord file path.
    '''
    reader = tf.TFRecordReader()
    filename_queue = tf.train.string_input_producer([filename])
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,features={
                                       'label': tf.FixedLenFeature([], tf.int64),
                                       'RDM_raw': tf.FixedLenFeature([], tf.string),
                                       'ATM_raw': tf.FixedLenFeature([], tf.string)
                                       })
    img_RDM = tf.decode_raw(features['RDM_raw'], tf.uint8)
    img_ATM = tf.decode_raw(features['ATM_raw'], tf.uint8)
    img_RDM = tf.reshape(img_RDM, [utils.img_depth,utils.img_width,utils.img_height,utils.img_channels])
    img_ATM = tf.reshape(img_ATM, [utils.img_width,utils.img_height,utils.img_channels])

    # img = tf.image.per_image_standardization(img)
    # img = img / 255.0
    label = tf.cast(features['label'], tf.int32)

    return img_RDM, img_ATM, label


def createBatch(filename, batch_size):
    '''
    Create img_batch and label_batch
    : tfrecords file path.
    : batch number.
    '''
    images, labels = readTFRecord(filename)

    img_batch, label_batch = tf.train.batch([images, labels],
                                            batch_size=batch_size)

    label_batch = tf.one_hot(label_batch, depth=utils.num_class)

    return img_batch, label_batch

def createShuffleBatch(filename, batch_size):
    '''
    Create img_batch and label_batch
    : tfrecords file path.
    : batch number.
    '''
    imagesRDM, imagesATM, labels = readTFRecord(filename)

    min_after_dequeue = 1000
    capacity = min_after_dequeue + 3 * batch_size

    imgRDM_batch, imgATM_batch, label_batch = tf.train.shuffle_batch([imagesRDM, imagesATM, labels],
                                                    batch_size=batch_size,
                                                    seed=12345,
                                                    capacity=capacity,
                                                    min_after_dequeue=min_after_dequeue)
    label_batch = tf.one_hot(label_batch, depth=utils.num_class)

    return imgRDM_batch, imgATM_batch, label_batch


def test():
    '''
    Testing above functions
    '''
    # img, label = createBatch('../tfRecords/test/test.DTM.tfrecords',10)
    # img2, label2 = createBatch('../tfRecords/test/test.RTM.tfrecords',10)

    # img, label = readTFRecord('../tfRecords/test/test.DTM.tfrecords')

    img1, img2, label = readTFRecord('../tfRecords/Test/test.tfrecords')

    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        txtfile = open('txt.md', 'w+')
        for i in range(2):
            example1, example2, l = sess.run([img1, img2, label])
            print('label = {}'.format(l))
            # txtfile.writelines(str(l) + ' : ' + str(l2) + '\n')
            # print(example1)
            print(example1.shape)
            print(type(example1))
            image1 = Image.fromarray(example2, 'RGB')
            # image2 = Image.fromarray(example2, 'RGB')
            image1.show()
            # image2.show()
        txtfile.close()
        coord.request_stop()
        coord.join(threads)

def main():

    createTrainTFRecord('./tfRecords/Train/train.tfrecords','./tfRecords/readme.train.md')
    createTestTFRecord ('./tfRecords/Test/test.tfrecords'  ,'./tfRecords/readme.test.md')
    # createTrainTFRecord('./RTM/','./tfRecords/Train/train.RTM.tfrecords','./tfRecords/readme.RTM.train.md')
    # createTestTFRecord ('./RTM/','./tfRecords/Test/test.RTM.tfrecords'  ,'./tfRecords/readme.RTM.test.md')
    # createTrainTFRecord('./ATM/','./tfRecords/Train/train.ATM.tfrecords','./tfRecords/readme.ATM.train.md')
    # createTestTFRecord ('./ATM/','./tfRecords/Test/test.ATM.tfrecords'  ,'./tfRecords/readme.ATM.test.md')

    # createAllTrainTFRecord('./ATM/', './DTM/', './RTM/', './tfRecords/Train/train.tfrecords')
    # createAllTestTFRecord ('./ATM/', './DTM/', './RTM/', './tfRecords/Test/test.tfrecords')

if __name__ == '__main__':
    main()
