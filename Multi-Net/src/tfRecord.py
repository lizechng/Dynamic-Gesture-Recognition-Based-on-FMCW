from PIL import Image
import tensorflow as tf
import os
import matplotlib.pyplot as plt
import utils

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

def createTrainTFRecord(data_dir,filename, mapfile):
    '''
    Images to tfrecords file.
    : tfrecords store path.
    : path of map of idx and class name.
    '''
    class_map = {}
    classes = ['QH/', 'SX/', 'XZ/', 'ZY/']
    CH = 'Train/'
    writer = tf.python_io.TFRecordWriter(filename)

    for i in range(4):
        for j in range(1):
            class_path = data_dir + classes[i] + CH
            class_map[i] = classes[i]
            for img_name in os.listdir(class_path):
                img_path = class_path + img_name
                img = Image.open(img_path)
                img = img.convert("RGB")
                img = img.resize((utils.img_width,utils.img_height))
                img_raw = img.tobytes()
                example = tf.train.Example(features=tf.train.Features(feature={
                    'label': _int64_feature(i),
                    'img_raw': _bytes_feature(img_raw)
                }))
                writer.write(example.SerializeToString())
    writer.close()
    print('tfRecord file write successfully.')

    txtfile = open(mapfile, 'w+')
    for key in class_map.keys():
        txtfile.writelines(str(key) + ' : ' + class_map[key] + '\n')
    txtfile.close()
    print('Class map file write successfully.')

def createTestTFRecord(data_dir,filename, mapfile):
    '''
    Images to tfrecords file.
    : tfrecords store path.
    : path of map of idx and class name.
    '''
    class_map = {}
    classes = ['QH/', 'SX/', 'XZ/', 'ZY/']
    CH = 'Test/'
    writer = tf.python_io.TFRecordWriter(filename)

    for i in range(4):
        class_path = data_dir + classes[i] + CH
        class_map[i] = classes[i]
        for img_name in os.listdir(class_path):
            img_path = class_path + img_name
            img = Image.open(img_path)
            img = img.convert("RGB")
            img = img.resize((utils.img_width,utils.img_height))
            img_raw = img.tobytes()
            example = tf.train.Example(features=tf.train.Features(feature={
                'label': _int64_feature(i),
                'img_raw': _bytes_feature(img_raw)
            }))
            writer.write(example.SerializeToString())
    writer.close()
    print('tfRecord file write successfully.')

    txtfile = open(mapfile, 'w+')
    for key in class_map.keys():
        txtfile.writelines(str(key) + ' : ' + class_map[key] + '\n')
    txtfile.close()
    print('Class map file write successfully.')

def createAllTrainTFRecord(ATM_dir, DTM_dir, RTM_dir, filename):
    '''
    : data_dir1 for DTM file path
    : data_dir2 for RTM file path
    '''
    classes = ['QH', 'SX', 'XZ', 'ZY']
    CH = 'Train/'
    writer = tf.python_io.TFRecordWriter(filename)

    for i in range(4):
        for j in range(1):
            class_path1 = ATM_dir + classes[i] + '/' + CH
            class_path2 = DTM_dir + classes[i] + '/' + CH
            class_path3 = RTM_dir + classes[i] + '/' + CH
            for img_name1, img_name2, img_name3 in zip(os.listdir(class_path1), os.listdir(class_path2), os.listdir(class_path3)):
                img_path1 = class_path1 + img_name1
                img_path2 = class_path2 + img_name2
                img_path3 = class_path3 + img_name3
                img1 = Image.open(img_path1)
                img2 = Image.open(img_path2)
                img3 = Image.open(img_path3)
                img1 = img1.convert('RGB')
                img2 = img2.convert('RGB')
                img3 = img3.convert('RGB')
                img1 = img1.resize((utils.img_width,utils.img_height))
                img2 = img2.resize((utils.img_width,utils.img_height))
                img3 = img3.resize((utils.img_width,utils.img_height))
                img_raw1 = img1.tobytes()
                img_raw2 = img2.tobytes()
                img_raw3 = img3.tobytes()
                example = tf.train.Example(features=tf.train.Features(feature={
                    'label': _int64_feature(i),
                    'img_ATM': _bytes_feature(img_raw1),
                    'img_DTM': _bytes_feature(img_raw2),
                    'img_RTM': _bytes_feature(img_raw3)
                }))
                writer.write(example.SerializeToString())
    writer.close()
    print('tfrecords file write successfully.')

def createAllTestTFRecord(ATM_dir, DTM_dir, RTM_dir, filename):
    '''
    : data_dir1 for DTM file path
    : data_dir2 for RTM file path
    '''
    classes = ['QH', 'SX', 'XZ', 'ZY']
    CH = 'Test/'
    writer = tf.python_io.TFRecordWriter(filename)

    for i in range(4):
        for j in range(1):
            class_path1 = ATM_dir + classes[i] + '/' + CH
            class_path2 = DTM_dir + classes[i] + '/' + CH
            class_path3 = RTM_dir + classes[i] + '/' + CH
            for img_name1, img_name2, img_name3 in zip(os.listdir(class_path1), os.listdir(class_path2), os.listdir(class_path3)):
                img_path1 = class_path1 + img_name1
                img_path2 = class_path2 + img_name2
                img_path3 = class_path3 + img_name3
                img1 = Image.open(img_path1)
                img2 = Image.open(img_path2)
                img3 = Image.open(img_path3)
                img1 = img1.convert('RGB')
                img2 = img2.convert('RGB')
                img3 = img3.convert('RGB')
                img1 = img1.resize((utils.img_width,utils.img_height))
                img2 = img2.resize((utils.img_width,utils.img_height))
                img3 = img3.resize((utils.img_width,utils.img_height))
                img_raw1 = img1.tobytes()
                img_raw2 = img2.tobytes()
                img_raw3 = img3.tobytes()
                example = tf.train.Example(features=tf.train.Features(feature={
                    'label': _int64_feature(i),
                    'img_ATM': _bytes_feature(img_raw1),
                    'img_DTM': _bytes_feature(img_raw2),
                    'img_RTM': _bytes_feature(img_raw3)
                }))
                writer.write(example.SerializeToString())
    writer.close()
    print('tfrecords file write successfully.')

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
                                       'img_raw': tf.FixedLenFeature([], tf.string)
                                       })
    img = tf.decode_raw(features['img_raw'], tf.uint8)
    img = tf.reshape(img, [utils.img_width,utils.img_height,utils.img_channels])

    # img = tf.image.per_image_standardization(img)
    # img = img / 255.0
    label = tf.cast(features['label'], tf.int32)

    return img, label

def readAllTFRecord(filename):
    '''
    Read the tfRecord file created.
    : the tfRecord file path.
    '''
    reader = tf.TFRecordReader()
    filename_queue = tf.train.string_input_producer([filename])
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,features={
                                       'label': tf.FixedLenFeature([], tf.int64),
                                       'img_ATM': tf.FixedLenFeature([], tf.string),
                                       'img_DTM': tf.FixedLenFeature([], tf.string),
                                       'img_RTM': tf.FixedLenFeature([], tf.string)
                                       })
    img1 = tf.decode_raw(features['img_DTM'], tf.uint8)
    img2 = tf.decode_raw(features['img_RTM'], tf.uint8)
    img3 = tf.decode_raw(features['img_ATM'], tf.uint8)
    img1 = tf.reshape(img1, [utils.img_width,utils.img_height,utils.img_channels])
    img2 = tf.reshape(img2, [utils.img_width,utils.img_height,utils.img_channels])
    img3 = tf.reshape(img3, [utils.img_width,utils.img_height,utils.img_channels])

    label = tf.cast(features['label'], tf.int32)

    return img1, img2, img3, label

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
    images, labels = readTFRecord(filename)

    min_after_dequeue = 1000
    capacity = min_after_dequeue + 3 * batch_size

    img_batch, label_batch = tf.train.shuffle_batch([images, labels],
                                                    batch_size=batch_size,
                                                    seed=12345,
                                                    capacity=capacity,
                                                    min_after_dequeue=min_after_dequeue)
    label_batch = tf.one_hot(label_batch, depth=utils.num_class)

    return img_batch, label_batch

def createAllShuffleBatch(filename, batch_size):
    '''
    Create img_batch and label_batch
    : tfrecords file path.
    : batch number.
    '''
    images1, images2, image3, labels = readAllTFRecord(filename)

    min_after_dequeue = 10
    capacity = min_after_dequeue + 3 * batch_size

    img_batch1, img_batch2, img_batch3, label_batch = tf.train.shuffle_batch([images1, images2, image3, labels],
                                                                batch_size=batch_size,
                                                                seed=12345,
                                                                capacity=capacity,
                                                                min_after_dequeue=min_after_dequeue)
    label_batch = tf.one_hot(label_batch, depth=utils.num_class)

    return img_batch1, img_batch2, img_batch3, label_batch
def createAllBatch(filename, batch_size):
    images1, images2, images3, labels = readAllTFRecord(filename)
    
    min_after_dequeue = 10
    capacity = min_after_dequeue + 3 * batch_size

    img_batch1, img_batch2, img_batch3, label_batch = tf.train.batch([images1, images2, images3, labels],
                                                        batch_size=batch_size)
    label_batch = tf.one_hot(label_batch, depth=utils.num_class)

    return img_batch1, img_batch2, img_batch3, label_batch

def test():
    '''
    Testing above functions
    '''
    # img, label = createBatch('../tfRecords/test/test.DTM.tfrecords',10)
    # img2, label2 = createBatch('../tfRecords/test/test.RTM.tfrecords',10)

    # img, label = readTFRecord('../tfRecords/test/test.DTM.tfrecords')

    img1, img2, _, label = readAllTFRecord('./tfRecords/Test/test.tfrecords')

    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        txtfile = open('txt.md', 'w+')
        for i in range(2):
            example1, example2, _, l = sess.run([img1, img2, label])
            print('label = {}'.format(l))
            # txtfile.writelines(str(l) + ' : ' + str(l2) + '\n')
            image1 = Image.fromarray(example1, 'RGB')
            # image2 = Image.fromarray(example2, 'RGB')
            image1.show()
            # image2.show()
        txtfile.close()
        coord.request_stop()
        coord.join(threads)

def main():

    # createTrainTFRecord('./DTM/','./tfRecords/Train/train.DTM.tfrecords','./tfRecords/readme.DTM.train.md')
    # createTestTFRecord ('./DTM/','./tfRecords/Test/test.DTM.tfrecords'  ,'./tfRecords/readme.DTM.test.md')
    # createTrainTFRecord('./RTM/','./tfRecords/Train/train.RTM.tfrecords','./tfRecords/readme.RTM.train.md')
    # createTestTFRecord ('./RTM/','./tfRecords/Test/test.RTM.tfrecords'  ,'./tfRecords/readme.RTM.test.md')
    # createTrainTFRecord('./ATM/','./tfRecords/Train/train.ATM.tfrecords','./tfRecords/readme.ATM.train.md')
    # createTestTFRecord ('./ATM/','./tfRecords/Test/test.ATM.tfrecords'  ,'./tfRecords/readme.ATM.test.md')

    createAllTrainTFRecord('./ATM/', './DTM/', './RTM/', './tfRecords/Train/train.tfrecords')
    createAllTestTFRecord ('./ATM/', './DTM/', './RTM/', './tfRecords/Test/test.tfrecords')

if __name__ == '__main__':
    main()
