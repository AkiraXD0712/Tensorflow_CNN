import os
import tensorflow as tf
from PIL import Image

NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 32
NUM_CLASS = 4
IMAGE_SIZE = 64
CHANNELS =3


def load_data(data_dir, classes):
    """Read images and save as TFRecord

        Args:
            data_dir: Path to the data directory.
            classes: a tuple of people

        Return:
            Nothing
    """
    writer = tf.python_io.TFRecordWriter(data_dir + 'data.tfRecord')
    for index, name in enumerate(classes):
        class_path = data_dir + name + '\\'
        for img_name in os.listdir(class_path):
            img_path = class_path + img_name
            img = Image.open(img_path)
            img_raw = img.tobytes()
            example = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
                        'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
                    }
                )
            )
            writer.write(example.SerializeToString())  # 序列化为字符串
    writer.close()


def read_data(filename_queue):
    """Reads and parses examples from data files.

    Recommendation: if you want N-way read parallelism, call this function
    N times.  This will give you N independent Readers reading different
    files & positions within those files, which will give better mixing of
    examples.

    Args:
        filename_queue: A queue of strings with the filenames to read from.

    Returns:
        An object representing a single example, with the following fields:
            height: number of rows in the result (32)
            width: number of columns in the result (32)
            depth: number of color channels in the result (3)
            key: a scalar string Tensor describing the filename & record number
            for this example.
            label: an int32 Tensor with the label in the range 0..3.
            uint8image: a [height, width, depth] uint8 Tensor with the image data
    """
    with tf.name_scope("read_data"):
        class DataRecord(object):
            pass

        result = DataRecord()

        # dimension of the images in dataset
        result.height = 64
        result.width = 64
        result.depth = 3

        # Read a record, getting filenames from the filename_queue.
        reader = tf.TFRecordReader()
        result.key, value = reader.read(filename_queue)

        features = tf.parse_single_example(value,
                                           features={
                                               'label': tf.FixedLenFeature([], tf.int64),
                                               'img_raw': tf.FixedLenFeature([], tf.string),
                                           })

        # Convert from a string to a vector of uint8 that is record_bytes long.
        result.uint8image = tf.decode_raw(features['img_raw'], tf.uint8)
        result.uint8image = tf.reshape(result.uint8image, [result.height, result.width, result.depth])
        print(result.uint8image)
        result.uint8image = tf.cast(result.uint8image, tf.float32) * (1. / 255) - 0.5

        # The first bytes represent the label, which we convert from uint8->int32.
        result.label = tf.cast(features['label'], tf.int32)
        result.label = tf.one_hot(result.label, depth=NUM_CLASS, axis=-1)
        result.label = tf.reshape(result.label, [NUM_CLASS])

        return result


def generate_image_and_label_batch(image, label, min_queue_examples, batch_size, shuffle):
    """Construct a queued batch of images and labels.

    Args:
        image: 3-D Tensor of [height, width, depth] of type.float32.
        label: 1-D Tensor of type.int32
        min_queue_examples: int32, minimum number of samples to retain
        in the queue that provides of batches of examples.
        batch_size: Number of images per batch.
        shuffle: boolean indicating whether to use a shuffling queue.

    Returns:
        images: Images. 4D tensor of [batch_size, height, width, depth] size.
        labels: Labels. 1D tensor of [batch_size] size.
    """
    # Create a queue that shuffles the examples, and then
    # read 'batch_size' images + labels from the example queue.
    num_preprocess_threads = 16
    if shuffle:
        images, label_batch = tf.train.shuffle_batch(
            [image, label],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size,
            min_after_dequeue=min_queue_examples)
    else:
        images, label_batch = tf.train.batch(
            [image, label],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size)

    # Display the training images in the visualizer.
    tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS)
    tf.summary.image('images', images)

    return images, tf.reshape(label_batch, [batch_size, NUM_CLASS])


def distorted_inputs(data_dir, batch_size):
    """Construct distorted input for training using the Reader ops.

    Args:
        data_dir: Path to the data directory.
        batch_size: Number of images per batch.

    Returns:
        images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, depth] size.
        labels: Labels. 1D tensor of [batch_size] size.
    """
    with tf.name_scope("distorted_inputs"):
        filename = [os.path.join(data_dir, 'data.tfRecord')]
        for f in filename:
            if not tf.gfile.Exists(f):
                raise ValueError('Failed to find file: ' + f)

        # Create a queue that produces the filename to read.
        filename_queue = tf.train.string_input_producer(filename, num_epochs=None, name='file_queue')

        # Read examples from files in the filename queue.
        read_input = read_data(filename_queue)
        reshaped_image = tf.cast(read_input.uint8image, tf.float32)

        height = IMAGE_SIZE
        width = IMAGE_SIZE
        depth = CHANNELS

        # Image processing for training the network. Note the many random
        # distortions applied to the image.

        # Randomly crop a [height, width] section of the image.
        distorted_image = tf.random_crop(reshaped_image, [height, width, depth])

        # Randomly flip the image horizontally.
        distorted_image = tf.image.random_flip_left_right(distorted_image)

        # Because these operations are not commutative, consider randomizing
        # the order their operation.
        distorted_image = tf.image.random_brightness(distorted_image,
                                                     max_delta=63)
        distorted_image = tf.image.random_contrast(distorted_image,
                                                   lower=0.2,
                                                   upper=1.8)

        # Subtract off the mean and divide by the variance of the pixels.
        float_image = tf.image.per_image_standardization(distorted_image)

        # with tf.Session() as sess:
        #     print(sess.run([float_image, read_input.label]))

        # Set the shapes of tensors.
        float_image.set_shape([height, width, depth])
        read_input.label.set_shape([NUM_CLASS])

        # Ensure that the random shuffling has good mixing properties.
        min_fraction_of_examples_in_queue = 0.4
        min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN * min_fraction_of_examples_in_queue)

        # Generate a batch of images and labels by building up a queue of examples.
        return generate_image_and_label_batch(float_image, read_input.label,
                                              min_queue_examples, batch_size,
                                              shuffle=True)


if __name__ == '__main__':
    # load_data('C:\\Users\Akira.DESKTOP-HM7OVCC\Desktop\data\\', ('ali', 'david', 'jordan', 'laurent'))
    # filename = 'C:\\Users\Akira.DESKTOP-HM7OVCC\Desktop\data\\data.tfrecord'
    # filename_queue = tf.train.string_input_producer([filename])
    # result = read_data(filename_queue)
    #
    # img_batch, label_batch = tf.train.shuffle_batch([result.uint8image, result.label],
    #                                                 batch_size=30, capacity=2000,
    #                                                 min_after_dequeue=1000)
    # sess = tf.Session()
    # init = tf.global_variables_initializer()
    # sess.run(init)
    # threads = tf.train.start_queue_runners(sess=sess)
    # img, label = sess.run([img_batch, label_batch])
    # print(label)


    img_batch, label_batch = distorted_inputs('C:\\Users\Akira.DESKTOP-HM7OVCC\Desktop\data\\', 20)

    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    threads = tf.train.start_queue_runners(sess=sess)

    for i in range(8):
        img, label = sess.run([img_batch, label_batch])
        print(label)