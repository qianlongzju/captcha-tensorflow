# -*- coding:utf-8 -*-
import argparse
import datetime
import sys
import tensorflow as tf
import numpy as np

import data_loader

MAX_STEPS = 10000
BATCH_SIZE = 100

LOG_DIR = 'log/cnn1-run-%s' % datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        # with tf.name_scope('stddev'):
        #    stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        # tf.summary.scalar('stddev', stddev)
        # tf.summary.scalar('max', tf.reduce_max(var))
        # tf.summary.scalar('min', tf.reduce_min(var))
        # tf.summary.histogram('histogram', var)

def main(data_dir):
    # load data
    meta, train_data, test_data = data_loader.load_data(data_dir, flatten=False)
    print 'train images: %s. test images: %s' % (train_data.images.shape[0], test_data.images.shape[0])

    NUM_PER_IMAGE = meta['num_per_image']
    LABEL_SIZE = meta['label_size']
    IMAGE_HEIGHT = meta['height']
    IMAGE_WIDTH = meta['width']
    IMAGE_SIZE = IMAGE_WIDTH * IMAGE_HEIGHT
    print 'label_size: %s, image_size: %s' % (LABEL_SIZE, IMAGE_SIZE)

    # variable in the graph for input data
    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT, IMAGE_WIDTH])
        y_ = [tf.placeholder(tf.float32, [None, LABEL_SIZE]) for _ in range(NUM_PER_IMAGE)]

        # must be 4-D with shape `[batch_size, height, width, channels]`
        x_image = tf.reshape(x, [-1, IMAGE_HEIGHT, IMAGE_WIDTH, 1])
        tf.summary.image('input', x_image, max_outputs=LABEL_SIZE)

    # define the model
    with tf.name_scope('convolution-layer-1'):
        h_conv1 = tf.layers.conv2d(x_image, 32, 7, 1, 'same', activation=tf.nn.relu)
        h_pool1 = tf.layers.max_pooling2d(h_conv1, 2, 2)

    with tf.name_scope('convolution-layer-2'):
        h_conv2 = tf.layers.conv2d(h_pool1, 64, 5, 1, 'same', activation=tf.nn.relu)
        h_pool2 = tf.layers.max_pooling2d(h_conv2, 2, 2)

    with tf.name_scope('densely-connected'):
        h_pool2_flat = tf.reshape(h_pool2, [-1, IMAGE_WIDTH*IMAGE_HEIGHT*4])
        h_fc1 = tf.layers.dense(h_pool2_flat, 1024)

    with tf.name_scope('dropout'):
        # To reduce overfitting, we will apply dropout before the out layer
        tf_is_training = tf.placeholder(tf.bool, None)
        h_fc1_drop = tf.layers.dropout(h_fc1, rate=0.5, training=tf_is_training)

    with tf.name_scope('out'):
        y = [tf.layers.dense(h_fc1_drop, LABEL_SIZE) for _ in range(NUM_PER_IMAGE)]

    # Define loss and optimizer
    # Returns:
    # A 1-D `Tensor` of length `batch_size`
    # of the same type as `logits` with the softmax cross entropy loss.
    with tf.name_scope('loss'):
        cross_entropy = 0
        accuracy = 0
        for label, logit in zip(y_, y):
            diff = tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=logit)
            cross_entropy += tf.reduce_mean(diff)

            predict = tf.argmax(logit, axis=1)
            expect = tf.argmax(label, axis=1)
            correct_prediction = tf.equal(predict, expect)
            accuracy += tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) 
        train_step = tf.train.AdamOptimizer(0.0001).minimize(cross_entropy)
        variable_summaries(cross_entropy)
        accuracy /= NUM_PER_IMAGE
        variable_summaries(accuracy)

    with tf.Session() as sess:

        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(LOG_DIR + '/train', sess.graph)
        test_writer = tf.summary.FileWriter(LOG_DIR + '/test', sess.graph)

        tf.global_variables_initializer().run()

        # Train
        for i in range(MAX_STEPS):
            batch_xs, batch_ys = train_data.next_batch(BATCH_SIZE)
            mapping = {label: ys for label, ys in zip(y_, np.split(batch_ys, NUM_PER_IMAGE, axis=1))}
            mapping[x] = batch_xs
            mapping[tf_is_training] = True

            step_summary, _ = sess.run([merged, train_step], feed_dict=mapping)
            train_writer.add_summary(step_summary, i)

            if i % 100 == 0:
                # Test trained model
                mapping[tf_is_training] = False
                valid_summary, train_accuracy = sess.run([merged, accuracy], feed_dict=mapping)
                train_writer.add_summary(valid_summary, i)

                # final check after looping
                test_x, test_y = test_data.next_batch(2000)
                mapping = {label: ys for label, ys in zip(y_, np.split(test_y, NUM_PER_IMAGE, axis=1))}
                mapping[x] = test_x
                mapping[tf_is_training] = False
                test_summary, test_accuracy = sess.run([merged, accuracy], feed_dict=test_mapping)
                test_writer.add_summary(test_summary, i)

                print 'step %s, training accuracy = %.2f%%, testing accuracy = %.2f%%' % (i, train_accuracy * 100, test_accuracy * 100)

        train_writer.close()
        test_writer.close()

        # final check after looping
        test_x, test_y = test_data.next_batch(2000)
        mapping = {label: ys for label, ys in zip(y_, np.split(test_y, NUM_PER_IMAGE, axis=1))}
        mapping[x] = test_x
        mapping[tf_is_training] = False
        test_accuracy = accuracy.eval(feed_dict=mapping)
        print 'testing accuracy = %.2f%%' % (test_accuracy * 100, )


if __name__ == '__main__':
    main(sys.argv[1])
