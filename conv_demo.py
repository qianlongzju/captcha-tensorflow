# -*- coding:utf-8 -*-
import argparse
import datetime
import sys
import tensorflow as tf

import data_loader

MAX_STEPS = 10000
BATCH_SIZE = 100

LOG_DIR = 'log/cnn1-run-%s' % datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

FLAGS = None


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


#def weight_variable(shape):
#    initial = tf.truncated_normal(shape, stddev=0.1)
#    return tf.Variable(initial)
#
#
#def bias_variable(shape):
#    initial = tf.constant(0.1, shape=shape)
#    return tf.Variable(initial)
#
#
#def conv2d(x, W):
#    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
#
#
#def max_pool_2x2(x):
#    return tf.nn.avg_pool(x, ksize=[1, 2, 2, 1],
#                          strides=[1, 2, 2, 1], padding='SAME')
#

def main(data_dir):
    # load data
    meta, train_data, test_data = data_loader.load_data(data_dir, flatten=False)
    print 'data loaded'
    print 'train images: %s. test images: %s' % (train_data.images.shape[0], test_data.images.shape[0])

    LABEL_SIZE = meta['label_size']
    IMAGE_HEIGHT = meta['height']
    IMAGE_WIDTH = meta['width']
    IMAGE_SIZE = IMAGE_WIDTH * IMAGE_HEIGHT
    print 'label_size: %s, image_size: %s' % (LABEL_SIZE, IMAGE_SIZE)

    # variable in the graph for input data
    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT, IMAGE_WIDTH])
        y1_ = tf.placeholder(tf.float32, [None, LABEL_SIZE])
        y2_ = tf.placeholder(tf.float32, [None, LABEL_SIZE])

        # must be 4-D with shape `[batch_size, height, width, channels]`
        x_image = tf.reshape(x, [-1, IMAGE_HEIGHT, IMAGE_WIDTH, 1])
        tf.summary.image('input', x_image, max_outputs=LABEL_SIZE)

    # define the model
    with tf.name_scope('convolution-layer-1'):
        #W_conv1 = weight_variable([3, 3, 1, 32])
        #b_conv1 = bias_variable([32])

        #h_conv1 = tf.nn.tanh(conv2d(x_image, W_conv1) + b_conv1)
        #h_pool1 = max_pool_2x2(h_conv1)

        h_conv1 = tf.layers.conv2d(x_image, 32, 7, 1, 'same', activation=tf.nn.relu)
        h_pool1 = tf.layers.max_pooling2d(h_conv1, 2, 2)

    with tf.name_scope('convolution-layer-2'):
        #W_conv2 = weight_variable([3, 3, 32, 64])
        #b_conv2 = bias_variable([64])

        #h_conv2 = tf.nn.tanh(conv2d(h_pool1, W_conv2) + b_conv2)
        #h_pool2 = max_pool_2x2(h_conv2)

        h_conv2 = tf.layers.conv2d(h_pool1, 64, 5, 1, 'same', activation=tf.nn.relu)
        h_pool2 = tf.layers.max_pooling2d(h_conv2, 2, 2)

    with tf.name_scope('densely-connected'):
        h_pool2_flat = tf.reshape(h_pool2, [-1, IMAGE_WIDTH*IMAGE_HEIGHT*4])

        #W1_fc1 = weight_variable([IMAGE_WIDTH * IMAGE_HEIGHT * 4, 1024])
        #b1_fc1 = bias_variable([1024])
        #h1_fc1 = tf.nn.tanh(tf.matmul(h_pool2_flat, W1_fc1) + b1_fc1)
        h_fc1 = tf.layers.dense(h_pool2_flat, 1024)

        #W2_fc1 = weight_variable([IMAGE_WIDTH * IMAGE_HEIGHT * 4, 1024])
        #b2_fc1 = bias_variable([1024])
        #h2_fc1 = tf.nn.tanh(tf.matmul(h_pool2_flat, W2_fc1) + b2_fc1)
        #h2_fc1 = tf.layers.dense(h_pool2_flat, 1024)


    with tf.name_scope('dropout'):
        # To reduce overfitting, we will apply dropout before the readout layer
        keep_prob = tf.placeholder(tf.float32)
        tf_is_training = tf.placeholder(tf.bool, None)
        #h1_fc1_drop = tf.nn.dropout(h1_fc1, keep_prob)
        #h2_fc1_drop = tf.nn.dropout(h2_fc1, keep_prob)
        h_fc1_drop = tf.layers.dropout(h_fc1, rate=0.5, training=tf_is_training)

    with tf.name_scope('readout'):
        #W1_fc2 = weight_variable([1024, LABEL_SIZE])
        #b1_fc2 = bias_variable([LABEL_SIZE])
        #y1_conv = tf.matmul(h1_fc1_drop, W1_fc2) + b1_fc2
        y1_conv = tf.layers.dense(h_fc1_drop, LABEL_SIZE)

        #W2_fc2 = weight_variable([1024, LABEL_SIZE])
        #b2_fc2 = bias_variable([LABEL_SIZE])
        #y2_conv = tf.matmul(h2_fc1_drop, W2_fc2) + b2_fc2
        y2_conv = tf.layers.dense(h_fc1_drop, LABEL_SIZE)

    # Define loss and optimizer
    # Returns:
    # A 1-D `Tensor` of length `batch_size`
    # of the same type as `logits` with the softmax cross entropy loss.
    with tf.name_scope('loss'):
        cross_entropy = (
            tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y1_, logits=y1_conv)) 
            + 
            tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y2_, logits=y2_conv))
            )
        train_step = tf.train.AdamOptimizer(0.0001).minimize(cross_entropy)
        variable_summaries(cross_entropy)

    # forword prop
    predict_1 = tf.argmax(y1_conv, axis=1)
    predict_2 = tf.argmax(y2_conv, axis=1)
    expect_1 = tf.argmax(y1_, axis=1)
    expect_2 = tf.argmax(y2_, axis=1)

    # evaluate accuracy
    with tf.name_scope('evaluate_accuracy'):
        correct_prediction_1 = tf.equal(predict_1, expect_1)
        correct_prediction_2 = tf.equal(predict_2, expect_2)
        accuracy = (
                tf.reduce_mean(tf.cast(correct_prediction_1, tf.float32)) 
                +
                tf.reduce_mean(tf.cast(correct_prediction_2, tf.float32))
                ) / 2.0
        variable_summaries(accuracy)

    with tf.Session() as sess:

        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(LOG_DIR + '/train', sess.graph)
        test_writer = tf.summary.FileWriter(LOG_DIR + '/test', sess.graph)

        tf.global_variables_initializer().run()

        # Train
        for i in range(MAX_STEPS):
            batch_xs, batch_ys_1, batch_ys_2 = train_data.next_batch(BATCH_SIZE)
            #print len(batch_xs), len(batch_ys_1), len(batch_ys_2)

            step_summary, _ = sess.run([merged, train_step], feed_dict=
                    {x: batch_xs, y1_: batch_ys_1, y2_: batch_ys_2, tf_is_training:True})
            train_writer.add_summary(step_summary, i)

            if i % 100 == 0:
                # Test trained model
                valid_summary, train_accuracy = sess.run([merged, accuracy], feed_dict={x: batch_xs, y1_: batch_ys_1, y2_: batch_ys_2, tf_is_training: False})
                train_writer.add_summary(valid_summary, i)

                # final check after looping
                test_x, test_y1, test_y2 = test_data.next_batch(200)
                test_summary, test_accuracy = sess.run([merged, accuracy], feed_dict={x: test_x, y1_: test_y1, y2_: test_y2, tf_is_training: False})
                test_writer.add_summary(test_summary, i)

                print 'step %s, training accuracy = %.2f%%, testing accuracy = %.2f%%' % (i, train_accuracy * 100, test_accuracy * 100)

        train_writer.close()
        test_writer.close()

        # final check after looping
        test_x, test_y1, test_y2 = test_data.next_batch(2000)
        test_accuracy = accuracy.eval(feed_dict={x: test_x, y1_: test_y1, y2_: test_y2, keep_prob: 1.0})
        print 'testing accuracy = %.2f%%' % (test_accuracy * 100, )


if __name__ == '__main__':
    main(sys.argv[1])
