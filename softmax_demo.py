# -*- coding:utf-8 -*-
import argparse
import sys
import tensorflow as tf

import data_loader

MAX_STEPS = 10000
BATCH_SIZE = 1000

def main(image_dir):
    # load data
    meta, train_data, test_data = data_loader.load_data(image_dir, flatten=True)
    print 'train images: %s. test images: %s' % (train_data.images.shape[0], test_data.images.shape[0])

    LABEL_SIZE = meta['label_size']
    IMAGE_SIZE = meta['width'] * meta['height']
    print 'label_size: %s, image_size: %s' % (LABEL_SIZE, IMAGE_SIZE)

    # variable in the graph for input data
    x = tf.placeholder(tf.float32, [None, IMAGE_SIZE])
    y1_ = tf.placeholder(tf.float32, [None, LABEL_SIZE])
    y2_ = tf.placeholder(tf.float32, [None, LABEL_SIZE])

    # define the model
    W1 = tf.Variable(tf.zeros([IMAGE_SIZE, LABEL_SIZE]))
    b1 = tf.Variable(tf.zeros([LABEL_SIZE]))
    y1 = tf.matmul(x, W1) + b1

    W2 = tf.Variable(tf.zeros([IMAGE_SIZE, LABEL_SIZE]))
    b2 = tf.Variable(tf.zeros([LABEL_SIZE]))
    y2 = tf.matmul(x, W2) + b2

    # Define loss and optimizer
    diff_1 = tf.nn.softmax_cross_entropy_with_logits(labels=y1_, logits=y1)
    diff_2 = tf.nn.softmax_cross_entropy_with_logits(labels=y2_, logits=y2)
    cross_entropy = tf.reduce_mean(diff_1) + tf.reduce_mean(diff_2)
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

    # forword prop
    predict_1 = tf.argmax(y1, axis=1)
    predict_2 = tf.argmax(y2, axis=1)
    expect_1 = tf.argmax(y1_, axis=1)
    expect_2 = tf.argmax(y2_, axis=1)

    # evaluate accuracy
    correct_prediction_1 = tf.equal(predict_1, expect_1)
    correct_prediction_2 = tf.equal(predict_2, expect_2)
    accuracy = (tf.reduce_mean(tf.cast(correct_prediction_1, tf.float32)) + 
            tf.reduce_mean(tf.cast(correct_prediction_2, tf.float32))) / 2.0

    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        # Train
        for i in range(MAX_STEPS):
            batch_xs, batch_ys_1, batch_ys_2 = train_data.next_batch(BATCH_SIZE)
            sess.run(train_step, feed_dict={x: batch_xs, y1_: batch_ys_1, y2_: batch_ys_2})

            if i % 100 == 0:
                # Test trained model
                r = sess.run(accuracy, feed_dict={x: test_data.images, y1_: test_data.labels_1, y2_:test_data.labels_2})
                print 'step = %s, accuracy = %.2f%%' % (i, r * 100)
        # final check after looping
        r_test = sess.run(accuracy, feed_dict={x: test_data.images, y1_: test_data.labels_1, y2_:test_data.labels_2})
        print 'testing accuracy = %.2f%%' % (r_test * 100, )


if __name__ == '__main__':
    main(sys.argv[1])
