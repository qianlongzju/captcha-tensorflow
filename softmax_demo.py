# -*- coding:utf-8 -*-
import argparse
import sys
import tensorflow as tf
import numpy as np

import data_loader

MAX_STEPS = 10000
BATCH_SIZE = 1000

def main(image_dir):
    # load data
    meta, train_data, test_data = data_loader.load_data(image_dir, flatten=True)
    print 'train images: %s. test images: %s' % (train_data.images.shape[0], test_data.images.shape[0])

    NUM_PER_IMAGE = meta['num_per_image']
    LABEL_SIZE = meta['label_size']
    IMAGE_SIZE = meta['width'] * meta['height']
    print 'label_size: %s, image_size: %s' % (LABEL_SIZE, IMAGE_SIZE)

    # variable in the graph for input data
    x = tf.placeholder(tf.float32, [None, IMAGE_SIZE])
    y_ = [tf.placeholder(tf.float32, [None, LABEL_SIZE]) for _ in range(NUM_PER_IMAGE)]

    test_mapping = {label: ys for label, ys in zip(y_, np.split(test_data.labels, NUM_PER_IMAGE, axis=1))}
    test_mapping[x] = test_data.images

    # define the model
    y = [tf.layers.dense(x, LABEL_SIZE) for _ in range(NUM_PER_IMAGE)]

    # Define loss and optimizer
    cross_entropy = 0
    accuracy = 0
    for label, logit in zip(y_, y):
        diff = tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=logit)
        cross_entropy += tf.reduce_mean(diff)

        predict = tf.argmax(logit, axis=1)
        expect = tf.argmax(label, axis=1)
        correct_prediction = tf.equal(predict, expect)
        accuracy += tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) 
   # train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
    train_step = tf.train.AdamOptimizer(0.0001).minimize(cross_entropy)
    accuracy /= NUM_PER_IMAGE

    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        # Train
        for i in range(MAX_STEPS):
            batch_xs, batch_ys = train_data.next_batch(BATCH_SIZE)
            mapping = {label: ys for label, ys in zip(y_, np.split(batch_ys, NUM_PER_IMAGE, axis=1))}
            mapping[x] = batch_xs
            sess.run(train_step, feed_dict=mapping)

            if i % 100 == 0:
                # Test trained model
                r = sess.run(accuracy, feed_dict=test_mapping)
                print 'step = %s, accuracy = %.2f%%' % (i, r * 100)
        # final check after looping
        r_test = sess.run(accuracy, feed_dict=test_mapping)
        print 'testing accuracy = %.2f%%' % (r_test * 100, )


if __name__ == '__main__':
    main(sys.argv[1])
