# -*- coding:utf-8 -*-
import os
from PIL import Image
import numpy as np
import json
current_dir = os.path.dirname(__file__)


def load_data(data_dir, flatten=False):
    train_dir = os.path.join(data_dir, 'train')
    test_dir = os.path.join(data_dir, 'test')

    meta_info = os.path.join(data_dir, 'meta.json')
    with open(meta_info, 'r') as f:
        meta = json.load(f)

    return (
        meta,
        DataSet(
            *_read_images_and_labels(train_dir, flatten=flatten, **meta)),
        DataSet(
            *_read_images_and_labels(test_dir, flatten=flatten, **meta)),
    )


class DataSet:
    """提供 next_batch 方法"""
    def __init__(self, images, labels):
        self._images = images
        self._labels = labels
        self._num_examples = images.shape[0]
        self.ptr = 0

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    def next_batch(self, size=100, shuffle=True):
        if self.ptr + size > self._num_examples:
            self.ptr = 0

        if self.ptr == 0:
            if shuffle:
                perm = np.arange(self._num_examples)
                np.random.shuffle(perm)
                self._images = self._images[perm]
                self._labels = self._labels[perm]

        self.ptr += size
        return (
            self._images[self.ptr - size: self.ptr],
            self._labels[self.ptr - size: self.ptr],
        )


def _read_images_and_labels(dir_name, flatten, ext='.png', **meta):
    images = []
    labels = []
    for fn in os.listdir(dir_name):
        if fn.endswith(ext):
            fd = os.path.join(dir_name, fn)
            images.append(_read_image(fd, flatten=flatten, **meta))
            labels.append(_read_lable(fd, **meta))
    return np.array(images), np.array(labels)


def _read_image(filename, flatten, width, height, **extra_meta):
    im = Image.open(filename).convert('L')
    data = np.asarray(im) / 255.
    if flatten:
        return data.reshape(width * height)
    return data

def _read_lable(filename, label_choices, **extra_meta):
    basename = os.path.basename(filename)
    data_list = []
    for c in basename.split('_')[0]:
        data = np.zeros(len(label_choices))
        idx = label_choices.index(c)
        data[idx] = 1
        data_list.append(data)
    return np.concatenate(data_list)

if __name__ == '__main__':
    meta, train, test = load_data('images/char-2-groups-10')
    print meta
    print train.next()
