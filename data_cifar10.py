import cPickle
import numpy as np
import os
from keras.utils.np_utils import to_categorical


IMAGENET_MEAN_RGB = np.array([123.151630838, 115.902882574, 103.062623801]).reshape(3, 1, 1)
DATA_ROOT = os.path.join(os.path.dirname(__file__), 'cifar-10-batches-py')
TRAIN_BATCHES = [os.path.join(DATA_ROOT, batch) for batch in
                 ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5']]
TEST_BATCH = os.path.join(DATA_ROOT, 'test_batch')
NUM_CLASSES = 10


def _load_batch(filename):
    d = cPickle.load(open(filename, 'rb'))
    xs = d['data']
    ys = to_categorical(np.array(d['labels']), NUM_CLASSES)
    xs = xs.reshape(-1, 3, 32, 32)
    xs = (xs - IMAGENET_MEAN_RGB) / 255.0
    # print 'Loading new batch:', xs.shape, ys.shape
    return xs, ys


def load_cifar10(train_batches=TRAIN_BATCHES, test_batch=TEST_BATCH):
    # the ordering is (samples, channels, width, height)
    train_xs = np.zeros((0, 3, 32, 32))
    train_ys = np.zeros((0, NUM_CLASSES), dtype=np.int32)
    for batch in train_batches:
        xs, ys = _load_batch(batch)
        train_xs = np.vstack((train_xs, xs))
        train_ys = np.vstack((train_ys, ys))
    # print train_xs.shape, train_ys.shape
    # order = np.array(range(len(train_ys)))
    # np.random.shuffle(order)
    # print len(order)
    # train_xs = train_xs[order]
    # train_ys = train_ys[order]
    test_xs, test_ys = _load_batch(test_batch)
    return train_xs, train_ys, test_xs, test_ys


if __name__ == '__main__':
    train_xs, train_ys, test_xs, test_ys = load_cifar10()
    print 'Loaded dataset:', train_xs.shape, train_ys.shape
