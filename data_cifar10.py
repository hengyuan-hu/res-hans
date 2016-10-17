import cPickle
import numpy as np
import os
from keras.utils.np_utils import to_categorical
from keras.datasets import cifar10

COLOR_MEAN_RGB = np.array([125.3, 123.0, 113.9]).reshape(3, 1, 1)
COLOR_STD_RGB  = np.array([63.0,  62.1,  66.7]).reshape(3, 1, 1)

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
    xs = (xs - COLOR_MEAN_RGB) / COLOR_STD_RGB
    return xs, ys


def load_cifar10(train_batches=TRAIN_BATCHES, test_batch=TEST_BATCH):
    # the ordering is (samples, channels, height, width)
    train_xs = np.zeros((0, 3, 32, 32))
    train_ys = np.zeros((0, NUM_CLASSES), dtype=np.int32)
    for batch in train_batches:
        xs, ys = _load_batch(batch)
        train_xs = np.vstack((train_xs, xs))
        train_ys = np.vstack((train_ys, ys))
    test_xs, test_ys = _load_batch(test_batch)

    return (train_xs, train_ys), (test_xs, test_ys)


def augment_batch(batch, pad_dim=4):
    augmented_batch = np.zeros(batch.shape)
    num_imgs, _, height, width = batch.shape
    for i in range(num_imgs):
        img = batch[i]
        if np.random.normal(0, 1, (1,))[0] >= 0.5:
            img = img[:, :, ::-1]
        img = np.pad(img, ((0,), (pad_dim,), (pad_dim,)), 'constant', constant_values=(0,1))
        start_h, start_w = np.random.randint(0, 2*pad_dim, (2,))
        augmented_batch[i] = img[:, start_h:start_h+height, start_w:start_w+width]
    return augmented_batch


if __name__ == '__main__':
    (train_xs, train_ys), (test_xs, test_ys) = load_cifar10()
    print 'Loaded dataset:', train_xs.shape, train_ys.shape
