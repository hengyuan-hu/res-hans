import cPickle

import tensorflow as tf
import keras

from tensorflow.python.platform import app, flags
import resnet_cifar10
from keras.preprocessing.image import ImageDataGenerator
import somenet_cifar10
import data_cifar10
from cleverhans.utils_tf import tf_model_train, tf_model_eval, batch_eval

# FLAGS = flags.FLAGS

# flags.DEFINE_string('train_dir', './train', 'Directory storing the saved model.')
# flags.DEFINE_string('filename', 'cifar10.ckpt', 'Filename to save model under.')
# flags.DEFINE_integer('nb_epochs', 10, 'Number of epochs to train model')
# flags.DEFINE_integer('batch_size', 128, 'Size of training batches')
# flags.DEFINE_integer('nb_classes', 10, 'Number of classification classes')
# flags.DEFINE_integer('img_rows', 32, 'Input row dimension')
# flags.DEFINE_integer('img_cols', 32, 'Input column dimension')
# flags.DEFINE_float('learning_rate', 0.01, 'Learning rate for training')


def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)


def _get_model_files(model_name):
    model_file = '%s.json' % model_name 
    weight_file = '%s.h5' % model_name
    history_file = '%s_history.pkl' % model_name
    return model_file, weight_file, history_file


def save_model(model, model_name, history):
    model_file, weight_file, history_file = _get_model_files(model_name)
    model_json = model.to_json()
    with open(model_file, 'w') as json_file:
        json_file.write(model_json)
    model.save_weights(weight_file)
    cPickle.dump(history, open(history_file, 'wb'))
    print('Saved model to disk')


def load_model(model_name): #model_file, weight_file):
    model_file, weight_file, history_file = _get_model_files(model_name)
    with open(model_file, 'r') as json_file:
        loaded_model_json = json_file.read()

    loaded_model = keras.models.model_from_json(loaded_model_json)
    loaded_model.load_weights(weight_file)
    history = cPickle.load(open(history_file, 'rb'))
    print('Loaded model from disk')
    return loaded_model, history


def main():
    if keras.backend.image_dim_ordering() != 'th':
        keras.backend.set_image_dim_ordering('th')
        print "INFO: temporarily set 'image_dim_ordering' to 'th'"

    sess = get_session()
    keras.backend.set_session(sess)

    train_xs, train_ys, test_xs, test_ys = data_cifar10.load_cifar10()
    print 'Loaded cifar10 data'

    datagen = ImageDataGenerator(
        featurewise_center=False,            # set input mean to 0 over the dataset
        samplewise_center=False,             # set each sample mean to 0
        featurewise_std_normalization=False, # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,                 # apply ZCA whitening
        rotation_range=0,                    # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,               # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,              # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,                # randomly flip images
        vertical_flip=False                  # randomly flip images
    )
    datagen.fit(train_xs)

    # x = tf.placeholder(tf.float32, shape=(None, 3, 32, 32))
    # y = tf.placeholder(tf.float32, shape=(None, 32))

    model, model_name = resnet_cifar10.resnet_cifar10(repetations=3)
    # model, model_name = somenet_cifar10.model()
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    print '-----------'
    print 'train set shape :', train_xs.shape, train_ys.shape
    test = False
    if test:
        history = model.fit(train_xs[:128], train_ys[:128], batch_size=128, nb_epoch=1,
                            validation_data=(test_xs[:128], test_ys[:128]))
    else:
        history = model.fit_generator(datagen.flow(train_xs, train_ys, batch_size=128),
                                      samples_per_epoch=train_xs.shape[0],
                                      nb_epoch=50, validation_data=(test_xs, test_ys))

    save_model(model, model_name, history.history)

    return history


if __name__ == '__main__':
    history = main()
