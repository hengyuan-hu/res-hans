import cPickle

import tensorflow as tf
import keras
from tensorflow.python.platform import app, flags

from cleverhans.utils_tf import tf_model_train, tf_model_eval, batch_eval
from cleverhans.attacks import fgsm
from misc import get_session, save_model, load_model
import resnet_cifar10
import somenet_cifar10
import data_cifar10


FLAGS = flags.FLAGS

flags.DEFINE_string('train_dir', './train', 'Directory storing the saved model.')
flags.DEFINE_string('filename', 'cifar10.ckpt', 'Filename to save model under.')
flags.DEFINE_integer('nb_epochs', 160, 'Number of epochs to train model')
flags.DEFINE_integer('batch_size', 128, 'Size of training batches')
flags.DEFINE_integer('nb_classes', 10, 'Number of classification classes')
flags.DEFINE_integer('img_rows', 32, 'Input row dimension')
flags.DEFINE_integer('img_cols', 32, 'Input column dimension')
flags.DEFINE_float('learning_rate', 0.1, 'Learning rate for training')


def adam_pretrain(model, model_name, train_xs, train_ys, num_epoch, test_xs, test_ys):
    model.compile(optimizer=keras.optimizers.Adam(),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_xs, train_ys, batch_size=128, nb_epoch=num_epoch,
              validation_data=(test_xs, test_ys), shuffle=True)
    model_name = '%s_adam_pretrain' % model_name
    save_model(model, model_name)
    model = load_model(model_name)
    return model


def main(net_type):
    if keras.backend.image_dim_ordering() != 'th':
        keras.backend.set_image_dim_ordering('th')
        print "INFO: temporarily set 'image_dim_ordering' to 'th'"

    sess = get_session()
    keras.backend.set_session(sess)

    (train_xs, train_ys), (test_xs, test_ys) = data_cifar10.load_cifar10()
    print 'Loaded cifar10 data'

    x = tf.placeholder(tf.float32, shape=(None, 3, 32, 32))
    y = tf.placeholder(tf.float32, shape=(None, 10))

    model, model_name = resnet_cifar10.resnet_cifar10(repetations=3, net_type=net_type)
    if net_type == 'squared_resnet':
        model = adam_pretrain(model, model_name, train_xs, train_ys, 1, test_xs, test_ys)

    predictions = model(x)
    tf_model_train(sess, x, y, predictions, train_xs, train_ys, test_xs, test_ys,
                   data_augmentor=data_cifar10.augment_batch)

    save_model(model, model_name)

    # Craft adversarial examples using Fast Gradient Sign Method (FGSM)
    adv_x = fgsm(x, predictions, eps=0.3)
    test_xs_adv, = batch_eval(sess, [x], [adv_x], [test_xs])
    assert test_xs_adv.shape[0] == 10000, test_xs_adv.shape

    # Evaluate the accuracy of the MNIST model on adversarial examples
    accuracy = tf_model_eval(sess, x, y, predictions, test_xs_adv, test_ys)
    print'Test accuracy on adversarial examples: ' + str(accuracy)

    print "Repeating the process, using adversarial training"
    # Redefine TF model graph
    model_2, _ = resnet_cifar10.resnet_cifar10(repetations=3, net_type=net_type)
    predictions_2 = model_2(x)
    adv_x_2 = fgsm(x, predictions_2, eps=0.3)
    predictions_2_adv = model_2(adv_x_2)

    # Perform adversarial training
    tf_model_train(sess, x, y, predictions_2, train_xs, train_ys, test_xs, test_ys,
                   predictions_adv=predictions_2_adv,
                   data_augmentor=data_cifar10.augment_batch)

    save_model(model, model_name+'_adv')

    # Craft adversarial examples using Fast Gradient Sign Method (FGSM) on
    # the new model, which was trained using adversarial training
    test_xs_adv_2, = batch_eval(sess, [x], [adv_x_2], [test_xs])
    assert test_xs_adv_2.shape[0] == 10000, test_xs_adv_2.shape

    # Evaluate the accuracy of the adversarially trained model on adversarial examples
    accuracy_adv = tf_model_eval(sess, x, y, predictions_2, test_xs_adv_2, test_ys)
    print'Test accuracy on adversarial examples: ' + str(accuracy_adv)


if __name__ == '__main__':
    net_type = 'squared_resnet'
    main(net_type)
