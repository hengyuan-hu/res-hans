import keras

import resnet_cifar10
import somenet_cifar10
import data_cifar10


if __name__ == '__main__':
    (train_xs, train_ys), (test_xs, test_ys) = data_cifar10.load_cifar10()

    net_type = 'resnet'

    model, model_name = resnet_cifar10.resnet_cifar10(repetations=3, net_type=net_type)
    optimizer = keras.optimizers.SGD(lr=0.1, momentum=0.9, decay=0.0, nesterov=True)
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy', metrics=['accuracy'])
    # model.compile(optimizer=keras.optimizers.Adam(),
    #               loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_xs, train_ys, batch_size=128, nb_epoch=10, 
              validation_data=(test_xs, test_ys), shuffle=True)
