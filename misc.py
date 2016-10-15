import tensorflow as tf
import keras

def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)


def _get_model_files(model_name):
    model_file = '%s.json' % model_name 
    weight_file = '%s.h5' % model_name
    return model_file, weight_file


def save_model(model, model_name):
    model_file, weight_file = _get_model_files(model_name)
    model_json = model.to_json()
    with open(model_file, 'w') as json_file:
        json_file.write(model_json)
    model.save_weights(weight_file)
    print('Saved model to disk')


def load_model(model_name): #model_file, weight_file):
    model_file, weight_file = _get_model_files(model_name)
    with open(model_file, 'r') as json_file:
        loaded_model_json = json_file.read()

    loaded_model = keras.models.model_from_json(loaded_model_json)
    loaded_model.load_weights(weight_file)
    print('Loaded model from disk')
    return loaded_model
