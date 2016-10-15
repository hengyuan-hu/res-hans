import tensorflow as tf 
from tensorflow.python.platform import app, flags

FLAGS = flags.FLAGS


def inference(images):
  """Build the CIFAR-10 model.
  Args:
    images: Images returned from distorted_inputs() or inputs().
  Returns:
    Logits.
  """
  # We instantiate all variables using tf.get_variable() instead of
  # tf.Variable() in order to share variables across multiple GPU training runs.
  # If we only ran this model on a single GPU, we could simplify this function
  # by replacing all instances of tf.get_variable() with tf.Variable().
  #
  # conv1
  with tf.variable_scope('conv1') as scope:
    kernel = tf.get_variable('weights',
                             [5, 5, 3, 64],
                             initializer=tf.truncated_normal_initializer(stddev=5e-2),
                             dtype=tf.float32)
    conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.get_variable('biases', [64], 
                             initializer=tf.constant_initializer(0.0), dtype=tf.float32)
    bias = tf.nn.bias_add(conv, biases)
    conv1 = tf.nn.relu(bias, name=scope.name)
    # _activation_summary(conv1)

  # pool1
  pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                         padding='SAME', name='pool1')
  # norm1
  norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                    name='norm1')

  # conv2
  with tf.variable_scope('conv2') as scope:
    kernel = tf.get_variable('weights',
                             [5, 5, 64, 64],
                             initializer=tf.truncated_normal_initializer(stddev=5e-2),
                             dtype=tf.float32)
    conv = tf.nn.conv2d(norm1, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.get_variable('biases', [64], 
                             initializer=tf.constant_initializer(0.0), dtype=tf.float32)
    bias = tf.nn.bias_add(conv, biases)
    conv2 = tf.nn.relu(bias, name=scope.name)
    # _activation_summary(conv2)

  # # norm2
  # norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
  #                   name='norm2')
  # pool2
  pool2 = tf.nn.max_pool(conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                         padding='SAME', name='pool2')

  flatten = tf.contrib.layers.flatten(pool2)

  # local3
  with tf.variable_scope('local3') as scope:
    # Move everything into depth so we can perform a single matrix multiply.
    # reshape = tf.reshape(pool2, [FLAGS.batch_size, -1])
    dim = flatten.get_shape()[1].value
    weights = tf.get_variable('weights', shape=[dim, 384],
                              initializer=tf.truncated_normal_initializer(stddev=5e-2),
                              dtype=tf.float32)
    biases = tf.get_variable('biases', [384], 
                             initializer=tf.constant_initializer(0.0), dtype=tf.float32)
    local3 = tf.nn.relu(tf.matmul(flatten, weights) + biases, name=scope.name)
    # _activation_summary(local3)

  # local4
  with tf.variable_scope('local4') as scope:
    weights = tf.get_variable('weights', shape=[384, 192],
                             initializer=tf.truncated_normal_initializer(stddev=5e-2),
                             dtype=tf.float32)
    biases = tf.get_variable('biases', [192],
                             initializer=tf.constant_initializer(0.0), dtype=tf.float32)
    local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name=scope.name)
    # _activation_summary(local4)

  # softmax, i.e. softmax(WX + b)
  with tf.variable_scope('softmax_linear') as scope:
    weights = tf.get_variable('weights', [192, FLAGS.nb_classes],
                              initializer=tf.truncated_normal_initializer(stddev=5e-2),
                              dtype=tf.float32)
    biases = tf.get_variable('biases', [FLAGS.nb_classes],
                             initializer=tf.constant_initializer(0.0), dtype=tf.float32)
    softmax_linear = tf.add(tf.matmul(local4, weights), biases, name=scope.name)
    # _activation_summary(softmax_linear)

  return softmax_linear
