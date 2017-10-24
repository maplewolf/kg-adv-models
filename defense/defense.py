"""
LGPL v3 License

More information from the user maplewolf on Kaggle.
 Implementation of sample defense.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
from scipy.misc import imread
from scipy.ndimage import median_filter

import tensorflow as tf

from tensorflow.contrib.slim.nets import inception
import inception_resnet_v2


slim = tf.contrib.slim

tf.flags.DEFINE_string(
    'master', '', 'The address of the TensorFlow master to use.')

tf.flags.DEFINE_string(
    'checkpoint_path', '', 'Path to checkpoint for inception network.')

tf.flags.DEFINE_string(
    'input_dir', '', 'Input directory with images.')

tf.flags.DEFINE_string(
    'output_file', '', 'Output file to save labels.')

tf.flags.DEFINE_integer(
    'image_width', 299, 'Width of each input images.')

tf.flags.DEFINE_integer(
    'image_height', 299, 'Height of each input images.')

tf.flags.DEFINE_integer(
    'batch_size', 16, 'How many images process at one time.')

FLAGS = tf.flags.FLAGS


def load_images(input_dir, batch_shape):
    """Read png images from input directory in batches.

    Args:
      input_dir: input directory
      batch_shape: shape of minibatch array, i.e. [batch_size, height, width, 3]

    Yields:
      filenames: list file names without path of each image
        Lenght of this list could be less than batch_size, in this case only
        first few images of the result are elements of the minibatch.
      images: array with all images from this batch
      images_smooth: array with all smooth images from this batch
    """
    images = np.zeros(batch_shape)
    images_smooth = np.zeros(batch_shape)
    filenames = []
    idx = 0
    batch_size = batch_shape[0]
    for filepath in tf.gfile.Glob(os.path.join(input_dir, '*.png')):
        with tf.gfile.Open(filepath, 'rb') as f:
            image = imread(f, mode='RGB').astype(np.float)
            image_smooth = median_filter(image, size=3)
            image = image / 255.0
            image_smooth = image_smooth / 255.0
        # Images for inception classifier are normalized to be in [-1, 1] interval.
        images[idx, :, :, :] = image * 2.0 - 1.0
        images_smooth[idx, :, :, :] = image_smooth * 2.0 - 1.0
        filenames.append(os.path.basename(filepath))
        idx += 1
        if idx == batch_size:
            yield filenames, images, images_smooth
            filenames = []
            images = np.zeros(batch_shape)
            images_smooth = np.zeros(batch_shape)
            idx = 0
    if idx > 0:
        yield filenames, images, images_smooth


def convert_dict(model_vars, checkpoint_path):
    from tensorflow.python import pywrap_tensorflow

    reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
    var_to_shape_map = reader.get_variable_to_shape_map()
    ckpt_name_set = set([key for key in var_to_shape_map])

    dict4vars = {}
    for var in model_vars:
        if var.name[8:-2] in ckpt_name_set:
            dict4vars[var.name[8:-2]] = var
    return dict4vars


def main(_):

    batch_shape = [FLAGS.batch_size, FLAGS.image_height, FLAGS.image_width, 3]
    num_classes = 1001

    tf.logging.set_verbosity(tf.logging.INFO)

    with tf.Graph().as_default():
        # Prepare graph
        x_input = tf.placeholder(tf.float32, shape=batch_shape)
        x_input_smooth = tf.placeholder(tf.float32, shape=batch_shape)

        with tf.variable_scope('model_a'):
            with slim.arg_scope(inception.inception_v3_arg_scope()):
                _, end_points_smooth_a = inception.inception_v3(
                    x_input_smooth, num_classes=num_classes, is_training=False)
        with tf.variable_scope('model_b'):
            with slim.arg_scope(inception.inception_v3_arg_scope()):
                _, end_points_smooth_b = inception.inception_v3(
                    x_input_smooth, num_classes=num_classes, is_training=False)
        with tf.variable_scope('model_c'):
            with slim.arg_scope(inception_resnet_v2.inception_resnet_v2_arg_scope()):
                _, end_points_smooth_c = inception_resnet_v2.inception_resnet_v2(
                    x_input_smooth, num_classes=num_classes, is_training=False)
            with slim.arg_scope(inception_resnet_v2.inception_resnet_v2_arg_scope()):
                _, end_points_c = inception_resnet_v2.inception_resnet_v2(
                    x_input, num_classes=num_classes, is_training=False, reuse=True)

        def wrap_predict_labels(preds_smooth_a, preds_smooth_b, preds_smooth_c, preds_c, preds_a = None, preds_b = None):
            arr_smooth_a = np.array(preds_smooth_a)
            arr_smooth_b = np.array(preds_smooth_b)
            arr_smooth_c = np.array(preds_smooth_c)
            arr_c = np.array(preds_c)
            argmax_smooth_c = np.argmax(arr_smooth_c, axis = 1)
            argmax_c = np.argmax(arr_c, axis = 1)

            for i in range(len(argmax_c)):
                if argmax_c[i] != argmax_smooth_c[i]:
                    argmax_smooth_c[i] = np.argmax(np.add(np.add(arr_smooth_a[i], arr_smooth_b[i]), 2 * arr_smooth_c[i]))
            return argmax_smooth_c.astype(np.int32)

        predicted_labels = tf.py_func(wrap_predict_labels, [end_points_smooth_a['Predictions'], end_points_smooth_b['Predictions'], \
                                                            end_points_smooth_c['Predictions'], end_points_c['Predictions']], tf.int32)

        with tf.Session() as sess:

            sess.run(tf.global_variables_initializer())

            all_vars = tf.global_variables()
            model_a_vars = [k for k in all_vars if k.name.startswith('model_a')]  # FLAGS.model_a_scope
            model_b_vars = [k for k in all_vars if k.name.startswith('model_b')]  # FLAGS.model_b_scope
            model_c_vars = [k for k in all_vars if k.name.startswith('model_c')]  # FLAGS.model_c_scope

            model_a_checkpoint = 'inception_v3.ckpt'
            model_b_checkpoint = 'adv_inception_v3.ckpt'
            model_c_checkpoint = 'ens_adv_inception_resnet_v2.ckpt'

            tf.train.Saver(convert_dict(model_a_vars, model_a_checkpoint)).restore(sess, model_a_checkpoint)
            tf.train.Saver(convert_dict(model_b_vars, model_b_checkpoint)).restore(sess, model_b_checkpoint)
            tf.train.Saver(convert_dict(model_c_vars, model_c_checkpoint)).restore(sess, model_c_checkpoint)

            with tf.gfile.Open(FLAGS.output_file, 'wb') as out_file:
                for filenames, images, images_smooth in load_images(FLAGS.input_dir, batch_shape):
                    labels = sess.run(predicted_labels, feed_dict={x_input: images, x_input_smooth: images_smooth})
                    for filename, label in zip(filenames, labels):
                        out_file.write('{0},{1}\n'.format(filename, label))


if __name__ == '__main__':
    tf.app.run()
