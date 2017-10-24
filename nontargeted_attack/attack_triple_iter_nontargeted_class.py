"""
LGPL v3 License

More information from the user maplewolf on Kaggle.Implementation of sample attack."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import os

import numpy as np
from scipy.misc import imread, imsave

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
    'output_dir', '', 'Output directory with images.')

tf.flags.DEFINE_float(
    'max_epsilon', 16.0, 'Maximum size of adversarial perturbation.')

tf.flags.DEFINE_float(
    'iter_alpha', 1.0, 'Step size for one iteration.')

tf.flags.DEFINE_integer(
    'num_iter', 20, 'Number of iterations.')

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
    """
    images = np.zeros(batch_shape)
    filenames = []
    idx = 0
    batch_size = batch_shape[0]
    for filepath in tf.gfile.Glob(os.path.join(input_dir, '*.png')):
        with tf.gfile.Open(filepath, 'rb') as f:
            image = imread(f, mode='RGB').astype(np.float) / 255.0
        # Images for inception classifier are normalized to be in [-1, 1] interval.
        images[idx, :, :, :] = image * 2.0 - 1.0
        filenames.append(os.path.basename(filepath))
        idx += 1
        if idx == batch_size:
            yield filenames, images
            filenames = []
            images = np.zeros(batch_shape)
            idx = 0
    if idx > 0:
        yield filenames, images


def save_images(images, filenames, output_dir):
    """Saves images to the output directory.

    Args:
      images: array with minibatch of images
      filenames: list of filenames without path
        If number of file names in this list less than number of images in
        the minibatch then only first len(filenames) images will be saved.
      output_dir: directory where to save images
    """
    for i, filename in enumerate(filenames):
        # Images for inception classifier are normalized to be in [-1, 1] interval,
        # so rescale them back to [0, 1].
        with tf.gfile.Open(os.path.join(output_dir, filename), 'wb') as f:
            imsave(f, np.round(255.0 * (np.float32(images[i, :, :, :]) + 1.0) * 0.5).astype(np.int16), format='png')


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

    # Images for inception classifier are normalized to be in [-1, 1] interval,
    # eps is a difference between pixels so it should be in [0, 2] interval.
    # Renormalizing epsilon from [0, 255] to [0, 2].
    eps = 2.0 * FLAGS.max_epsilon / 255.0
    alpha = 2.0 * FLAGS.iter_alpha / 255.0
    num_iter = FLAGS.num_iter
    batch_shape = [FLAGS.batch_size, FLAGS.image_height, FLAGS.image_width, 3]
    num_classes = 1001

    tf.logging.set_verbosity(tf.logging.INFO)

    label_smoothing = 0.1
    coeff_mul_alpha = 0.019
    AUX_ENS_V2 = 2.4
    AUX_INC_V3 = 0.87
    aux_weight = AUX_INC_V3
    model_mode = 0

    with tf.Graph().as_default():
        # Prepare graph
        x_input = tf.placeholder(tf.float32, shape=batch_shape)
        x_max = tf.clip_by_value(x_input + eps, -1.0, 1.0)
        x_min = tf.clip_by_value(x_input - eps, -1.0, 1.0)

        with tf.variable_scope('model_a'):
            with slim.arg_scope(inception.inception_v3_arg_scope()):
                inception.inception_v3(
                    x_input, num_classes=num_classes, is_training=False)
        with tf.variable_scope('model_b'):
            with slim.arg_scope(inception.inception_v3_arg_scope()):
                inception.inception_v3(
                    x_input, num_classes=num_classes, is_training=False)
        with tf.variable_scope('model_c'):
            with slim.arg_scope(inception_resnet_v2.inception_resnet_v2_arg_scope()):
                original_logits, _ = inception_resnet_v2.inception_resnet_v2(
                    x_input, num_classes=num_classes, is_training=False)

        x_adv = x_input

        def wrap_guess_target_class(logits, arr_kth_largest):
            arr_logits = np.array(logits)
            kth_largest = int(arr_kth_largest)
            return np.array([arr.argsort()[- kth_largest] for arr in arr_logits], dtype = np.int32)
        target_class_input = tf.py_func(wrap_guess_target_class, [original_logits, np.array(5)], tf.int32)

        one_hot_target_class = tf.one_hot(target_class_input, num_classes)

        for i_iter in range(num_iter):
            model_mode = i_iter % 4
            if i_iter >= 16:
                model_mode = 3

            if i_iter == 0:
                label_smoothing = 0.1
                coeff_mul_alpha = 0.019
            elif i_iter == 10:
                label_smoothing = 0
                coeff_mul_alpha = 0.031

            if model_mode == 1:
                with tf.variable_scope('model_a'):
                    with slim.arg_scope(inception.inception_v3_arg_scope()):
                        logits, end_points = inception.inception_v3(
                            x_adv, num_classes=num_classes, is_training=False, reuse=True)
            elif model_mode == 0:
                with tf.variable_scope('model_b'):
                    with slim.arg_scope(inception.inception_v3_arg_scope()):
                        logits, end_points = inception.inception_v3(
                            x_adv, num_classes=num_classes, is_training=False, reuse=True)
            elif model_mode == 2:
                with tf.variable_scope('model_a'):
                    with slim.arg_scope(inception.inception_v3_arg_scope()):
                        logits, end_points = inception.inception_v3(
                            x_adv, num_classes=num_classes, is_training=False, reuse=True)
            else:
                with tf.variable_scope('model_c'):
                    with slim.arg_scope(inception_resnet_v2.inception_resnet_v2_arg_scope()):
                        logits, end_points = inception_resnet_v2.inception_resnet_v2(
                            x_adv, num_classes=num_classes, is_training=False, reuse=True)

            if model_mode == 3:
                aux_weight = AUX_ENS_V2
            else:
                aux_weight = AUX_INC_V3

            cross_entropy = tf.losses.softmax_cross_entropy(one_hot_target_class,
                                                            logits,
                                                            label_smoothing=label_smoothing,
                                                            weights=1.0)
            cross_entropy += tf.losses.softmax_cross_entropy(one_hot_target_class,
                                                             end_points['AuxLogits'],
                                                             label_smoothing=label_smoothing,
                                                             weights = aux_weight)
            compute_gradient = tf.gradients(cross_entropy, x_adv)[0]

            if model_mode == 2:
                with tf.variable_scope('model_b'):
                    with slim.arg_scope(inception.inception_v3_arg_scope()):
                        logits_2, end_points_2 = inception.inception_v3(
                            x_adv, num_classes=num_classes, is_training=False, reuse=True)
                cross_entropy_2 = tf.losses.softmax_cross_entropy(one_hot_target_class,
                                                                logits_2,
                                                                label_smoothing=label_smoothing,
                                                                weights=1.0)
                cross_entropy_2 += tf.losses.softmax_cross_entropy(one_hot_target_class,
                                                                 end_points_2['AuxLogits'],
                                                                 label_smoothing=label_smoothing,
                                                                 weights = aux_weight)
                compute_gradient_2 = tf.gradients(cross_entropy_2, x_adv)[0]

                equal_gradient_sign = tf.cast(tf.equal(tf.sign(compute_gradient), tf.sign(compute_gradient_2)), tf.float32)
                compute_gradient = tf.multiply(tf.add(compute_gradient, compute_gradient_2), equal_gradient_sign)

            gradient_clip = tf.clip_by_value(compute_gradient, -0.0001, 0.0001)
            multiplier = 1.0
            if model_mode == 3:
                if i_iter < 13:
                    multiplier = (1 - 0.0025          * i_iter) * 2.5 * alpha
                else:
                    multiplier = (1 - coeff_mul_alpha * i_iter) * 2.5 * alpha
            elif model_mode == 2:
                multiplier = (1 - coeff_mul_alpha * i_iter) * 3.6 * alpha
            else:
                multiplier = (1 - coeff_mul_alpha * i_iter) * 1.2 * alpha

            x_next = x_adv - multiplier * tf.add(0.5 * tf.sign(compute_gradient), 0.5 * gradient_clip * 10000)

            x_next = tf.clip_by_value(x_next, x_min, x_max)
            x_adv = x_next

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            all_vars = tf.global_variables()
            model_a_vars = [k for k in all_vars if k.name.startswith('model_a')] # FLAGS.model_a_scope
            model_b_vars = [k for k in all_vars if k.name.startswith('model_b')] # FLAGS.model_b_scope
            model_c_vars = [k for k in all_vars if k.name.startswith('model_c')] # FLAGS.model_c_scope

            model_a_checkpoint = 'inception_v3.ckpt'
            model_b_checkpoint = 'adv_inception_v3.ckpt'
            model_c_checkpoint = 'ens_adv_inception_resnet_v2.ckpt'

            tf.train.Saver(convert_dict(model_a_vars, model_a_checkpoint)).restore(sess, model_a_checkpoint)
            tf.train.Saver(convert_dict(model_b_vars, model_b_checkpoint)).restore(sess, model_b_checkpoint)
            tf.train.Saver(convert_dict(model_c_vars, model_c_checkpoint)).restore(sess, model_c_checkpoint)

            for filenames, images in load_images(FLAGS.input_dir, batch_shape):
                adv_images = sess.run(x_adv, feed_dict={x_input: images})
                save_images(adv_images, filenames, FLAGS.output_dir)


if __name__ == '__main__':
    tf.app.run()
