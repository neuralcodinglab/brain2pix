from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import numpy as np
from .common import delta_lookup, fit_func

npa = np.array

slim = tf.contrib.slim


def tf_quad_func(x, func_pars):
  return func_pars[0] * x ** 2 + func_pars[1] * x


def tf_exp_func(x, func_pars):
  return tf.exp(func_pars[0] * x) + func_pars[1]


def tf_image_translate(images, t, interpolation='NEAREST'):
  transforms = [1, 0, -t[0], 0, 1, -t[1], 0, 0]
  return tf.contrib.image.transform(tf.expand_dims(images, 0), transforms, interpolation)[0]


def tf_inv_quad_func(x, func_pars):
  a = func_pars[0]
  b = func_pars[1]
  return (-b + tf.sqrt(b ** 2 + 4*a*x))/(2*a)


def find_retina_mapping(input_size, output_size, fit_mode='quad'):
  """
  Fits a function to the distance data so it will map the outmost pixel to the border of the image
  :param fit_mode:
  :return:
  """
  r, r_raw = delta_lookup(in_size=input_size, out_size=output_size)
  if fit_mode == 'quad':
    func = lambda x, a, b: a * x ** 2 + b * x
    tf_func = tf_quad_func
  elif fit_mode == 'exp':
    func = lambda x, a, b: np.exp(a * x) + b
    tf_func = tf_exp_func
  else:
    raise ValueError('Fit mode not defined. Choices are ''linear'', ''exp''.')
  popt, pcov = fit_func(func, r, r_raw)

  return popt, tf_func


def warp_func(xy, orig_img_size, func, func_pars, shift):
  # Centeralize the indices [-n, n]
  xy = tf.cast(xy, tf.float32)
  center = tf.reduce_mean(xy, axis=0)
  xy_cent = xy - center

  # Polar coordinates
  r = tf.sqrt(xy_cent[:, 0] ** 2 + xy_cent[:, 1] ** 2)
  theta = tf.atan2(xy_cent[:, 1], xy_cent[:, 0])
  r = func(r, func_pars)

  xs = r * tf.cos(theta)
  xs += orig_img_size[0] / 2. - shift[0]
  # Added + 2.0 is for the additional zero padding
  xs = tf.minimum(orig_img_size[0] + 2.0, xs)
  xs = tf.maximum(0., xs)
  xs = tf.round(xs)

  ys = r * tf.sin(theta)
  ys += orig_img_size[1] / 2 - shift[1]
  ys = tf.minimum(orig_img_size[1] + 2.0, ys)
  ys = tf.maximum(0., ys)
  ys = tf.round(ys)

  xy_out = tf.stack([xs, ys], 1)

  xy_out = tf.cast(xy_out, tf.int32)
  return xy_out


def warp_image(img, output_size, input_size=None, shift=None):
  """

  :param img: (tensor) input image
  :param retina_func:
  :param retina_pars:
  :param shift:
  :return:
  """
  original_shape = img.shape

  if input_size is None:
    input_size = np.min([original_shape[0], original_shape[1]])

  retina_pars, retina_func = find_retina_mapping(input_size, output_size)

  if shift is None:
    shift = [tf.constant([0], tf.float32), tf.constant([0], tf.float32)]
  else:
    assert len(shift) == 2
    shift = [tf.constant([shift[0]], tf.float32), tf.constant([shift[1]], tf.float32)]
  paddings = tf.constant([[2, 2], [2, 2], [0, 0]])
  img = tf.pad(img, paddings, "CONSTANT")
  row_ind = tf.tile(tf.expand_dims(tf.range(output_size), axis=-1), [1, output_size])
  row_ind = tf.reshape(row_ind, [-1, 1])
  col_ind = tf.tile(tf.expand_dims(tf.range(output_size), axis=0), [1, output_size])
  col_ind = tf.reshape(col_ind, [-1, 1])
  indices = tf.concat([row_ind, col_ind], 1)
  xy_out = warp_func(indices, tf.cast(original_shape, tf.float32), retina_func, retina_pars, shift)

  out = tf.reshape(tf.gather_nd(img, xy_out), [output_size, output_size, 3])
  return out
