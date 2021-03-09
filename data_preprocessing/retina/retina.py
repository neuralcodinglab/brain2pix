import numpy as np
from .common import delta_lookup, fit_func
from skimage.util import crop, pad
from skimage.transform import resize, warp

npa = np.array


def get_fit_func(fit_mode='quad'):
  """
  Gets the fitting function matching the fit_mode ['quad', 'linear', 'exp']
  :param fit_mode: choice of the fitting function ['quad', 'linear', 'exp']
  :return: fitting function
  """
  if fit_mode == 'quad':
    func = lambda x, a, b: a * x ** 2 + b * x
  elif fit_mode == 'linear':
    func = lambda x, a, b: a * x + b
  elif fit_mode == 'exp':
    func = lambda x, a, b: np.exp(a * x) + b
  else:
    raise ValueError('Fit mode not defined. Choices are ''linear'', ''exp''.')
  return func


def warp_func(xy, func, a, b):
  """
  Warps the xy points given the function func and parameter a and b.
  :param xy: Input points [2 x num_points]
  :param func: Warping function
  :param a: func parameter 1
  :param b: func parameter 2
  :return: Transformed points on the original image
  """
  center = np.mean(xy, axis=0)
  xc, yc = (xy - center).T

  # Polar coordinates
  r = np.sqrt(xc ** 2 + yc ** 2)
  theta = np.arctan2(yc, xc)

  r = func(r, a, b)
  out = np.column_stack((
    r * np.cos(theta), r * np.sin(theta)
  ))
  out = out + center
  return out


def warp_image(image, output_size=None, input_size=None, fill_value=0.):
  """
  transforms the input image using a fisheye transformation
  :param image: (ndarray) input image
  :param output_size: (int) size of the output image
  :param fill_value: (float) value to fill in the missing values with
  :return: (ndarray) transformed image
  """
  original_shape = npa(image.shape[:2])

  if input_size is None:
    input_size = np.min([original_shape[0], original_shape[1]])

  if output_size is None:
    output_size = np.min(image.shape[0:2])


  if any(original_shape < output_size):
    scale_ratio = np.max(float(output_size) / original_shape)
    warped_img = resize(image, npa(scale_ratio*original_shape, dtype=int))
    input_size = npa(input_size * scale_ratio, dtype=int)
  else:
    warped_img = image

  shape_diff = output_size - npa(warped_img.shape[:2])


  r, r_raw = delta_lookup(in_size=input_size, out_size=output_size)
  func = get_fit_func()
  popt, pcov = fit_func(func, r, r_raw)
  warped_img = warp(warped_img, warp_func, map_args={'func': func, 'a': popt[0], 'b': popt[1]}, cval=fill_value)

  if any(shape_diff < 0):
    crop_width = -shape_diff * (npa(shape_diff) < 0).astype(int)
    warped_img = crop(warped_img, crop_width=((crop_width[0]//2,)*2, (crop_width[1]//2,)*2, (0, 0)))

  # if any(shape_diff > 0):
  #   pad_width = shape_diff * (npa(shape_diff) > 0).astype(int)
  #   warped_img = pad(warped_img, pad_width=((pad_width[0]//2,)*2, (pad_width[1]//2,)*2, (0, 0)), mode='constant')

  return warped_img
