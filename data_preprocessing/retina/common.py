import numpy as np
from scipy.optimize import curve_fit, brenth
from functools import partial


def sampling_mismatch(rf, in_size=None, out_size=None, max_ratio=10.):
  """
  This function returns the mismatch between the radius of last sampled point and the image size.
  """
  if out_size is None:
    out_size = in_size
  r_max = in_size // 2

  # Exponential relationship
  a = np.log(max_ratio) / r_max
  r, d = [0.], []
  for i in range(1, out_size // 2):
    d.append(1. / np.sqrt(np.pi * rf) * np.exp(a * r[-1] / 2.))
    r.append(r[-1] + d[-1])
  r = np.array(r)

  return in_size / 2 - r[-1]


def get_rf_value(input_size, output_size, rf_range=(0.01, 5.)):
  """
  The RF parameter should be tuned in a way that the last sample would be taken from the outmost pixel of the image.
  This function returns the mismatch between the radius of last sampled point and the image size. We use this function
  together with classic root finding methods to find the optimal RF value given the input and output sizes.
  """
  func = partial(sampling_mismatch, in_size=input_size, out_size=output_size)
  return brenth(func, rf_range[0], rf_range[1])


def get_foveal_density(output_image_size, input_image_size):
    return get_rf_value(input_image_size, output_image_size)


def delta_lookup(in_size, out_size=None, max_ratio=10.):
  """
  Divides the range of radius values based on the image size and finds the distances between samples
  with respect to each radius value. Different function types can be used to form the mapping. All function
  map to delta values of min_delta in the center and max_delta at the outmost periphery.
  :param in_size: Size of the input image
  :param out_size: Size of the output (retina) image
  :param max_ratio: ratio between density at the fovea and periphery
  :return: Grid of points on the retinal image (r_prime) and original image (r)
  """
  rf = get_foveal_density(out_size, in_size)
  if out_size is None:
    out_size = in_size
  r_max = in_size // 2

  # Exponential relationship
  a = np.log(max_ratio) / r_max
  r, d = [0.], []
  for i in range(out_size // 2):
    d.append(1. / np.sqrt(np.pi * rf) * np.exp(a * r[-1] / 2.))
    r.append(r[-1] + d[-1])
  r = np.array(r)
  r_prime = np.arange(out_size // 2)

  return r_prime, r[:-1]


def fit_func(func, r, r_raw):
  """
  Fits a function to map the radius values in the
  :param func: function template
  :param r: Inputs to the function (grid points on the retinal image)
  :param r_raw: Outputs for the function (grid points on the original image)
  :return: Estimated parameters, estimaged covariance of parameters
  """
  popt, pcov = curve_fit(func, r, r_raw, p0=[0, 0.4], bounds=(0, np.inf))
  return popt, pcov

