from __future__ import division

import numpy as np
import scipy
from scipy.sparse import linalg
import cv2
from numpy import r_

class CompressionMetada:
  def __init__(self, original_shape, grid_size):
    self.original_shape = original_shape
    self.grid_size = grid_size

def compress(image, grid_size=4, save_dct_path=None):
  """
  Compresses an image and returns (image, metdata) where `image` is the
  compressed image, and `metadata` is of type `CompressionMetada` and
  contains data neeeded for decompression.
  """
  dct_image = _dct_image(image, grid_size)
  dct_image = scipy.expand_dims(dct_image, axis=2)

  if save_dct_path != None:
    cv2.imwrite(save_dct_path, dct_image * 255)

  compressed = dct_image[grid_size//2:-1:grid_size, grid_size//2:-1:grid_size, 0]
  metadata = CompressionMetada(image.shape, grid_size)
  return compressed, metadata

def decompress(image, metadata, fidelity=10, tolerance=1e-5, iterations=1500, step=0.1, show_steps=False):
  """
  Decompresses a compressed image and returns it.
  """
  grid_size = metadata.grid_size
  w, h = metadata.original_shape
  decompressed = np.zeros((w, h, 1)) 
  decompressed[grid_size//2:-1:grid_size, grid_size//2:-1:grid_size, 0] = image
  mask = _make_mask(decompressed.shape, metadata.grid_size)
  return _harmonic(decompressed, mask, fidelity, tolerance, iterations, step, show_steps=True)


# Decompression helpers

def _harmonic(input, mask, fidelity, tolerance, maxiter, dt, show_steps=False):

  m, n = input.shape[:2]
  u = input.copy()

  for iter in range(0,maxiter):

    u_vals = u[:,:,0]

    laplacian = cv2.Laplacian(u_vals, cv2.CV_64F)
    u_new = u_vals + dt * (laplacian + fidelity * mask[:,:,0] * (input[:,:,0] - u_vals))

    u_new_flat = u_new.reshape(m*n,1)
    u_vals_flat = u_vals.reshape(m*n,1)

    diff_u = np.linalg.norm(u_new_flat - u_vals_flat, 2) / np.linalg.norm(u_new_flat, 2);

    u[:,:,0] = u_new

    if show_steps:
      if iter % 50 == 0:
        cv2.imwrite("images/iter" + str(iter) + ".png", u * 255)
      cv2.imshow("image", u)
      cv2.waitKey(10)

    if diff_u < tolerance:
      break
  
  return u

def _make_mask(image_size, grid_size):
  mask_size = image_size
  mask = np.zeros(mask_size)
  for i in r_[:mask_size[0]:grid_size]:
    for j in r_[:mask_size[1]:grid_size]:
      mask[i + grid_size // 2, j + grid_size // 2] = 1
  return mask


# Compression helpers

def _dct(a):
  return scipy.fftpack.dct(scipy.fftpack.dct( a, axis=0, norm='ortho'), axis=1, norm='ortho')

def _inverse_dct(a):
  return scipy.fftpack.idct(scipy.fftpack.idct(a, axis=0 , norm='ortho'), axis=1 , norm='ortho')

def _dct_image(image, grid_size):
  im = image.copy()
  imsize = im.shape
  dct = np.zeros(imsize)

  # Do 8x8 DCT on image (in-place)
  for i in r_[:imsize[0]:grid_size]:
      for j in r_[:imsize[1]:grid_size]:
          dct[i:(i+grid_size),j:(j+grid_size)] = _dct( im[i:(i+grid_size),j:(j+grid_size)] )

  thresh = 0.012
  dct_thresh = dct * (abs(dct) > (thresh*np.max(dct)))

  im_dct = np.zeros(imsize)

  for i in r_[:imsize[0]:grid_size]:
    for j in r_[:imsize[1]:grid_size]:
      im_dct[i + grid_size // 2, j + grid_size // 2] = _inverse_dct(dct_thresh[i:(i+grid_size),j:(j+grid_size)])[0, 0]

  return im_dct
