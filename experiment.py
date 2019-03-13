import cv2
import compression as cmp
import image
import matplotlib.image as mpimg
import numpy as np
import scipy.fftpack
import matplotlib.pyplot as plt
from numpy import r_

# Constants

REMOVE_PIXELS_PERCENT = 0.9
IMAGE_PATH = "images/dcm-image.dcm"
CLEAN_IMAGE_PATH = "images/clean-image.png"
RESULT_PATH = "images/result.png"
INPUT_IMAGE_PATH = "images/input.png"
DCT_IMAGE_PATH = "images/dct.png"
COMPRESSED_IMAGE_PATH = "images/compressed.png"


for i in range(1, 10):

  # Get image

  path = "experiment/input/" + str(i) + ".dcm"
  print(path)
  image_clean = image.read_dcm_image(path)
  #normalize CT image for viewing
  image_clean = (image_clean / np.max(image_clean))
  cv2.imwrite("experiment/output/" + str(i) + "_original.png", image_clean * 255)


  # Compress

  input_image, metadata = cmp.compress(image_clean)
  cv2.imwrite("experiment/output/" + str(i) + "_compressed.png", input_image * 255)
  clean_pixels = np.count_nonzero(image_clean)
  input_pixels = np.count_nonzero(input_image)
  print("Compressed percent of pixels: " + str(input_pixels / clean_pixels))


  # Decompress

  decompressed = cmp.decompress(input_image, metadata)
  cv2.imwrite("experiment/output/" + str(i) + "_result.png", decompressed * 255)