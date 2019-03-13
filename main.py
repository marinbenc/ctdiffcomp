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

# Get image

image_clean = image.read_dcm_image(IMAGE_PATH)
#normalize CT image for viewing
image_clean = (image_clean / np.max(image_clean))
cv2.imwrite(CLEAN_IMAGE_PATH, image_clean * 255)


# Compress

input_image, metadata = cmp.compress(image_clean, save_dct_path=DCT_IMAGE_PATH)
cv2.imwrite(INPUT_IMAGE_PATH, input_image * 255)
clean_pixels = np.count_nonzero(image_clean)
input_pixels = np.count_nonzero(input_image)
print("Compressed percent of pixels: " + str(input_pixels / clean_pixels))


# Decompress

decompressed = cmp.decompress(input_image, metadata, show_steps=True)
cv2.imwrite(RESULT_PATH, decompressed * 255)