import matplotlib.pyplot as plt
import pydicom

def read_dcm_image(path):
  dicom = pydicom.dcmread(path)
  data = dicom.pixel_array
  return data

def show_dcm_image(image):
  plt.imshow(image, cmap=plt.cm.bone)
  plt.waitforbuttonpress()
