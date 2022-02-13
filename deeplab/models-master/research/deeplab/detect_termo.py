import os
from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
import tensorflow.compat.v1 as tf #tensorflow_version 1.x
import time
import sys

#PARAMS
WIDTH_SIZE = 640
HEIGHT_SIZE = 480
EXP_NAME= sys.argv[1] #"E001-06"
MODEL_NAME = EXP_NAME + ".pb"

MODELS_PATH = sys.argv[2] #"/home/oscar/Desktop/DeepLabV3+PNOA-TF-Docker/DeepLabGPU-TF-EMIDLEVELS-docker/models/"
IMAGES_FOLDER = sys.argv[3] #"/home/oscar/Desktop/DeepLabV3+PNOA-TF-Docker/DeepLabGPU-TF-EMIDLEVELS-docker/images/"
DETECTIONS_FOLDER = sys.argv[4] #"/media/oscar/New Volume/DATASETS/EMID/20210526/AVI/All/expLevels/vis/" + EXP_NAME + "/detections2/"

# If destination folder does not exist, create it.
if not os.path.isdir(DETECTIONS_FOLDER):
    os.mkdir(DETECTIONS_FOLDER)

class DeepLabModel(object):
  """Class to load deeplab model and run inference."""

  INPUT_TENSOR_NAME = 'ImageTensor:0'
  OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
  #INPUT_SIZE = 512


  def __init__(self):
    """Creates and loads pretrained deeplab model."""
    self.graph = tf.Graph()
    graph_def = None
    with tf.io.gfile.GFile(os.path.join(MODELS_PATH, MODEL_NAME), 'rb') as f:
        graph_def = tf.GraphDef.FromString(f.read())
        
    if graph_def is None:
      raise RuntimeError('Cannot find inference graph.')

    with self.graph.as_default():
      tf.import_graph_def(graph_def, name='')

    self.sess = tf.Session(graph=self.graph)

  def run(self, image):
    """Runs inference on a single image.

    Args:
      image: A PIL.Image object, raw input image.

    Returns:
      resized_image: RGB image resized from original input image.
      seg_map: Segmentation map of `resized_image`.
    """
    #width, height = image.size
    #resize_ratio = 1.0 * self.INPUT_SIZE / max(width, height)
    width=WIDTH_SIZE
    height=HEIGHT_SIZE
    resize_ratio = 1.0 
    target_size = (int(resize_ratio * width), int(resize_ratio * height))
    resized_image = image.convert('RGB').resize(target_size, Image.ANTIALIAS)
    batch_seg_map = self.sess.run(
        self.OUTPUT_TENSOR_NAME,
        feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(resized_image)]})
    seg_map = batch_seg_map[0]
    return resized_image, seg_map


def create_pnoa_label_colormap():
  """Creates a label colormap

  Returns:
    A Colormap for visualizing segmentation results.
  """
  colormap = np.zeros((256, 3), dtype=np.uint8)
  #colormap[0] = [0, 0, 0]
  #colormap[1] = [0, 0, 255]
  #colormap[2] = [0, 255, 0]
  #colormap[3] = [255, 0, 0]
  colormap[0] = [0, 0, 0]
  colormap[1] = [255, 255, 255]

  #colormap[6] = [255, 255, 255]
  return colormap


def label_to_color_image(label):
  """Adds color defined by the dataset colormap to the label.

  Args:
    label: A 2D array with integer type, storing the segmentation label.

  Returns:
    result: A 2D array with floating type. The element of the array
      is the color indexed by the corresponding element in the input label

  Raises:
    ValueError: If label is not of rank 2 or its value is larger than color
      map maximum entry.
  """
  if label.ndim != 2:
    raise ValueError('Expect 2-D input label')

  colormap = create_pnoa_label_colormap()

  if np.max(label) >= len(colormap):
    raise ValueError('label value too large.')

  return colormap[label]


def vis_segmentation(image, seg_map, image_name):
  seg_image = label_to_color_image(seg_map).astype(np.uint8)
 ## Save raw segmentation image 
  seg_image_file = Image.fromarray(seg_image)
  seg_image_file.save(DETECTIONS_FOLDER + "/" + image_name.replace(".jpg", ".png"), format='png')


# Load pretrained 
MODEL = DeepLabModel()
print('model loaded successfully!')


def run_visualization(path, name):
  """Inferences DeepLab model and visualizes result."""
  try:
    original_im = Image.open(path)
  except IOError:
    print('Cannot retrieve image. Please check path: ' + path)
    return
  print('running deeplab on image %s...' % path)
  resized_im, seg_map = MODEL.run(original_im)
  vis_segmentation(resized_im, seg_map, name)

# Run on images folder
for root, dirs, files in os.walk(IMAGES_FOLDER, topdown=False):
  for name in files:
    path = IMAGES_FOLDER + name
    run_visualization(path, name)
print("done.")
