import matplotlib.image as mpimg
import numpy as np
import matplotlib.pyplot as plt
import os, sys
import tensorflow as tf
from tensorflow.keras import layers
from PIL import Image

import helper_functions as hf
from tensorflow.keras.preprocessing.image import ImageDataGenerator









#  if __name__ == "__main__":
def run():
    print(tf.VERSION)
    print(tf.keras.__version__)

    # Load data
    training_dir = "data/training/"
    image_dir = training_dir + "images/"
    gt_dir = training_dir + "groundtruth/"
    aug_image_dir = training_dir + "aug_images/"
    aug_gt_dir = training_dir + "aug_groundtruth/"

    files = os.listdir(image_dir)
    # Load all images and groundtruths
    n = len(files)
    print(n)
    imgs = [hf.load_image(image_dir + files[i]) for i in range(n)]
    gt_imgs = [hf.load_image(gt_dir + files[i]) for i in range(n)]

    augment_num = 10

    if not os.listdir(aug_image_dir):
        print("Created augemented pictures")
        hf.augment_images(imgs, aug_image_dir, gt_imgs, aug_gt_dir, augment_num)
    else:
        print("Augmented pictures exists")

    files = os.listdir(aug_image_dir)
    # Load all images and groundtruths
    n = len(files)
    print(n)
    aug_imgs = [hf.load_image(aug_image_dir + files[i]) for i in range(n)]
    aug_gt_imgs = [hf.load_image(aug_gt_dir + files[i]) for i in range(n)]