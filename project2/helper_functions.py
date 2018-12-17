# Helper functions
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import os, sys
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image

def load_image(infilename):
    data = mpimg.imread(infilename)
    return data


def img_float_to_uint8(img):
    rimg = img - np.min(img)
    rimg = (rimg / np.max(rimg) * 255).round().astype(np.uint8)
    return rimg


# Concatenate an image and its groundtruth
def concatenate_images(img, gt_img):
    nChannels = len(gt_img.shape)
    w = gt_img.shape[0]
    h = gt_img.shape[1]
    if nChannels == 3:
        cimg = np.concatenate((img, gt_img), axis=1)
    else:
        gt_img_3c = np.zeros((w, h, 3), dtype=np.uint8)
        gt_img8 = img_float_to_uint8(gt_img)
        gt_img_3c[:,:,0] = gt_img8
        gt_img_3c[:,:,1] = gt_img8
        gt_img_3c[:,:,2] = gt_img8
        img8 = img_float_to_uint8(img)
        cimg = np.concatenate((img8, gt_img_3c), axis=1)
    return cimg


def img_crop(im, w, h):
    list_patches = []
    imgwidth = im.shape[0]
    imgheight = im.shape[1]
    is_2d = len(im.shape) < 3
    for i in range(0,imgheight,h):
        for j in range(0,imgwidth,w):
            if is_2d:
                im_patch = im[j:j+w, i:i+h]
            else:
                im_patch = im[j:j+w, i:i+h, :]
            list_patches.append(im_patch)
    return list_patches


def augment_images(images, images_path, gts, gts_path, augment_num):
    data_gen_args  = dict(horizontal_flip=True, 
                          vertical_flip=True, 
                          height_shift_range=0.2, 
                          width_shift_range=0.2, 
                          rotation_range=20, 
                          zoom_range=[0.9, 1.25], 
                          # share_range = 0.15,
                          fill_mode="reflect"  # Because ......
                         )
    
    image_gen = ImageDataGenerator(**data_gen_args)
    gt_gen = ImageDataGenerator(**data_gen_args)
    
    save_prefix = "aug_"
    
    for i in range(len(images)):
        image_expanded = np.expand_dims(images[i], 0)
        gt_expanded = np.expand_dims(gts[i], 2)  # 1 i siste dimensjon gir gray scale???
        gt_expanded = np.expand_dims(gt_expanded, 0)

        seed = 12345
        
        # Nå bør denne også funke!!!
        """
        aug_iter = image_gen.flow(image_expanded, seed=seed, save_to_dir=images_path, save_prefix=save_prefix+str(i))
        gt_iter = gt_gen.flow(gt_expanded, seed=seed, save_to_dir=gts_path, save_prefix=save_prefix+str(i))

        for j in range(augment_num):
            next(aug_iter)
            next(gt_iter)
        """
        j = 0
        for b in image_gen.flow(image_expanded, seed=seed+j, save_to_dir=images_path, save_prefix=save_prefix+str(i)):
            j += 1
            if (j == augment_num):
                break
        
        j = 0
        for b in gt_gen.flow(gt_expanded, seed=seed+j, save_to_dir=gts_path, save_prefix=save_prefix+str(i)):
            j += 1
            if (j == augment_num):
                break
                
def add_salt_pepper_noise(X_imgs):
    # Need to produce a copy as to not modify the original image
    row, col, _ = X_imgs[0].shape
    salt_vs_pepper = 0.2
    amount = 0.006
    num_salt = np.ceil(amount * X_imgs[0].size * salt_vs_pepper)
    num_pepper = np.ceil(amount * X_imgs[0].size * (1.0 - salt_vs_pepper))
    out = []
    for X_img in X_imgs:
        # Add Salt noise
        X_img = np.array(X_img)
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in X_img.shape]
        X_img[coords[0], coords[1], :] = 1

        # Add Pepper noise
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in X_img.shape]
        X_img[coords[0], coords[1], :] = 0
        
        out.append(X_img)
    return out


def create_test_images(path, save_path, test_files, num_pictures, image_size):
    for i in range(num_pictures):
        img = Image.open(path + test_files[i] + "/" + os.listdir(path + test_files[i])[0])
        img = img.resize(image_size, Image.ANTIALIAS)
        
        # Save the Image with the same name as it had in the original test set
        img.save(save_path + str(test_files[i]) + ".png", "PNG")
        img = np.divide(np.array(img), 255.0)
        

def make_img_overlay(img, predicted_img):
    w = img.shape[0]
    h = img.shape[1]
    color_mask = np.zeros((w, h, 3), dtype=np.uint8)
    color_mask[:,:,0] = predicted_img*255

    img8 = img_float_to_uint8(img)
    background = Image.fromarray(img8, 'RGB').convert("RGBA")
    overlay = Image.fromarray(color_mask, 'RGB').convert("RGBA")
    new_img = Image.blend(background, overlay, 0.2)
    return new_img


def plot_history(history):
    plt.figure(figsize=(8, 8))
    plt.title("Learning curve")
    plt.plot(history["loss"], label="loss")
    plt.plot(history["val_loss"], label="val_loss")
    plt.plot(np.argmin(history["val_loss"]), np.min(history["val_loss"]), marker="x", color="r", label="bestmodel")
    plt.xlabel("Epochs")
    plt.ylabel("log_loss")
    plt.legend()
    plt.show()