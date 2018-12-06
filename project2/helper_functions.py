# Helper functions
import matplotlib.image as mpimg
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

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