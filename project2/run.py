import matplotlib.image as mpimg
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os, sys
import tensorflow as tf
from tensorflow.keras import layers
from PIL import Image

import helper_functions as hf
import mask_to_submission as submission_helper

from models import big_u_convnet
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, History
from tensorflow.keras.layers import Input


#  if __name__ == "__main__":
def run():
    """
    READ
    Changeable parameters:
    :MODEL_NAME:
    :EPOCHS:
    :AUGMENT_NUM:
    :SALT_AND_PEPPER:
    :BATCHNORMALIZAION:
    :DROPOUT:
    :KERNEL_SIZE: ??????????
    :DILATION_RATE:
    """
    
    print(tf.VERSION)
    print(tf.keras.__version__)

    # Load data
    training_dir = "data/training/"
    image_dir = training_dir + "images/"
    gt_dir = training_dir + "groundtruth/"
    aug_image_dir = training_dir + "aug_images/"
    aug_gt_dir = training_dir + "aug_groundtruth/"
    test_dir = "data/test_set_images/"
    aug_test_dir = "data/400_test_set/"
    
    MODEL_NAME = "uconvnet_dotd_augmented20_batchnorm_dropout0.1_dilation2"
    EPOCHS = 100
    AUGMENT_NUM = 20
    SALT_AND_PEPPER = False
    BATCHNORMALIZAION = True
    DROPOUT = 0.1
    KERNEL_SIZE = 3
    # If we want to use it, need to add a path for it, normaly it is 1 for each dimention (standard for the layers we are using
    DILATION_RATE = 2

    files = os.listdir(image_dir)
    # Load all images and groundtruths
    n = len(files)
    print("Lenght of training data: " + str(n))
    imgs = [hf.load_image(image_dir + files[i]) for i in range(n)]
    gt_imgs = [hf.load_image(gt_dir + files[i]) for i in range(n)]

    # Create augmented images if they don't exist already
    aug_imgs = []
    aug_gt_imgs = []
    if AUGMENT_NUM != 0:
        if not os.listdir(aug_image_dir):
            print("Creating augemented pictures")
            hf.augment_images(imgs, aug_image_dir, gt_imgs, aug_gt_dir, AUGMENT_NUM)
        else:
            print("Augmented pictures exists")
        
        # Load in the augmented data
        files = os.listdir(aug_image_dir)
        # Load all augmented images and groundtruths
        n = len(files)
        print("Length of Augmented data: " + str(n))
        aug_imgs = [hf.load_image(aug_image_dir + files[i]) for i in range(n)] + imgs
        aug_gt_imgs_temp = [hf.load_image(aug_gt_dir + files[i]) for i in range(n)] + gt_imgs

        aug_gt_imgs = [np.expand_dims(groundtruth, len(groundtruth.shape)) for groundtruth in aug_gt_imgs_temp]
    else:
        aug_imgs = imgs
        aug_gt_imgs = [np.expand_dims(groundtruth, len(groundtruth.shape)) for groundtruth in gt_imgs]
    
    print("Length of Augmented data + trainin data: " + str(len(aug_gt_imgs)))

    # Image shape used for the Input layer of the network
    im_height = aug_imgs[0].shape[0]
    im_width = aug_imgs[0].shape[1]
   

    # Create training and validation set
    X_train, X_valid, y_train, y_valid = train_test_split(aug_imgs, aug_gt_imgs, test_size=0.20, random_state=2018)
    
    # Add salt and pepper to approx. 50% of the pictures, but first start with all and see what happens..
    # Salt and pepper only needs to be added to the training data and not the validation or test data
    if SALT_AND_PEPPER:
        print("Adding salt and pepper to Images")
        # Does this work when the Images are in a numpy array?
        X_train = hf.add_salt_pepper_noise(X_train)

    # Turning the lists into numpy array so that they will fit into the model
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_valid = np.array(X_valid)
    y_valid = np.array(y_valid)
    
    plt.imshow(X_train[0])

    # Model
    print("Creating model with dropout: {}, batchnormalization: {} and kernelsize: {}, dilation_rate: {}".format(DROPOUT, BATCHNORMALIZAION, KERNEL_SIZE, DILATION_RATE))
    input_img = Input((im_height, im_width, 3), name='img')
    model = big_u_convnet.create_model(input_img, dropout=DROPOUT, batchnorm=BATCHNORMALIZAION, kernel_size=KERNEL_SIZE, dilation_rate=DILATION_RATE)

    # Compile model
    model.compile(optimizer=Adam(), loss="binary_crossentropy", metrics=["accuracy"])
    model.summary()

    weights_path = "models/model_weights/"
    model_weight_name = MODEL_NAME + ".h5"
    history_name = model_weight_name[:-3] + "_history"

    callbacks = [
        EarlyStopping(patience=10, verbose=1, monitor='val_loss', mode='auto'),
        ReduceLROnPlateau(factor=0.1, patience=3, min_lr=0.00001, verbose=1),
        ModelCheckpoint(weights_path + model_weight_name, monitor='val_loss', mode='auto', verbose=1,
                        save_best_only=True, save_weights_only=True)
    ]

    # Fit model IF weights don't exist, else load the weights into the model
    
    if not any(fname == model_weight_name for fname in os.listdir(weights_path)):
        print("Fitting the model over " + str(EPOCHS) + " epochs")
        results = model.fit(X_train, y_train, batch_size=4, epochs=EPOCHS, callbacks=callbacks,
                            validation_data=(X_valid, y_valid), shuffle=True)
        
        # Should save the history of the model too, need to add the", History callback then!
        with open(weights_path + history_name, 'wb') as file_pi:
            pickle.dump(results.history, file_pi)
            
    else:
        print("Model weights already exists, loading them in")
        model.load_weights(weights_path + model_weight_name)

        # Evaluate on validation set (this must be equals to the best log_loss)
        evaluation = model.evaluate(X_valid, y_valid, verbose=1)
        print(evaluation)
    
    # Variable to hold our history
    history = {}
    
    # Load in the history
    history = pickle.load(open(weights_path + history_name, "rb"))
        
    # Need to plot the history here!! Read it in if saved
    hf.plot_history(history)

    # Load the test data, reshape it if the shape is not the same as the training data
    test_files = os.listdir(test_dir)
    n = len(test_files)
    print("Number of test Images: " + str(n))
    image_size = (400, 400)
    
    test_foler_name = "test_"
    # Be careful to read the test data in the right order!
    test_imgs = [hf.load_image(test_dir + test_foler_name + str(i+1) + "/" + os.listdir(test_dir + test_foler_name + str(i+1))[0]) for i in range(n)]

    if not os.listdir(aug_test_dir):
        print("Creating 400x400 test images")
        hf.create_test_images(test_dir, aug_test_dir, test_files, n, image_size)
    else:
        print("Test images exists")
     
    # Load the 400x400 test Images
    test_400_files = os.listdir(aug_test_dir)
    n_400 = len(test_400_files)
    print("Number of Test Images: " + str(n_400))
    aug_test_imgs = [hf.load_image(aug_test_dir + "test_" + str(i+1) + ".png") for i in range(n_400)]

    print("Shape of 'augmented' test data" + str(aug_test_imgs[0].shape))
    
    X_test = np.array(aug_test_imgs)

    # Predict on test
    print("Predicting")
    predictions = model.predict(X_test, batch_size=4)
    
    prediction_path = "data/predictions/"
    result_path = prediction_path + "result_"

    original_image_size = (608, 608)

    for i in range(len(predictions)):
        im = Image.fromarray(np.uint8(np.multiply(np.squeeze(predictions[i]), 255.0)))
        im = im.resize(original_image_size, Image.ANTIALIAS)
        im.save(result_path + str(i+1) + ".png", "PNG")
    
    # Load in the prediction Images
    prediction_files = os.listdir(prediction_path)
    # Load all images and groundtruths
    n = len(prediction_files)
    print(n)
    predictions = [hf.load_image(result_path + str(i+1) + ".png") for i in range(n)]
    
    # Show a prediction
    image_index = 32
    over = hf.make_img_overlay(test_imgs[image_index], predictions[image_index])
    fig1 = plt.figure(figsize=(10, 10))
    plt.imshow(over)
    
    # Create submission
    submission_name = MODEL_NAME + ".csv"
    path_to_submission = "submissions/" + submission_name
    prediction_path = "data/predictions/"
    prediction_name = "result_"

    path_to_predictions = []

    number_of_predictions = os.listdir(prediction_path)
    
    for i in range(len(number_of_predictions)):
        path_to_predictions.append(prediction_path + prediction_name + str(i+1) + ".png")

    print("Number of predictions " + str(len(path_to_predictions)))

    submission_helper.masks_to_submission(path_to_submission, *path_to_predictions)