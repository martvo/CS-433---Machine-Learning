# Machine Learning: Road Segmentation

For this problem, we are provided with a set of satellite/aerial images acquired from GoogleMaps. We are also given ground-truth images where each pixel is labeled as {road, background}. Our goal is to train a classifier to segment roads in these images, i.e. assign a label {road=1, background=0} to each pixel.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

For this project, we used Keras with a TensorFlow backend. We followed [these](http://inmachineswetrust.com/posts/deep-learning-setup/#cell3) steps for setting up a virtual environment for deep learning. For this, you'll need Anaconda (download the Python 3.7 version [here](https://www.anaconda.com/download/)). A shorter version of the approach is explained below. 

### Create a virtual environment

First start by creating a new conda environment with Python in Anaconda Prompt: 

```
conda create --name deeplearning python
```

Then activate the environment: 

```
activate deeplearning
```

### Installation 

Now, install the necessary data science libraries. Make sure to install them in order listed below.

```
conda install ipython
conda install jupyter
conda install -c conda-forge matplotlib
conda install pandas
conda install scipy
conda install scikit-learn
```

Finally, install the deep learning libraries, TensorFlow and Keras. Neither library is officially available via a conda package (yet), so you'll need to install them with pip. Make sure to use Python 3.6 (not 3.7) for this. 

```
pip install --upgrade tensorflow
pip install --upgrade keras
```

We emphasize, you need Python version 3.6 to run this project. 
We use Keras version 2.1.6, and TensorFlow version 1.12.0.

## Content

 * The `model`-folder contains the code for the convolutional U-Net, as well as all the model weights. 
 * The `submissions`-folder contains all the submission .csv files.
 * `helper_functions.py` contains all the provided helper functions, as well as the code for data augmentation.

## Deployment

In order to be able to run the code, you'll need a specific folder structure wit a root-path `project2/data/`.

In the `data`-folder, you have to make the following folders: `400_test_set`, `predictions`. In addition, you have to download the `training.zip` and `test_set_images.zip` from [CrowdAI](https://www.crowdai.org/challenges/epfl-ml-road-segmentation/dataset_files) and un-zip them in the `data`-folder. `400_test_set` and `predictions` will fill up when you run the code.

Within the new `data\training`-folder you've obtained, you'll have to make the following two folders: `aug_images` `aug_groundtruth`.

Now, you're ready to run the `run.py`. The simplest way to do this is probably to open the `run.ipynb` Jupyter notebook, and run the two cells you find there. 
