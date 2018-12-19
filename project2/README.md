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

### Installing 

Now, install the necessary data science libraries. Make sure to install them in order listed below.

```
conda install ipython
conda install jupyter
conda install pandas
conda install scipy
conda install seaborn
conda install scikit-learn
```

Finally, install the deep learning libraries, TensorFlow and Keras. Neither library is officially available via a conda package (yet), so you'll need to install them with pip. Make sure to use Python 3.6 (not 3.7) for this. 

```
pip install --upgrade tensorflow
pip install --upgrade keras
```
