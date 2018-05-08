Final Project for CSCI E-89 Deep Learning Spring 2018
Professor Zoran B. Djordjević

By Fernando Trias

============

This project attempts to identify tree species form images of leaves. It uses
the Leafsnap dataset to train a MobileNet workwork. The training is implemented
in Keras, Tensorflow and OpenCV.

First, look at the `leaf-setup.ipynb` notebook to see how to download the Leafsnap
data and preprocess the images for training.

Next run `run.sh`, which will train both the CNN and MobileNet models.

The notebook `tests.ipynb` is used to explore and validate the resulting models.
