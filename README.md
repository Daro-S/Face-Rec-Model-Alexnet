# Firstly, install the needed packages :

    pip install -r requirements-3.11.txt

Check the path for the Dataset:
the path that we set in the Jupyter notebook is Dataset/Raw/(LFW Dataset)

Since the LFW Dataset is too big so I am not going to push it to our Repository, so you must add the dataset yourself (Dataset/Raw/(LFW Dataset))

# Secondly, run the augmentation export in the terminal :

    python -m Code.preprocess.export

The Jupyter File is in the "Code\Model\AlexNet.ipynb"
The path for the Dataset Should be "Dataset\Raw\Sim_Daro"

# Face Recognition Model

## Description

This project is a face recognition model built using a variant of the AlexNet architecture. The model is trained and tested using the Labeled Faces in the Wild (LFW) dataset before being applied to real-world data.

## Data Exploration

The first step in our pipeline involves exploring the LFW dataset. We examine the size of the images, the number of classes, and other necessary information to understand the structure and characteristics of our data.

## Image Preprocessing

We perform several preprocessing steps to prepare our data for the model:

- **Data Augmentation**: To increase the diversity and amount of our training data, we apply various data augmentation techniques. The augmented data is then compressed into a .npz file for efficient storage and access.
- **Leakage Prevention**: We ensure that none of the augmented data leaks into our test set, maintaining the integrity of our testing process.
- **Image Resizing and Normalization**: All images are resized and normalized to ensure consistency in our input data.
- **Face Detection and Cropping**: We use face detection algorithms to identify faces in the images. The faces are then cropped from the images to focus the model on the relevant features.

## Model Building

We use a variant of the AlexNet architecture for our face recognition model. AlexNet is a convolutional neural network that is well-suited for image classification tasks. It automatically extracts features from the input images, eliminating the need for manual feature extraction.
