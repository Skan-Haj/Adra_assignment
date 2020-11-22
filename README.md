# Adra_assignment

## Instructions

- Unzip the CNN '6Conv250Epochs.zip' and the dataset 'xrays database_split.zip'.

- Place images to test into the 'xrays database_split/upload/1 root' and 'xrays database_split/upload/2 or more roots' folders.

- Run 'Tooth_roots_model.py' to get the accuracy of the model on the upload folder's images.

- Run 'Create_model.py' to re-build the model or create a different one.

## Comments

The average accuracy on the validation set was of 0.94.

The model used to classify the teeths roots images is a convolutional neural network. CNN was chosen as it is known to have the ablility to develop a representation of a two-dimensional image and is often used for image classification tasks. 

Multiple layer combinations were tested and the final layout is composed of 6 Convolution layers, 4 Pooling layers and 2 Full connection layers. As a starting point, a layout similar to the one of the paper from https://doi.org/10.1016/j.compbiomed.2016.11.003, using a dataset of an teeth x-ray images, was used. 

Different filter shapes were tested, similar to the shape of the roots (thin vertical rectangles), but square filters proved to be more effective.
Slight overfitting was sometime observed, dropout and spatial dropout layers were added to reduce it but it only worsened the performance of the model and were removed.

The dataset was normalized and resized for better accuracy. The metrics used to assess the performance of the model are the binary cross-entropy loss, the accuracy, the AUC and F1-score. The ADAM optimizer was used as it automates the learning-rate. 


### Improvements

- Multiply the data by intensity transformation and rotation of the training set images to further improve the accuracy.
- Find a way to effectively reduce overfitting via regularization.
- Test a wider variety of filter shapes.
- Test other activation functions for each layer.
- Use a combination of multiple models to cope with the relatively small size of the dataset.
