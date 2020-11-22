from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import tensorflow as tf
import numpy as np


#Set to true to get the name and class of each file and the prediction result
details = False

model = tf.keras.models.load_model('./6Conv250Epochs')

upload_datagen = ImageDataGenerator(rescale=1/255)

upload_datagen = upload_datagen.flow_from_directory(
        './xrays database_split/upload/',
        classes = ['1 root', '2 or more roots'],
        target_size=(200, 200),
        batch_size=1,
        class_mode='binary')

accuracy = 0
accuracy_0 = 0
accuracy_1 = 0


for i in range(len(upload_datagen)):
    img, label = upload_datagen._get_batches_of_transformed_samples(np.array([i]))
    pred = int(round(model.predict(upload_datagen._get_batches_of_transformed_samples(np.array([i]))[0])[0][0]))
    if(label == pred and label == 0):
        accuracy+=1
        accuracy_0+=1
    elif(label == pred and label == 1):
        accuracy+=1
        accuracy_1+=1
    if(details):
        print((upload_datagen.filepaths[i]))
        print('Original label: ',list(upload_datagen.class_indices.keys())[int(label)],' |  Predicted label: ',list(upload_datagen.class_indices.keys())[pred],'\n')

accuracy /= len(upload_datagen)
accuracy_0 /= len(os.listdir('./xrays database_split/upload/1 root/'))
accuracy_1 /= len(os.listdir('./xrays database_split/upload/2 or more roots/'))

print('Accuracy: ', accuracy)
print('Accuracy for 1 root: ', accuracy_0)
print('Accuracy for 2 or more roots: ', accuracy_1)
