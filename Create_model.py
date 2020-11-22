import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import numpy as np

train_1root_dir = os.path.join('./xrays database_split/train/1 root')
try:
    os.remove('./xrays database_split/train/1 root/.DS_Store')
except:
    pass

train_2roots_dir = os.path.join('./xrays database_split/train/2 or more roots')
try:
    os.remove('./xrays database_split/train/2 or more roots/.DS_Store')
except:
    pass

valid_1root_dir = os.path.join('./xrays database_split/test/1 root')
try:
    os.remove('./xrays database_split/test/1 root/.DS_Store')
except:
    pass

valid_2roots_dir = os.path.join('./xrays database_split/test/2 or more roots')
try:
    os.remove('./xrays database_split/test/2 or more roots/.DS_Store')
except:
    pass

print('total training 1 root images:', len(os.listdir(train_1root_dir)))
print('total training 2 or more roots images:', len(os.listdir(train_2roots_dir)))
print('total validation 1 root images:', len(os.listdir(valid_1root_dir)))
print('total validation 2 or more roots images:', len(os.listdir(valid_2roots_dir)))

train_datagen = ImageDataGenerator(rescale=1/255)
validation_datagen = ImageDataGenerator(rescale=1/255)

train_generator = train_datagen.flow_from_directory(
        './xrays database_split/train/',
        classes = ['1 root', '2 or more roots'],
        target_size=(200, 200),
        batch_size=10,
        class_mode='binary')

validation_generator = validation_datagen.flow_from_directory(
        './xrays database_split/test/',
        classes = ['1 root', '2 or more roots'],
        target_size=(200, 200),
        batch_size=4,
        class_mode='binary',
        shuffle=False)

model = tf.keras.models.Sequential([tf.keras.layers.Conv2D(16, (5,5), activation='relu', input_shape=(200, 200, 3)),
                                    tf.keras.layers.MaxPooling2D(2, 2),
                                    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
                                    tf.keras.layers.MaxPooling2D(2,2),
                                    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
                                    tf.keras.layers.MaxPooling2D(2,2),
                                    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
                                    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
                                    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
                                    tf.keras.layers.MaxPooling2D(2,2),

                                    tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(512, activation='relu'),
                                    tf.keras.layers.Dense(1, activation='sigmoid')])

print(model.summary())

model.compile(optimizer = tf.optimizers.Adam(),
              loss = 'binary_crossentropy',
              metrics=['accuracy',tf.keras.metrics.AUC(),tf.keras.metrics.Precision(),tf.keras.metrics.Recall()])

history = model.fit(train_generator,
      steps_per_epoch=8,
      epochs=250,
      verbose=1,
      validation_data = validation_generator,
      validation_steps=8)

model.evaluate(validation_generator)
model.save('./6Conv250Epochs')
