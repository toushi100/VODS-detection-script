from __future__ import absolute_import, division, print_function, unicode_literals

import imageio
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
from tensorflow import keras
import zipfile
import cv2
import datetime as datetime
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
from utils import label_map_util
from utils import visualization_utils as vis_util
from object_detection.utils import visualization_utils as vis_util
logger = tf.get_logger()
logger.setLevel(logging.ERROR)
# retriving dataset from this URL (it is commented out because the dataset is now local)
'''_URL = 'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'
zip_dir = tf.keras.utils.get_file('cats_and_dogs_filterted.zip', origin=_URL, extract=True)
zip_dir_base = os.path.dirname(zip_dir)'''

base_dir = 'data'
train_dir = 'data/train'
validation_dir = 'data/test'

train_cats_dir = os.path.join(train_dir, 'cats')  # directory with our training cat pictures
train_dogs_dir = os.path.join(train_dir, 'dogs')  # directory with our training dog pictures
validation_cats_dir = os.path.join(validation_dir, 'cats')  # directory with our validation cat pictures
validation_dogs_dir = os.path.join(validation_dir, 'dogs')  # directory with our validation dog pictures

num_cats_tr = len(os.listdir(train_cats_dir))
num_dogs_tr = len(os.listdir(train_dogs_dir))

num_cats_val = len(os.listdir(validation_cats_dir))
num_dogs_val = len(os.listdir(validation_dogs_dir))

total_train = num_cats_tr + num_dogs_tr
total_val = num_cats_val + num_dogs_val

print('total training cat images:', num_cats_tr)
print('total training dog images:', num_dogs_tr)

print('total validation cat images:', num_cats_val)
print('total validation dog images:', num_dogs_val)
print("--")
print("Total training images:", total_train)
print("Total validation images:", total_val)

BATCH_SIZE = 100
IMG_SHAPE = 150
'''
train_image_generator = ImageDataGenerator(rescale=1. / 255)
validation_image_generator = ImageDataGenerator(rescale=1. / 255)

train_data_gen = train_image_generator.flow_from_directory(batch_size=BATCH_SIZE,
                                                           directory=train_dir,
                                                           shuffle=True,
                                                           target_size=(IMG_SHAPE, IMG_SHAPE),
                                                           class_mode='binary')

val_data_gen = validation_image_generator.flow_from_directory(batch_size=BATCH_SIZE,
                                                              directory=validation_dir,
                                                              shuffle=False,
                                                              target_size=(IMG_SHAPE, IMG_SHAPE),
                                                              class_mode='binary')

sample_training_images, _ = next(train_data_gen)

# plotImages(sample_training_images[:5])  # Plot images 0-4


model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(IMG_SHAPE, IMG_SHAPE,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3), input_shape=(IMG_SHAPE, IMG_SHAPE)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), input_shape=(IMG_SHAPE, IMG_SHAPE)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

model.summary()

EPOCHS = 1

history = model.fit_generator(train_data_gen, steps_per_epoch=int(np.ceil(total_train / float(BATCH_SIZE))),
                              epochs=EPOCHS, validation_data=val_data_gen,
                              validation_steps=int(np.ceil(total_val / float(BATCH_SIZE))))
model.save("model_cpu.h5")

acc = history.history['acc']
val_acc = history.history['val_acc']

loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(EPOCHS)
plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.savefig('./foo.png')
plt.show()
'''
model = keras.models.load_model("model.h5")
detection_graph=tf.Graph()
with detection_graph.as_default():
    with tf.Session(graph=detection_graph)as sess:
        image_tensor= detection_graph.get_tensor_by_name('image_tensor:0')
        detection_boxes=detection_graph.get_tensor_by_name('detection_boxes:0')
        detection_scores=detection_graph.get_tensor_by_name('detection_scores:0')
        detection_classes=detection_graph.get_tensor_by_name('detection_classes:0')
        num_detection=detection_graph.get_tensor_by_name('num_detection:0')

        input_video='sample'
        video_reader=imageio.get_reader('%s.mp4'%input_video)
        video_writer=imageio.get_writer('%s_annotated.mp4'%input_video,fps=10)

        t0 = datetime.now()
        n_frames=0
        for frame in video_reader:
            image_np = frame
            n_frames +=1

            image_np_expanded = np.expand_dims(image_np,axis=0)
            (boxes, scores, classes, num) = sess.run([detection_boxes,detection_scores,detection_classes,num_detection],feed_dict={image_tensor:image_np_expanded})
            vis_util.visualize_boxes_and_labels_on_image_array(
                image_np,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index,
                use_normalized_coordinates=True,
                line_thickness=8)
            video_writer.append_data(image_np)
        fps=n_frames/(determine.now()-t0).total_seconds()
        video_writer.close()


img = image.load_img('catvdog/cat.9.jpg', target_size=(150, 150))
img = image.img_to_array(img)
img = np.expand_dims(img, axis=0)

result = model.predict(img)
print(result)
if result[0][0] == 1:
    pred = "dog"
else:
    pred = "cat"
print(pred)
