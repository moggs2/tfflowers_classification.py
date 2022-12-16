import tensorflow as tf
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow_datasets as tfds
import gc

#folderlist=[]
#
#photodir = Path('/root/flower_photos')
#
#for x in photodir.iterdir():
#    if x.is_dir():
#        folderlist.append(x.name)

#print(folderlist)

#train_ds, info = tfds.load('mnist', split='train', batch_size=32, shuffle_files=True, as_supervised=True, with_info=True)
#val_ds, info2 = tfds.load('mnist', split='test', batch_size=32, shuffle_files=True, as_supervised=True, with_info=True)

train_ds_all, info = tfds.load('tf_flowers', split='train', batch_size=2, shuffle_files=True, as_supervised=True, with_info=True)

train_ds_all = tfds.load('tf_flowers', split='train[:70%]', batch_size=32, shuffle_files=True, as_supervised=True, with_info=False)

#train_ds, val_ds, test_ds = tfds.load('tf_flowers', split=['train[:50%]', 'train[50%:75%]', 'train[75%:]'], batch_size=32, shuffle_files=True, as_supervised=True, with_info=False)

print(info)

#print(info2)

#train_ds = tf.keras.utils.image_dataset_from_directory(
#  Path('/root/pizza_steak/train'),
#  #validation_split=0.2,
#  #subset="training", #deactivate splitting
#  seed=123,
#  #image_size=(128, 128),
#  label_mode="binary",
#  #label_mode="categorical",
#  batch_size=32
#  )
#
#print(train_ds)
#
#val_ds = tf.keras.utils.image_dataset_from_directory(
#  Path('/root/pizza_steak/test'),
#  #validation_split=0.2,
#  #subset="validation", deactivate splitting
#  seed=123,
#  #image_size=(128, 128),
#  label_mode="binary",
#  #label_mode="categorical",
#  batch_size=32
#  )
#
#print(val_ds)

#class_names = train_ds.class_names
class_names=info.features['label'].names
print(class_names)


import plotext as plt
#plt.image_plot("/root/pizza_steak/test/pizza/11297.jpg")
#plt.show()
#plt.clear_figure()

from PIL import Image

import os
#imageview=tfds.as_numpy(train_ds_all)
#print(imageview)
#i=0
#for image in tfds.as_numpy(train_ds_all):   #Iterating through all batches and shows example image
#   #print(image)
#   #print(image['image'])
#   #print(image['label'])
#   print(image['image'][0][0][0])
#   print(image['label'][0])
#   print(str(i))
#   print(image['image'][0].shape)
#   img = tf.keras.preprocessing.image.array_to_img(image['image'][0])
#   img.save('pv_' + str(image['label'][0]) + '_' + str(i) + ".jpg")
#   #plt.image_plot(str(image['label'][0]) + ".jpg")
#   #plt.show()
#   #plt.clear_figure()
#   #os.remove("jpgforplotext.jpg")
#   i=i+1
#   #break

#print(int(image[1][0]))
#print(np.where(image[1][0]==1))

from tensorflow.keras import layers

data_augmentation_test = tf.keras.Sequential([
  #layers.RandomFlip("horizontal_and_vertical"),
  tf.keras.layers.RandomZoom(height_factor=(-0.2,-0.2), width_factor=(-0.2,-0.2)),
  tf.keras.layers.Resizing(350, 250),
  #layers.RandomRotation(0.05),
])

imageview=tfds.as_numpy(train_ds_all)
newaugmentedimages=[]
newlabels=[]
batchno=0
for batchimages in imageview: 
   batchshape=batchimages[0].shape
   imagesinbatch=batchshape[0]
   print("images in batch "  + str(imagesinbatch))
   i=0
   print(batchimages[0][0][0][0])
   while i < imagesinbatch:
          augmented_image = data_augmentation_test(batchimages[0][i])
          new_label=batchimages[1][i]
          newlabels.append(new_label)
          #img = tf.keras.preprocessing.image.array_to_img(augmented_image)
          newaugmentedimages.append(augmented_image.numpy())
          #img.save(str(batchimages[1][i]) + "_" + str(batchno)  + "_" + str(i) + ".jpg")
          #img2 = tf.keras.preprocessing.image.array_to_img(batchimages[0][i])
          #img2.save(str(batchimages[1][i]) + "_" + str(batchno)  + "_" + str(i) + "o.jpg")
          i=i+1
   batchno=batchno+1  

gc.collect()
#print(np.array(newaugmentedimages).shape)
print(len(newaugmentedimages))
print(newaugmentedimages[0])
print(newlabels)

dflabels=pd.get_dummies(newlabels)

newaugmentedimages=np.array(newaugmentedimages)
#newaugmentedimages=np.expand_dims(np.array(newaugmentedimages), axis=1)
#newlabels=np.expand_dims(np.array(newlabels), axis=1)
newlabels=np.array(newlabels)
print(newlabels)
newdataset = tf.data.Dataset.from_tensor_slices((newaugmentedimages, newlabels))

trainlength=round(len(newaugmentedimages)*0.6)

train_ds=newdataset.take(trainlength)

val_ds=newdataset.skip(trainlength)

train_ds=train_ds.batch(batch_size=32)

val_ds=val_ds.batch(batch_size=32)

#os.remove("jpgforplotext.jpg")

AUTOTUNE = tf.data.AUTOTUNE

#newdataset = newdataset.cache().prefetch(buffer_size=AUTOTUNE)
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

gc.collect()

#model = tf.keras.Sequential([
#    tf.keras.layers.Flatten(input_shape=[28, 28]),
#    tf.keras.layers.AlphaDropout(rate=0.2),
#    tf.keras.layers.BatchNormalization(),
#    tf.keras.layers.Dense(300, activation="relu"),
#    tf.keras.layers.AlphaDropout(rate=0.2),
#    tf.keras.layers.BatchNormalization(),
#    tf.keras.layers.Dense(100, activation="relu"),
#    tf.keras.layers.AlphaDropout(rate=0.2),
#    tf.keras.layers.BatchNormalization(),
#    tf.keras.layers.Dense(10, activation="softmax")
#])


total_imagefolders = 5 #

#model = tf.keras.Sequential([
#  tf.keras.layers.Input(shape=(350,250,3), batch_size=32),
#  #tf.keras.layers.Reshape((200,200,3), input_shape=(1,200,200,3)),
#  #tf.keras.layers.RandomWidth(factor=(0.1, 0.4), interpolation='bilinear', seed=None),
#  #tf.keras.layers.RandomHeight(factor=(0.2, 0.4), interpolation='bilinear', seed=None),
#  #tf.keras.layers.Resizing(180, 180),
#  tf.keras.layers.Rescaling(1./255),
#
#  #tf.keras.layers.RandomFlip("horizontal_and_vertical"),
#  #tf.keras.layers.RandomRotation(0.2),
#  #tf.keras.layers.RandomZoom(height_factor=(-0.2,-0.2), width_factor=(-0.2,-0.2)),
#  #Shearing an image is not included here
#
#  tf.keras.layers.Conv2D(filters=10, kernel_size=(3,3), strides=1, padding='same', activation='relu'),
#  tf.keras.layers.AveragePooling2D(pool_size=(2,2), data_format='channels_last'),
#  tf.keras.layers.Conv2D(filters=10, kernel_size=(3,3), strides=1, padding='same', activation='relu'),
#  #tf.keras.layers.Dropout(0.2),
#  tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
#  tf.keras.layers.Conv2D(filters=10, kernel_size=(2,2), strides=1, padding='same', activation='relu'),
#  
#  tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
#  #tf.keras.layers.BatchNormalization(),
#  tf.keras.layers.Conv2D(filters=10, kernel_size=(2,2), strides=1, padding='same', activation='relu'),
#  #tf.keras.layers.AveragePooling2D(pool_size=(2,2)),
#  tf.keras.layers.Flatten(),
#  #tf.keras.layers.GlobalAveragePooling2D(),
#  #tf.keras.layers.Dropout(0.2),
#  tf.keras.layers.Dense(16, activation='relu'),
#  tf.keras.layers.Dense(total_imagefolders, activation='softmax')
#  #tf.keras.layers.Dense(1, activation='sigmoid') # binary activation output
#])

#model = tf.keras.applications.resnet50.ResNet50(weights="imagenet")

xception_model=tf.keras.applications.xception.Xception(weights="imagenet", include_top=False)
for layer in xception_model.layers:
    layer.trainable=False
for layer in xception_model.layers[-3:]:
    layer.trainable=True
for layer in xception_model.layers[0:1]:
    layer.trainable=True

model = tf.keras.Sequential([
   #tf.keras.layers.Reshape((350,250,3), input_shape=(1,350,250,3)),
   tf.keras.Input(shape=(350, 250, 3)),
   tf.keras.layers.Rescaling(1./255),
   xception_model,
   tf.keras.layers.MaxPooling2D(),
   #tf.keras.layers.GlobalMaxPooling2D(),
   tf.keras.layers.Flatten(),
   tf.keras.layers.Dense(16, activation="relu"),
   tf.keras.layers.Dense(5, activation="softmax"),
   ])

model.compile(
  optimizer=tf.keras.optimizers.Adam(),
  loss=tf.losses.SparseCategoricalCrossentropy(from_logits=False),
  #loss="binary_crossentropy",   
  metrics=['accuracy'])

early_stopping_callback = tf.keras.callbacks.EarlyStopping(patience=10)
best_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint("mnist_model.h5", save_best_only=True)
learningratecallbackchange=tf.keras.callbacks.LearningRateScheduler(lambda epoch: 0.0015 * 0.9 ** epoch)

model.summary()

#fittingdiagram=model.fit(
#  train_ds,
#  validation_data=val_ds,
#  epochs=100,
#  callbacks=[best_checkpoint_callback, early_stopping_callback, learningratecallbackchange])
#  #callbacks=[best_checkpoint_callback, early_stopping_callback])

model=tf.keras.models.load_model("mnist_model.h5")


#model.evaluate(val_ds)

import plotext as plt

#plt.clp()
#plt.plot(fittingdiagram.history['loss'], xside= "lower", yside = "left", label="loss")
#plt.plot(fittingdiagram.history['accuracy'], xside= "lower", yside = "left", label="accuracy")
#plt.plot(fittingdiagram.history['val_loss'], xside= "lower", yside = "left", label="val_loss")
#plt.plot(fittingdiagram.history['val_accuracy'], xside= "lower", yside = "left", label="val_accuracy")
#plt.plot(fittingdiagram.history['lr'], xside= "lower", yside = "left", label="learning_rate")
#plt.title("Loss and accuracy")
#plt.show()

#lrloss=pd.DataFrame(fittingdiagram.history['lr'], fittingdiagram.history['loss'])
#print(lrloss)
#plt.clp()
#plt.plot(lrloss[0], xside= "lower", yside = "left", label="learning rate * 100")
#plt.plot(lrloss.index, xside= "lower", yside = "right", label="loss")
#plt.show()

gc.collect()
imageview=tfds.as_numpy(val_ds)
batchno=0
truelabellist=[]
forecastlist=[]
for batchimages in imageview:
 batchshape=batchimages[0].shape
 imagesinbatch=batchshape[0]
 i=0
 while i < imagesinbatch: 
      #pred_image_name=class_names[int(image[1][3])]
      #print(pred_image_name)
      prediction=model.predict(batchimages[0][i][np.newaxis, :, :])
      #prediction=model.predict(batchimages[0][i])  
      #print(prediction)
      predictionno=np.where(prediction[0] == np.amax(prediction[0]))[0][0]
      #print(predictionno)
      truelabel=batchimages[1][i]
      #truelabel=batchimages[1][i][0]
      #print(truelabel)
      truelabellist.append(truelabel)
      forecastlist.append(predictionno)
      if round(truelabel)!=predictionno:
          img = tf.keras.preprocessing.image.array_to_img(batchimages[0][i])
          img.save('ff_' + str(truelabel) + '_' + str(predictionno) + "_" + str(batchno)  + "_" + str(i) + ".jpg")
         
      #print(class_names[round(prediction[0][0])])
      i=i+1
 batchno=batchno+1
