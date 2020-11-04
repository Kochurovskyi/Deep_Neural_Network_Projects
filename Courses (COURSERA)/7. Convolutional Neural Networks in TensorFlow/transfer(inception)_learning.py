import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import random
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Input, Dropout
from tensorflow.keras.applications.inception_v3 import InceptionV3

def hs_plot(history):
    ''' history plot '''
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    dice_coeff = history.history['acc']
    val_dice_coeff = history.history['val_acc']
    epochs = range(1, len(loss) + 1)
    plt.plot(epochs, loss, color='red', label='Training loss')
    plt.plot(epochs, val_loss, color='deeppink', label='Validation loss')
    plt.plot(epochs, dice_coeff, color='lime', label='acc')
    plt.plot(epochs, val_dice_coeff, color='green', label='val_acc')
    plt.title('Training and validation loss & Metrics(acc.)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss / Dice coeff')
    plt.grid()
    plt.legend()
    plt.savefig('hist.png')
    plt.show()

base_dir = 'input'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')

train_cats_dir = os.path.join(train_dir, 'cat')
train_dogs_dir = os.path.join(train_dir, 'dog')
validation_cats_dir = os.path.join(validation_dir, 'cat')
validation_dogs_dir = os.path.join(validation_dir, 'dog')

train_cats_fnames = os.listdir(train_cats_dir)
train_dogs_fnames = os.listdir(train_dogs_dir)
validation_cats_fnames = os.listdir(validation_cats_dir)
validation_dogs_fnames = os.listdir(validation_dogs_dir)

'''
n_samples = 4
sel_imgs_cats = random.sample(train_cats_fnames, n_samples)
sel_imgs_dogs = random.sample(train_dogs_fnames, n_samples)
fif, ax = plt.subplots(2, n_samples, figsize=(12, 6))
for n, im_file_name in enumerate(sel_imgs_cats):
    im_file_path = os.path.join(train_cats_dir, im_file_name)
    img = plt.imread(im_file_path)
    ax[0, n].imshow(img)
    ax[0, n].axis('off')
for n, im_file_name in enumerate(sel_imgs_dogs):
    im_file_path = os.path.join(train_dogs_dir, im_file_name)
    img = plt.imread(im_file_path)
    ax[1, n].imshow(img)
    ax[1, n].axis('off')
#plt.show()
'''
local_weights = 'inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'
pre_trained_model = InceptionV3(input_shape=(150, 150, 3),
                                include_top=False,
                                weights=None)
pre_trained_model.load_weights(local_weights)
for layer in pre_trained_model.layers:
  layer.trainable = False
#print(pre_trained_model.summary())
last_layer = pre_trained_model.get_layer('mixed7')
print('last layer output shape: ', last_layer.output_shape)
last_output = last_layer.output
# Flatten the output layer to 1 dimension
x = Flatten()(last_output)
# Add a fully connected layer with 1,024 hidden units and ReLU activation
x = Dense(1024, activation='relu')(x)
# Add a dropout rate of 0.2
x = Dropout(0.8)(x)
# Add a final sigmoid layer for classification
x = Dense(1, activation='sigmoid')(x)

model = Model(pre_trained_model.input, x)
model.compile(loss='binary_crossentropy',
              optimizer=RMSprop(lr=0.001),
              metrics=['acc'])

train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=.2,
                                   shear_range=.2,
                                   zoom_range=.2,
                                   horizontal_flip=True,
                                   fill_mode='nearest')
test_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(train_dir,
                                                    target_size=(150, 150),
                                                    batch_size=20,
                                                    class_mode='binary')
validation_generator = test_datagen.flow_from_directory(validation_dir,
                                                        target_size=(150, 150),
                                                        batch_size=20,
                                                        class_mode='binary')
history = model.fit(train_generator,
                            steps_per_epoch=100,
                            epochs=20,
                            validation_data=validation_generator,
                            validation_steps=50,
                            verbose=1)
model.save('my_net.h5')
hs_plot(history)
