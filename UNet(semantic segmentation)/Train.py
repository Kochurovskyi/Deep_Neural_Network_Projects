import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tqdm import tqdm
from skimage.io import imread, imshow
from skimage.transform import resize
import matplotlib.pyplot as plt
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Concatenate
from tensorflow.keras.layers import UpSampling2D, Conv2D, MaxPooling2D
import optparse


def dice_coeff(y_true, y_pred, smooth=1):
    ''' metrics function Dice-score '''
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    return (2. * intersection + smooth) / (K.sum(y_true, -1) + K.sum(y_pred, -1) + smooth)

def hs_plot(history):
    ''' history plot '''
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    dice_coeff = history.history['dice_coeff']
    val_dice_coeff = history.history['val_dice_coeff']
    epochs = range(1, len(loss) + 1)
    plt.plot(epochs, loss, color='red', label='Training loss')
    plt.plot(epochs, val_loss, color='deeppink', label='Validation loss')
    plt.plot(epochs, dice_coeff, color='lime', label='dice_coeff')
    plt.plot(epochs, val_dice_coeff, color='green', label='val_dice_coeff')
    plt.title('Training and validation loss & Metrics(Dice coeff.)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss / Dice coeff')
    plt.grid()
    plt.legend()
    plt.savefig('hist.png')
    plt.show()


# initial options: Image size for preprocessing and channel rate
parser = optparse.OptionParser()
parser.add_option('-s', action="store", dest='ims', type='int', default=128)
parser.add_option('-c', action="store", dest="ch", type='float', default=0.5)
parser.add_option('-e', action="store", dest="ep", type='int', default=20)
options, args = parser.parse_args()
if options.ims not in [64, 128, 256] or \
        options.ch not in [0.5, 1, 2] or \
        options.ep >= 50 or options.ep <= 5:
    print('Wrong input. -s[64, 128, 256], -c[1, 2, 3] 5<Epochs<50')
    print('Script Terminated!')
    exit()
print('Chosen options:')
print('Image size - {}, Channel rate - {}, Epochs - {}'.format(options.ims, options.ch, options.ep))

# open and preparation (resize) images from the train set
im_s = options.ims
tr_path = 'input/stage1_train/'
train_ids = next(os.walk(tr_path))[1][:20]
X_train = np.zeros((len(train_ids), im_s, im_s, 3), dtype=np.uint8)
Y_train = np.zeros((len(train_ids), im_s, im_s, 1), dtype=np.bool)
for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
    path = tr_path + id_
    img = imread(path + '/images/' + id_ + '.png')[:, :, :3]
    img = resize(img, (im_s, im_s), mode='constant', preserve_range=True)
    X_train[n] = img
    mask = np.zeros((im_s, im_s, 1), dtype=np.bool)
    for mask_file in next(os.walk(path + '/masks/'))[2]:
        mask_ = imread(path + '/masks/' + mask_file)
        mask_ = np.expand_dims(resize(mask_, (im_s, im_s), mode='constant',
                                      preserve_range=True), axis=-1)
        mask = np.maximum(mask, mask_)
    Y_train[n] = mask
X_train = X_train / 255
#-----------------------------------------model
ch_m = options.ch            # adjustment channels in the layers
inp = Input(shape=(im_s, im_s, 3))
conv_1_1 = Conv2D(32*ch_m, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inp)
conv_1_2 = Conv2D(32*ch_m, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv_1_1)
pool_1 = MaxPooling2D(2)(conv_1_2)

conv_2_1 = Conv2D(64*ch_m, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool_1)
conv_2_2 = Conv2D(64*ch_m, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv_2_1)
pool_2 = MaxPooling2D(2)(conv_2_2)

conv_3_1 = Conv2D(128*ch_m, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool_2)
conv_3_2 = Conv2D(128*ch_m, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv_3_1)
pool_3 = MaxPooling2D(2)(conv_3_2)

conv_4_1 = Conv2D(256*ch_m, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool_3)
conv_4_2 = Conv2D(256*ch_m, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv_4_1)
#drop_4 = Dropout(0.2)(conv_4_2)
pool_4 = MaxPooling2D(2)(conv_4_2)

conv_5_1 = Conv2D(512*ch_m, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool_4)
conv_5_2 = Conv2D(512*ch_m, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv_5_1)
#drop_5 = Dropout(0.2)(conv_5_2)

up_1 = UpSampling2D(2, interpolation='bilinear')(conv_5_2)
conc_1 = Concatenate()([conv_4_2, up_1])
conv_up_1_1 = Conv2D(256*ch_m, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conc_1)
conv_up_1_2 = Conv2D(256*ch_m, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv_up_1_1)

up_2 = UpSampling2D(2, interpolation='bilinear')(conv_up_1_2)
conc_2 = Concatenate()([conv_3_2, up_2])
conv_up_2_1 = Conv2D(128*ch_m, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conc_2)
conv_up_2_2 = Conv2D(128*ch_m, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv_up_2_1)

up_3 = UpSampling2D(2, interpolation='bilinear')(conv_up_2_2)
conc_3 = Concatenate()([conv_2_2, up_3])
conv_up_3_1 = Conv2D(64*ch_m, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conc_3)
conv_up_3_2 = Conv2D(64*ch_m, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv_up_3_1)

up_4 = UpSampling2D(2, interpolation='bilinear')(conv_up_3_2)
conc_4 = Concatenate()([conv_1_2, up_4])
conv_up_4_1 = Conv2D(32*ch_m, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conc_4)
conv_up_4_2 = Conv2D(32*ch_m, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv_up_4_1)
conv_up_4_3 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv_up_4_2)
result = Conv2D(1, 1, activation='sigmoid')(conv_up_4_3)

model = Model(inputs=inp, outputs=result)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[dice_coeff])
history = model.fit(X_train, Y_train,
                    validation_split=0.1,
                    batch_size=16,
                    epochs=options.ep,
                    verbose=1)
# output
model.save('my_UNET.h5')
hs_plot(history)


