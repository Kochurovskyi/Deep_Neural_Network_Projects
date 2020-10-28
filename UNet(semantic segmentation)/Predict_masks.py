import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.keras.models import load_model
import random
import cv2
from tqdm import tqdm
from skimage.io import imread, imshow
from skimage.transform import resize
import matplotlib.pyplot as plt
import tensorflow.keras.backend as K
import optparse


def dice_coeff(y_true, y_pred, smooth=1):
    ''' metrics function Dice-score '''
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    return (2. * intersection + smooth) / (K.sum(y_true, -1) + K.sum(y_pred, -1) + smooth)

# initial options: Image size for preprocessing and channel rate
parser = optparse.OptionParser()
parser.add_option('-s', action="store", dest='ims', type='int', default=128)
options, args = parser.parse_args()
if options.ims not in [64, 128, 256]:
    print('Wrong input. -s[64, 128, 256]')
    print('Script Terminated!')
    exit()
print('Chosen Image size - {}'.format(options.ims))

# open and preparation (resize) 2 randomly chosen 2 images from the train set
im_s = options.ims
print('Choosing (randomly) converting 2 images from the train set...')
tr_path = 'input/stage1_train/'
train_ids = next(os.walk(tr_path))[1]
im_num = random.randint(0, len(train_ids)-1)
ch_train_ids = train_ids[im_num:im_num+2]
X_train = np.zeros((len(ch_train_ids), im_s, im_s, 3), dtype=np.uint8)
Y_train = np.zeros((len(ch_train_ids), im_s, im_s, 1), dtype=np.bool)
for n, id_ in tqdm(enumerate(ch_train_ids), total=len(ch_train_ids)):
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

# open and preparation (resize) test set
ts_path = 'input/stage1_test/'
print('Preparing test set...')
test_ids = next(os.walk(ts_path))[1]
X_test = np.zeros((len(test_ids), im_s, im_s, 3), dtype=np.uint8)
sizes_test = []
for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):
    path = ts_path + id_
    img = imread(path + '/images/' + id_ + '.png')[:, :, :3]
    sizes_test.append([img.shape[0], img.shape[1]])
    img = resize(img, (im_s, im_s), mode='constant', preserve_range=True)
    X_test[n] = img

# open model and run predictions
model = load_model('my_UNET.h5', custom_objects={'dice_coeff': dice_coeff})
preds_train = model.predict(X_train, verbose=3)
print('Predicting on test set (65 images)...')
preds_test = model.predict(X_test, verbose=1)

# applying trash-hold
tr_hold = 0.1
preds_test_t = (preds_test > tr_hold).astype(np.uint8)
preds_train_t = (preds_train > tr_hold).astype(np.uint8)

# Plot randomly chosen 2 prediction (with real image) from the train set
fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(8, 6))
ax[0, 0].imshow(X_train[0])
ax[0, 0].axis('off')
ax[0, 0].set_title('Real Image')
ax[0, 1].imshow(Y_train[0])
ax[0, 1].axis('off')
ax[0, 1].set_title('Train Image')
ax[0, 2].imshow(preds_train_t[0])
ax[0, 2].axis('off')
score = model.evaluate(np.expand_dims(X_train[0], axis=0),
                       np.expand_dims(Y_train[0], axis=0),
                       verbose=3)
ax[0, 2].set_title('Predicted. DICE({}%)'.format(round(score[1]*100, 1)))
ax[1, 0].imshow(X_train[1])
ax[1, 0].axis('off')
ax[1, 0].set_title('Real Image')
ax[1, 1].imshow(Y_train[1])
ax[1, 1].axis('off')
ax[1, 1].set_title('Train Image')
ax[1, 2].imshow(preds_train_t[1])
ax[1, 2].axis('off')
score = model.evaluate(np.expand_dims(X_train[1], axis=0),
                       np.expand_dims(Y_train[1], axis=0),
                       verbose=3)
ax[1, 2].set_title('Predicted. DICE({}%)'.format(round(score[1]*100, 1)))
plt.savefig('training.png')
plt.show()

# Plot randomly chosen prediction (with real image) from the test set
im_num_ts = random.randint(0, len(test_ids))
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 3))
ax[0].imshow(np.squeeze(X_test[im_num_ts]))
ax[0].axis('off')
ax[0].set_title('Real Image')
ax[1].imshow(preds_test_t[im_num_ts])
ax[1].axis('off')
ax[1].set_title('Predicted Image')
plt.savefig('testing.png')
plt.show()

# Output predicted images into folder "output" (optionally)
print('Output predicted images into folder "output" (set of 65)...')
for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):
    cv2.imwrite('output/' + test_ids[n] + ".png",
                cv2.resize((preds_test_t[n] * 255), dsize=(512, 512),
                           interpolation=cv2.INTER_LANCZOS4))


