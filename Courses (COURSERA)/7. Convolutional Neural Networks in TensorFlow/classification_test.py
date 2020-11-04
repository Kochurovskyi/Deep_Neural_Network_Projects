import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image


model = load_model('my_net.h5')
play_folder = 'input/play/'
im_list = os.listdir(play_folder)
fig, ax = plt.subplots(1, len(im_list), figsize=(8, 2))
for n, im_file in enumerate(im_list):
    img_file_fold = os.path.join(play_folder, im_file)
    img = image.load_img(img_file_fold, target_size=(150, 150))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    classes = model.predict(x, batch_size=10)
    ax[n].imshow(img)
    ax[n].axis('off')
    if classes[0] > 0:
        res = ' - dog'
    else:
        res = ' - cat'
    ax[n].set_title(im_file + res)
plt.show()