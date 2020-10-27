# Spot Nuclei. Speed Cures.
## Identifying the cells’ nuclei using Convolutional Neural Network (UNet Architecure)
### Table of Contents
* [Introduction](#Introduction)
* [Project Structure](#Project-Structure)
* [Model Description](#Model-Description)
* [Requirements](#Requirements)
* [Training Run](#Training-Run)


### Introduction
This project demonstrates capabilities of DNN (Deep Neural Networks) for Semantic Image Segmentation based on Kaggle Competition 2018 DATA SCIENCE BOWL (https://www.kaggle.com/c/data-science-bowl-2018). The task was related to Biomedical Science and the goal was to develop a model which able to identify a shape of cell’s nuclei reading microscope scaled images.
Why nuclei?


Identifying the cells’ nuclei is the starting point for most analyses because most of the human body’s 30 trillion cells contain a nucleus full of DNA, the genetic code that programs each cell. Identifying nuclei allows researchers to identify each individual cell in a sample, and by measuring how cells react to various treatments, the researcher can understand the underlying biological processes at work.

![Spot Nuclei. Speed Cures.](https://github.com/Kochurovskyi/Deep_Neural_Network_Projects/blob/main/UNet(semantic%20segmentation)/misc_items/dsb.jpg)

### Project Structure
* Folder with input data [(**/input/**)](https://github.com/Kochurovskyi/Deep_Neural_Network_Projects/tree/main/UNet(semantic%20segmentation)/input)
* Folder with output data [(**/output/**)](https://github.com/Kochurovskyi/Deep_Neural_Network_Projects/tree/main/UNet(semantic%20segmentation)/output)
* Some miscellaneous data  [(**/misc_items/**)](https://github.com/Kochurovskyi/Deep_Neural_Network_Projects/tree/main/UNet(semantic%20segmentation)/misc_items)
* Exploratory Data Analysis Jupiter Notebook [(**EDA.ipynb**)](https://github.com/Kochurovskyi/Deep_Neural_Network_Projects/tree/main/UNet(semantic%20segmentation)/EDA.ipynb)
* Data processing & model training script [(**Train.py**)](https://github.com/Kochurovskyi/Deep_Neural_Network_Projects/blob/main/UNet(semantic%20segmentation)/Train.py)
* Model prediction & result (masks) output script [(**Predict_masks.py**)](https://github.com/Kochurovskyi/Deep_Neural_Network_Projects/blob/main/UNet(semantic%20segmentation)/Predict_masks.py)
* Compiled model file [(**my_UNET.h5**)](https://github.com/Kochurovskyi/Deep_Neural_Network_Projects/blob/main/UNet(semantic%20segmentation)/my_UNET.h5)
* Requirements file [(**requirements.txt**)](https://github.com/Kochurovskyi/Deep_Neural_Network_Projects/blob/main/UNet(semantic%20segmentation)/requirements.txt)



### Model Description
#### U-Net: Convolutional Networks for Biomedical Image Segmentation.
For our task experiencely the best option is U-net Architecture of Convolutional Networks for fast and precise segmentation of images. Up to now it has outperformed the prior best method (a sliding-window convolutional for segmentation of neuronal structures in electron microscopic stacks.

![UNet Arhc](https://github.com/Kochurovskyi/Deep_Neural_Network_Projects/blob/main/UNet(semantic%20segmentation)/misc_items/u-net-architecture.png)

### Requirements 
This implementation is developed using:
* optparse==1.5.3
* tqdm 
* skimage==0.17.2
* cv2==4.4.0
* tensorflow==2.3.1
* matplotlib==3.3.2
* numpy==1.18.5
* pandas==1.1.2
If **pip** is set up on your system, those packages should be able to be fetched and installed by running

<pre><code>
pip install -r requirements.txt
</code></pre>

### Training Run
To run correctly the script you should decide which parameters of Model running you will choose. There are only a three parameters which will effect performance and running time of the model:
* **Image size**. The size of image for preprocessing. The Model will take as an input the images of this size (Image size X Image size). Available options [64, 128, 256]
* **Channel rate**. Normally U-net architecture contain layers with channel size from 32 to 512, but to save a time during experiments there is an option to reduce channel quantities dividing, or multiplication by 2. Available options   [0.5 , 1, 2]
* **Epochs**. Epoch quantity while model running.  Available options   5 < EPOCHS < 50
#### How to run main.py / predict_masks.py / keras_train_frcnn.py:
<pre><code>
usage: Train.py [-s Image Size] [-c Channel rate] [-e EPOCHS]
</code></pre>
#### Sample Command for **Train.py**
<pre><code>
usage: py Train.py -s 256 -c 0.5 -e 20
</code></pre>
#### Sample Command for **Predict_masks.py**
<pre><code>
usage: py Train.py -s 256 
</code></pre>

#### **!!!Be careful!!! Image size option for Train.py and Predict_masks.py must be the same**



