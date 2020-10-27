# Spot Nuclei. Speed Cures.
## Identifying the cells’ nuclei using Convolutional Neural Network (UNet Architecure)
### Table of Contents
* [Introduction](#Introduction)
* [Project Structure](#Project-Structure)
* [Model Description](#Model-Description)
* [Requirements](#Requirements)
* [Training Run](# Training-Run)


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



### Model Description
#### U-Net: Convolutional Networks for Biomedical Image Segmentation.
For our task the experiencely the best option is U-net Architecture of  Convolutional Networks
The u-net is convolutional network architecture for fast and precise segmentation of images. Up to now it has outperformed the prior best method (a sliding-window convolutional network) on the ISBI challenge for segmentation of neuronal structures in electron microscopic stacks.

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

**pip install -r requirements.txt**

### Training Run
