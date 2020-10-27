# Spot Nuclei. Speed Cures.
## Identifying the cells’ nuclei using Convolutional Neural Network (UNet Architecure)
### Table of Contents
* [Introduction](#Introduction)
* [Model Description](#Model_Description)
* Requirements
* Model Description


### Introduction
This project demonstrates capabilities of DNN (Deep Neural Networks) for Semantic Image Segmentation based on Kaggle Competition 2018 DATA SCIENCE BOWL (https://www.kaggle.com/c/data-science-bowl-2018). The task was related to Biomedical Science and the goal was to develop a model which able to identify a shape of cell’s nuclei reading microscope scaled images.
Why nuclei?


Identifying the cells’ nuclei is the starting point for most analyses because most of the human body’s 30 trillion cells contain a nucleus full of DNA, the genetic code that programs each cell. Identifying nuclei allows researchers to identify each individual cell in a sample, and by measuring how cells react to various treatments, the researcher can understand the underlying biological processes at work.

![Spot Nuclei. Speed Cures.](https://github.com/Kochurovskyi/Deep_Neural_Network_Projects/blob/main/UNet(semantic%20segmentation)/misc_items/dsb.jpg)

### [Model Description](#Model_Description)
#### U-Net: Convolutional Networks for Biomedical Image Segmentation.
For our task the experiencely the best option is U-net Architecture of  Convolutional Networks
The u-net is convolutional network architecture for fast and precise segmentation of images. Up to now it has outperformed the prior best method (a sliding-window convolutional network) on the ISBI challenge for segmentation of neuronal structures in electron microscopic stacks.

![UNet Arhc](https://github.com/Kochurovskyi/Deep_Neural_Network_Projects/blob/main/UNet(semantic%20segmentation)/misc_items/u-net-architecture.png)
