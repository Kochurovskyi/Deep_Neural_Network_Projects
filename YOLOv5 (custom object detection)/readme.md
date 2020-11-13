# Wheat Spikes Detection

### Table of Contents
* [Introduction](#Introduction)
* [Project Structure](#Project-Structure)
* [Model Description](#Model-Description)
* [Accuracy Metrics](#Accuracy-Metrics)
* [Requirements](#Requirements)
* [Training Run](#Training-Run)
* [Image Prediction and result output](#Image-Prediction-and-result-output)
* [Results and Conclusion](#Results-and-Conclusion)

### Introduction
In this mini-project, I will detect wheat heads(spikes) from outdoor images of wheat plants, including wheat datasets from around the globe. Using worldwide data, I will focus on a generalized solution to estimate the number and size of wheat heads. To better gauge the performance for unseen genotypes, environments, and observational conditions, the training dataset covers multiple regions.

<img src="https://github.com/Kochurovskyi/Deep_Neural_Network_Projects/blob/main/YOLOv5%20(custom%20object%20detection)/misc_items/%2B.jpg" alt="drawing" width="1000"/>

I am going to demonstrate capabilities of YOLOv5 convolutional Neural Networks for object detection based on Kaggle Competition **Global Wheat Detection** (https://www.kaggle.com/c/global-wheat-detection). 

### Project Structure
* Folder with input data [(**/input/**)](https://github.com/Kochurovskyi/Deep_Neural_Network_Projects/tree/main/YOLOv5%20(custom%20object%20detection)/input)
* Folder with data pre-processed data [(**/wheat_yolo_train_data/**)](https://github.com/Kochurovskyi/Deep_Neural_Network_Projects/tree/main/YOLOv5%20(custom%20object%20detection)/wheat_yolo_train_data)
* Folder with output data [(**/output/**)](https://github.com/Kochurovskyi/Deep_Neural_Network_Projects/tree/main/YOLOv5%20(custom%20object%20detection)/output)
* Some miscellaneous data  [(**/misc_items/**)](https://github.com/Kochurovskyi/Deep_Neural_Network_Projects/tree/main/YOLOv5%20(custom%20object%20detection)/misc_items)
* Exploratory Data Analysis Jupiter Notebook [(**EDA.ipynb**)](https://github.com/Kochurovskyi/Deep_Neural_Network_Projects/blob/main/YOLOv5%20(custom%20object%20detection)/EDA_wheat.ipynb)
* Data processing & model training script [(**Train.py**)](https://github.com/Kochurovskyi/Deep_Neural_Network_Projects/blob/main/UNet(semantic%20segmentation)/Train.py)
* Model prediction & result (masks) output script [(**Predict_masks.py**)](https://github.com/Kochurovskyi/Deep_Neural_Network_Projects/blob/main/UNet(semantic%20segmentation)/Predict_masks.py)
* Compiled model file [(**my_UNET.h5**)](https://github.com/Kochurovskyi/Deep_Neural_Network_Projects/blob/main/UNet(semantic%20segmentation)/my_UNET.h5)
* Requirements file [(**requirements.txt**)](https://github.com/Kochurovskyi/Deep_Neural_Network_Projects/blob/main/UNet(semantic%20segmentation)/requirements.txt)
