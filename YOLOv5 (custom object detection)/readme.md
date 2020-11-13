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
* Data processing Jupiter Notebook [(**Pre.ipynb**)](https://github.com/Kochurovskyi/Deep_Neural_Network_Projects/tree/main/YOLOv5%20(custom%20object%20detection))
* Model training Jupiter Notebook [(**yolov5_wheat.ipynb**)](https://github.com/Kochurovskyi/Deep_Neural_Network_Projects/blob/main/YOLOv5%20(custom%20object%20detection)/yolov5_wheat.ipynb)
* Model prediction & result output Jupiter Notebook [(**Eval.ipynb**)](https://github.com/Kochurovskyi/Deep_Neural_Network_Projects/blob/main/YOLOv5%20(custom%20object%20detection)/Eval.ipynb)
* Requirements file [(**requirements.txt**)](https://github.com/Kochurovskyi/Deep_Neural_Network_Projects/blob/main/YOLOv5%20(custom%20object%20detection)/requirements.txt)

### Model Description
#### YOLOv5: Convolutional Networks for Biomedical Image Segmentation.
Reference: https://github.com/ultralytics/yolov5

YOLO “You Only Look Once” is one of the most popular and most favorite algorithms for AI engineers. It always has been the first preference for real-time object detection. YOLO v1 was introduced in May 2016 by Joseph Redmon with paper “You Only Look Once: Unified, Real-Time Object Detection.” This was one of the biggest evolution in real-time object detection. And for today i was happy to discover the latest release PyTorch based version of YOLOv5 with exceptional improvements. Yolo is free open code model and in this project I used the version 3.1 released in October'20
![img](https://github.com/Kochurovskyi/Deep_Neural_Network_Projects/blob/main/YOLOv5%20(custom%20object%20detection)/misc_items/social-image.png)
