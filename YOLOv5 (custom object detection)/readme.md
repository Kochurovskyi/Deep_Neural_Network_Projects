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
#### YOLOv5 “You Only Look Once”
Reference: https://github.com/ultralytics/yolov5

YOLO “You Only Look Once” is one of the most popular and most favorite algorithms for AI engineers. It always has been the first preference for real-time object detection. YOLO v1 was introduced in May 2016 by Joseph Redmon with paper “You Only Look Once: Unified, Real-Time Object Detection.” This was one of the biggest evolution in real-time object detection. And for today i was happy to discover the latest release PyTorch based version of YOLOv5 with exceptional improvements. Yolo is free open code model and in this project I used the version 3.1 released in October'20

<img src="https://github.com/Kochurovskyi/Deep_Neural_Network_Projects/blob/main/YOLOv5%20(custom%20object%20detection)/misc_items/social-image.png" alt="drawing" width="500"/>

### Accuracy Metrics
It was decided to choose a mean average precision values at different intersection over union (IoU) thresholds. The IoU of a set of predicted bounding boxes and ground truth bounding boxes is calculated as:
![Form](https://github.com/Kochurovskyi/Deep_Neural_Network_Projects/blob/main/YOLOv5%20(custom%20object%20detection)/misc_items/iou-fr.png)
The metric sweeps over a range of IoU thresholds, at each point calculating an average precision value. The threshold values range from 0.5 to 0.75 with a step size of 0.05. In other words, at a threshold of 0.5, a predicted object is considered a "hit" if its intersection over union with a ground truth object is greater than 0.5.

The average precision of a single image is calculated as the mean of the above precision values at each IoU threshold

ntersection over Union (IoU)
Intersection over Union is a measure of the magnitude of overlap between two bounding boxes (or, in the more general case, two objects). It calculates the size of the overlap between two objects, divided by the total area of the two objects combined. As long as we have these two sets of bounding boxes we can apply Intersection over Union.
Below I have included a visual example of a ground-truth bounding box versus a predicted bounding box:
![stop](https://github.com/Kochurovskyi/Deep_Neural_Network_Projects/blob/main/YOLOv5%20(custom%20object%20detection)/misc_items/iou_stop_sign.jpg)
Computing Intersection over Union can therefore be determined via:
![](https://github.com/Kochurovskyi/Deep_Neural_Network_Projects/blob/main/YOLOv5%20(custom%20object%20detection)/misc_items/iou_equation.png)

Due to varying parameters of our model (image pyramid scale, sliding window size, feature extraction method, etc.), a complete and total match between predicted and ground-truth bounding boxes is simply unrealistic.

Because of this, we need to define an evaluation metric that rewards predicted bounding boxes for heavily overlapping with the ground-truth:
![ex](https://github.com/Kochurovskyi/Deep_Neural_Network_Projects/blob/main/YOLOv5%20(custom%20object%20detection)/misc_items/iou_examples.png)
