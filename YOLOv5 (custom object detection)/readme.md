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
* Folder with output data (.txt file with bounding boxes) [(**/output/**)](https://github.com/Kochurovskyi/Deep_Neural_Network_Projects/tree/main/YOLOv5%20(custom%20object%20detection)/output)
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

<img src="https://github.com/Kochurovskyi/Deep_Neural_Network_Projects/blob/main/YOLOv5%20(custom%20object%20detection)/misc_items/iou-fr.png" alt="drawing" width="300"/>

The metric sweeps over a range of IoU thresholds, at each point calculating an average precision value. The threshold values range from 0.5 to 0.75 with a step size of 0.05. In other words, at a threshold of 0.5, a predicted object is considered a "hit" if its intersection over union with a ground truth object is greater than 0.5.
The average precision of a single image is calculated as the mean of the above precision values at each IoU threshold
ntersection over Union (IoU)
Intersection over Union is a measure of the magnitude of overlap between two bounding boxes (or, in the more general case, two objects). It calculates the size of the overlap between two objects, divided by the total area of the two objects combined. As long as we have these two sets of bounding boxes we can apply Intersection over Union.
Below I have included a visual example of a ground-truth bounding box versus a predicted bounding box:

<img src="https://github.com/Kochurovskyi/Deep_Neural_Network_Projects/blob/main/YOLOv5%20(custom%20object%20detection)/misc_items/iou_stop_sign.jpg" alt="drawing" width="200"/>

Computing Intersection over Union can therefore be determined via:

<img src="https://github.com/Kochurovskyi/Deep_Neural_Network_Projects/blob/main/YOLOv5%20(custom%20object%20detection)/misc_items/iou_equation.png" alt="drawing" width="200"/>

Due to varying parameters of our model (image pyramid scale, sliding window size, feature extraction method, etc.), a complete and total match between predicted and ground-truth bounding boxes is simply unrealistic.

Because of this, we need to define an evaluation metric that rewards predicted bounding boxes for heavily overlapping with the ground-truth:

<img src="https://github.com/Kochurovskyi/Deep_Neural_Network_Projects/blob/main/YOLOv5%20(custom%20object%20detection)/misc_items/iou_examples.png" alt="drawing" width="300"/>

### Requirements 
This implementation is developed using:

- matplotlib==3.3.2
- numpy==1.19.4
- opencv-python==4.4.0.46
- pandas==1.1.4
- tensorflow==2.3.1
- torch==1.7.0+cu110
- tqdm==4.51.0

If **pip** is set up on your system, those packages should be able to be fetched and installed by running

<pre><code>
pip install -r requirements.txt
</code></pre>

### Training Run
Training Custom YOLOv5 Detector
To run the training command with the following options:
- img: define input image size (**1024**)
- batch: determine batch size (**4**)
- epochs: define the number of training epochs. (Note: often, 3000+ are common here!) (**20**)
- data: set the path to our yaml file (**./wheat_ds_tr/data.yaml**)
- cfg: specify our model configuration (**./wheat_ds_tr/custom_yolov5m.yaml**)
- weights: specify a custom path to weights (**./wheat_ds_tr/yolov5m.pt**)
- name: result names (**yolov5s_results**)
- nosave: only save the final checkpoint (**-**)
- cache: cache images for faster training (**-**)
- device: to select the training device, “0” for GPU, and “cpu” for CPU. (**-**)
First of, I need to be sure I change directory to the root project directory and run in the Noutbook the training command below:

<pre><code>
!python train.py --img 1024 --batch 4 --epochs 20 --data ./wheat_ds_tr/data.yaml --cfg ./wheat_ds_tr/custom_yolov5m.yaml --weights ./wheat_ds_tr/yolov5m.pt --name yolov5s_results
</code></pre>

**Training logs:**

<img src="https://github.com/Kochurovskyi/Deep_Neural_Network_Projects/blob/main/YOLOv5%20(custom%20object%20detection)/misc_items/logs.png" alt="drawing" width="1000"/>

**Training mAP:0.5-0.95 dynamics:**

<img src="https://github.com/Kochurovskyi/Deep_Neural_Network_Projects/blob/main/YOLOv5%20(custom%20object%20detection)/misc_items/mAP_05-095.png" alt="drawing" width="1000"/>


**Training mAP:0.5 dynamics:**

<img src="https://github.com/Kochurovskyi/Deep_Neural_Network_Projects/blob/main/YOLOv5%20(custom%20object%20detection)/misc_items/mAP_05.png" alt="drawing" width="1000"/>


**Training Precision:**

<img src="https://github.com/Kochurovskyi/Deep_Neural_Network_Projects/blob/main/YOLOv5%20(custom%20object%20detection)/misc_items/prec.png" alt="drawing" width="1000"/>

**Training Recall:**

<img src="https://github.com/Kochurovskyi/Deep_Neural_Network_Projects/blob/main/YOLOv5%20(custom%20object%20detection)/misc_items/rec.png" alt="drawing" width="1000"/>

### Image Prediction and result output

Now I will take my trained model and make inference on test images. For inference we invoke those weights along with a conf specifying model confidence (higher confidence required makes less predictions), and a inference source. source can accept a directory of images, individual images, video files, and also a device's webcam port. For source, I have moved all test images to the folder **./wheat_ds_val/images**
Than, run in my Notebook script with the wollowing opptions:

detect.py - script, going to be run
- --weights (**./runs/train/exp18_yolov5s_results/weights/best.pt model weigth after training**)
- --img (**1024**) - images size
- --conf (**0.5**) - trashold
- --source (**./wheat_ds_val/images**) - source of test data
- --save-txt - options to get bounding boxes for further analysis

<pre><code>
!python detect.py --weights ./runs/train/exp18_yolov5s_results/weights/best.pt --img 1024 --conf 0.5 --source ./wheat_ds_val/images --save-txt
</code></pre>

The prediction result contains files with bounding boxes ([**/output/**](https://github.com/Kochurovskyi/Deep_Neural_Network_Projects/tree/main/YOLOv5%20(custom%20object%20detection)/output)) related to each image. Some images with detected wheat spikes you will find below:

<img src="https://github.com/Kochurovskyi/Deep_Neural_Network_Projects/blob/main/YOLOv5%20(custom%20object%20detection)/misc_items/res%2B.jpg" alt="drawing" width="1000"/>


### Results and Conclusion
Analyzing training process, loss dynamics, EDA and the results it’s clear that:
-	Images are taken at different zoom levels. Crop and resize data augmentations to be used for model training.
-	Images are taken at various lighting conditions. Special filters should be used to address that.
-	Bounding boxes are messy!
-	Giant bounding boxes should be filtered out by area and removed before model training.
-	Micro bounding boxes. These can stay. They won't have much effect on the IOU metric.
-	Some spikes are not surrounded by a bounding box (missing bounding boxes).

All these problems with images affect the performance of the model. In fact I reached mean IoU score for 1000 testing images around 0.443 and this result is not the best that Yolo can give us.  This model is just a base model and there a lot of thoughts for furhter improvements. 
