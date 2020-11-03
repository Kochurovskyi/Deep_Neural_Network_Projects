# DNN (ResNet50) vs. XGBoost.
## Nonlinear Regression task in price prediction (ResNet50 Architecture)
### Table of Contents
* [Introduction](#Introduction)
* [Project Structure](#Project-Structure)
* [Model Description](#Model-Description)
* [Accuracy Metrics](#Accuracy-Metrics)
* [Requirements](#Requirements)
* [Model Run](#Model-Run)

### Introduction
This project demonstrates capabilities of Deep Neural Networks (DNN) for nonlinear Regression task in price prediction. As a dataset was taken a CSV file with 13 features and target **price** and the task was to predict the car price which was in use analyzing another features. Also to compare performance of Deep Neural Networks another model from classic machine learning was used, it was XGBoost.

![DNN](https://github.com/Kochurovskyi/Deep_Neural_Network_Projects/blob/main/DNN%20(ResNet50)%20vs.%20XGBoost/misc_items/hqdefault.jpg)

### Project Structure
* CSV-file with input data [(**train.csv**)](https://github.com/Kochurovskyi/Deep_Neural_Network_Projects/blob/main/DNN%20(ResNet50)%20vs.%20XGBoost/train.csv)
* 
* Some miscellaneous data  [(**/misc_items/**)](https://github.com/Kochurovskyi/Deep_Neural_Network_Projects/tree/main/DNN%20(ResNet50)%20vs.%20XGBoost/misc_items)
* Exploratory Data Analysis Jupiter Notebook [(**EDA.ipynb**)](https://github.com/Kochurovskyi/Deep_Neural_Network_Projects/blob/main/DNN%20(ResNet50)%20vs.%20XGBoost/EDA.ipynb)
* Model. Training and prediction [(**model.ipynb**)](https://github.com/Kochurovskyi/Deep_Neural_Network_Projects/blob/main/DNN%20(ResNet50)%20vs.%20XGBoost/model.ipynb)
* Feature selection script [(**feat_selection.py**)](https://github.com/Kochurovskyi/Deep_Neural_Network_Projects/blob/main/UNet(semantic%20segmentation)/Predict_masks.py)
* Profiling script. Prepares EDA html-report [(**profiling.py**)](https://github.com/Kochurovskyi/Deep_Neural_Network_Projects/blob/main/DNN%20(ResNet50)%20vs.%20XGBoost/profiling.py)
* Requirements file [(**requirements.txt**)](https://github.com/Kochurovskyi/Deep_Neural_Network_Projects/tree/main/DNN%20(ResNet50)%20vs.%20XGBoost)



### Model Description
#### ResNet50: Deep Residual Learning for Nonlinear Regression.
For our task we tried to use ResNet50 Architecture as a most powerful network.

Residual neural network is one of the most successfully applied deep networks  introduces residual shortcut connections and argues that they are indispensable for training very deep convolutional models, since the shortcuts introduce neither extra parameters nor computation complexity and increase the depth of neural network. 


![ResNet Arhc](https://github.com/Kochurovskyi/Deep_Neural_Network_Projects/blob/main/DNN%20(ResNet50)%20vs.%20XGBoost/misc_items/resnet.png)

### Accuracy Metrics
The mean absolute percentage error (MAPE) is a statistical measure of how accurate a forecast system is. It measures this accuracy as a percentage, and can be calculated as the average absolute percent error for each time period minus actual values divided by actual values.

![MAPE](https://github.com/Kochurovskyi/Deep_Neural_Network_Projects/blob/main/DNN%20(ResNet50)%20vs.%20XGBoost/misc_items/MAPE.png)


### Requirements 
This implementation is developed using:
* lightgbm==3.0.0
* numpy==1.18.5
* pandas==1.1.2
* pandas-profiling==2.9.0
* scikit-learn==0.23.2
* scipy==1.5.2
* seaborn==0.11.0
* tensorflow==2.3.1
* xgboost==1.2.0

If **pip** is set up on your system, those packages should be able to be fetched and installed by running

<pre><code>
pip install -r requirements.txt
</code></pre>

### Model Run
No special requirements or instructions. Just place csv-files into the same folder


