# DNN (ResNet50) vs. XGBoost.
## Nonlinear Regression task in price prediction (ResNet50 Architecture)
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
To run correctly the script you should decide which parameters of Model running you will choose. There are only three parameters avalable to be changed which will effect performance and running time of the model:
* **Image size**. The size of image for preprocessing. The Model will take as an input the images of this size (Image size x Image size). Available options [64, 128, 256]
* **Channel rate**. Normally U-net architecture contain layers with channel size from 32 to 512, but to save a time during experiments there is an option to reduce/increase channel quantities dividing (multiplicating) by 2. Available options   [0.5 , 1, 2]
* **Epochs**. Epoch number while model running.  Available options   5 < EPOCHS < 50
#### How to run Train.py / Predict_masks.py:
<pre><code>
usage:  
Train.py [-s Image Size] [-c Channel rate] [-e EPOCHS]
Predict_masks.py [-s Image Size]
</code></pre>
Sample Command for **Train.py**
<pre><code>
usage: py Train.py -s 256 -c 0.5 -e 20
</code></pre>
Sample Command for **Predict_masks.py**
<pre><code>
usage: py Predict_masks.py -s 256 
</code></pre>

#### **!!!Be careful!!! Image size option for Train.py and Predict_masks.py must be the same!!!**

After a couple experiments it was decided that the best balance performance/time was reached with Image size 256x256, channels rate 1(nums of channels starts from 32 and reach 512), and 30 epochs while training. Here are results:

![Log](https://github.com/Kochurovskyi/Deep_Neural_Network_Projects/blob/main/UNet(semantic%20segmentation)/misc_items/training%20log.png)
![hist](https://github.com/Kochurovskyi/Deep_Neural_Network_Projects/blob/main/UNet(semantic%20segmentation)/misc_items/hist.png)

As a result the script will save the loss/accuracy to plot file **hist.png** and model itself to the file **my_UNET.h5** which will be opened in the script **Predict_masks.py** with all parameters for further prediction run.

### Image Prediction and result output

To get the prediction you need to run script **Predict_masks.py** with option [-s Image Size]. The size of the prediction images have to match the size of training image entered as one of the options while **Train.py ** script run. 
Sample Command for **Predict_masks.py**: 

<pre><code>
usage: py Predict_masks.py -s 256 
</code></pre>


The script will prepare (resize) the images from the test set and a couple randomly chosen images from the training set firstly. Than it will run prediction and will show some predicted samples from the Training set compering train masks and predicted masks. Than the script will show some example of predicted test images.
Finally the script will check all 65 test images, predict masks and output results into the folder [(**/output/**)](https://github.com/Kochurovskyi/Deep_Neural_Network_Projects/tree/main/UNet(semantic%20segmentation)/output).

### Results and Conclusion
#### Train set:
![(**Random Images from the Train Set**)](https://github.com/Kochurovskyi/Deep_Neural_Network_Projects/blob/main/UNet(semantic%20segmentation)/misc_items/training.png).
#### Test set:
![(**Random Images from the Test Set**)](https://github.com/Kochurovskyi/Deep_Neural_Network_Projects/blob/main/UNet(semantic%20segmentation)/misc_items/testing.png).

At the picture is clearly seen the result which is not so bad, but definitely there is a huge room for experiments and improvement. At the picture is clearly seen that not all nucleis were detected correctly even total accuracy score reached 96.6% . For sure in Biomedicine where people's lives depending on these results this will be not enough.
So, for the further performance improvement it would be nice to try:

Image Preprocessing:
*    Image Augmentation
* Color (Incl. background) adjustment
* Implement edge detection algorithms 

Model Adjustment: 
* Implement more channels in each layer
* Implement ResNet blocks 
* Meanwhile model complicity grows try to implement Dropout 

and more, and more, and more ...
