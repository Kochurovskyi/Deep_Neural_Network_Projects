# Spot Nuclei. Speed Cures.
## Identifying the cells’ nuclei using Convolutional Neural Network (UNet Architecure)
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
This project demonstrates capabilities of DNN (Deep Neural Networks) for Semantic Image Segmentation based on Kaggle Competition 2018 DATA SCIENCE BOWL (https://www.kaggle.com/c/data-science-bowl-2018). The task is related to Biomedical Science and the main goal is to develop a model which will be able to identify a shape of cell’s nuclei reading microscope scaled images.
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

### Accuracy Metrics
For our task as accuracy metric was decided to choose **DICE** coefficient.
The Dice coefficient (DICE), also called the overlap index, is the most used metric in validating medical volume segmentations. In addition to the direct comparison between automatic and ground truth segmentations, it is common to use the DICE to measure reproducibility (repeatability). Using the DICE as a measure of the reproducibility as a statistical validation of manual annotation where segmenters repeatedly annotated the same MRI image, then the pair-wise overlap of the repeated segmentations is calculated using the DICE, which is defined by

![Form](https://github.com/Kochurovskyi/Deep_Neural_Network_Projects/blob/main/UNet(semantic%20segmentation)/misc_items/Dice_fmr.png)


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
* **Channel rate**. Normally U-net architecture contain layers with channel size from 32 to 512, but to save a time during experiments there is an option to reduce/increase channel quantities dividing (multiplicating) by 2. Available options   [0.5 , 1, 2]
* **Epochs**. Epoch quantity while model running.  Available options   5 < EPOCHS < 50
#### How to run Train.py / predict_masks.py:
<pre><code>
usage:  
Train.py [-s Image Size] [-c Channel rate] [-e EPOCHS]
predict_masks.py [-s Image Size]
</code></pre>
Sample Command for **Train.py**
<pre><code>
usage: py Train.py -s 256 -c 0.5 -e 20
</code></pre>
Sample Command for **Predict_masks.py**
<pre><code>
usage: py Train.py -s 256 
</code></pre>

#### **!!!Be careful!!! Image size option for Train.py and Predict_masks.py must be the same!!!**

After a couple experiments it was decided that the best balance performance/time was reached with Image size 256x256, channels rate 1(nums of channels starts from 32 and reach 512), and 30 epochs while training. Here are results:

![Log](https://github.com/Kochurovskyi/Deep_Neural_Network_Projects/blob/main/UNet(semantic%20segmentation)/misc_items/training%20log.png)
![hist](https://github.com/Kochurovskyi/Deep_Neural_Network_Projects/blob/main/UNet(semantic%20segmentation)/misc_items/hist.png)

In the end the script will save the loss/accuracy plot file **hist.png** and model itself to the file **my_UNET.h5** which will be opened in the script Predict_masks.py with all parameters for further prediction run.

### Image Prediction and result output

To get the prediction we need to run script **Predict_masks.py** with option [-s Image Size]. The size of the prediction images have to match the size of training image entered as one of the options while **Train.py ** script run. 
Sample Command for **Predict_masks.py**: 

<pre><code>
usage: py Predict_masks.py -s 256 
</code></pre>


The script will prepare (resize) the images from the test set and a couple randomly chosen images from the training set firstly. Than it will run prediction and will show to us some predicted samples from the Training set compering train masks and predicted masks. Than the script will show some examples of predicted test images.
In the end the script will check all 65 test images, predict masks and output results into the folder [(**/output/**)](https://github.com/Kochurovskyi/Deep_Neural_Network_Projects/tree/main/UNet(semantic%20segmentation)/output).

### Results and Conclusion
![(**Random Images from the Train Set**)](https://github.com/Kochurovskyi/Deep_Neural_Network_Projects/blob/main/UNet(semantic%20segmentation)/misc_items/training.png).
![(**Random Images from the Test Set**)](https://github.com/Kochurovskyi/Deep_Neural_Network_Projects/blob/main/UNet(semantic%20segmentation)/misc_items/testing.png).

On the pictures we see not bad results, but definitely there is a huge room for experiments and improvement. On the picture is clearly seen that not all nucleis were detected correctly even total accuracy score reached 96.6% . I’m sure in Biomedicine where people's lives depend on these results this will be not enough.
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
