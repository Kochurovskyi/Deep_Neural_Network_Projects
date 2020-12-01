# Jigsaw Multilingual Toxic Comment Classification
## Classification the comments using Recurrent Neural Network (Bidirectional LSTM+Attention)
### Table of Contents
* [Introduction](#Introduction)
* [Project Structure](#Project-Structure)
* [Model Description](#Model-Description)
* [Requirements](#Requirements)
* [Training Run](#Training-Run)
* [Results and Conclusion](#Results-and-Conclusion)



### Introduction
In this challenge I will try to identify toxicity in online conversations, where toxicity is defined as anything rude, disrespectful or otherwise likely to make someone leave a discussion. If these toxic contributions can be identified, we could have a safer, more collaborative internet.

This is the third competition of its kind. The first competition in 2018 [Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge), Kagglers built multi-headed models to recognize toxicity and several subtypes of toxicity. In 2019, in the [Unintended Bias in Toxicity Classification Challenge](https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification), you worked to build toxicity models that operate fairly across a diverse range of conversations. And here it is - year 2020, another [Jigsaw Multilingual Toxic Comment Classification](https://www.kaggle.com/c/jigsaw-multilingual-toxic-comment-classification).

**What am I predicting?**
Competitors are to predict the probability that a comment is toxic. A toxic comment would receive a 1.0. A benign, non-toxic comment would receive a 0.0. In the test set, all comments are classified as either a 1.0 or a 0.0.

<img src="https://github.com/Kochurovskyi/Deep_Neural_Network_Projects/blob/main/RNN%20(Text%20Classification)/misc_items/Toxicity.png" alt="drawing" width="300"/>

### Project Structure
* Some miscellaneous data  [(**/misc_items/**)](https://github.com/Kochurovskyi/Deep_Neural_Network_Projects/tree/main/RNN%20(Text%20Classification))
* Exploratory Data Analysis Jupiter Notebook [(**EDA_toxic_class_2.ipynb**)](https://github.com/Kochurovskyi/Deep_Neural_Network_Projects/blob/main/RNN%20(Text%20Classification)/EDA_Toxic_calss_2.ipynb)
* Data processing, model training & evaluation Jupiter Notebook [(**Toxic_train_2.py**)](https://github.com/Kochurovskyi/Deep_Neural_Network_Projects/blob/main/UNet(semantic%20segmentation)/Train.py)([**Script**](https://github.com/Kochurovskyi/Deep_Neural_Network_Projects/blob/main/RNN%20(Text%20Classification)/tf.py))
* Requirements file [(**requirements.txt**)](https://github.com/Kochurovskyi/Deep_Neural_Network_Projects/blob/main/RNN%20(Text%20Classification)/requirements.txt)



### Model Description
#### Bidirectional LSTM and Attention.
Even LSTM cells can’t capture long terms dependencies to arbitrary lengths, they start to perform lesser and lesser as the sequence length increases from about 30 as explained in this paper. Attention, as the name suggests, provides a mechanism where output can ‘attend to’ (focus on) certain input time step for an input sequence of arbitrary length. 

![LSTM Arhc](https://github.com/Kochurovskyi/Deep_Neural_Network_Projects/blob/main/RNN%20(Text%20Classification)/misc_items/Model.png)

### Requirements 
This implementation is developed using:
- beautifulsoup4==4.9.3
- nltk==3.5
- numpy==1.19.4
- pandas==1.1.4
- regex==2020.10.23
- scikit-learn==0.23.2
- seaborn==0.11.0
- tensorflow==2.3.1
- tqdm==4.51.0

If **pip** is set up on your system, those packages should be able to be fetched and installed by running

<pre><code>
pip install -r requirements.txt
</code></pre>

### Training Run
No special requirements or instructions. Just place csv-files into the same folder.
In the main folder I placed a Jupiter Notebook [**Toxic_train ().ipynb**](https://github.com/Kochurovskyi/Deep_Neural_Network_Projects/blob/main/RNN%20(Text%20Classification)/Toxic_train%20(1).ipynb) with basic RNN models experiments. Here the performance history results running each model for 5 epochs:
- **Vanilla RNN**
- **Conv1D**
- **GRU**
- **LSTM**


<img src="https://github.com/Kochurovskyi/Deep_Neural_Network_Projects/blob/main/RNN%20(Text%20Classification)/misc_items/hist_Vanila.png"/>
<img src="https://github.com/Kochurovskyi/Deep_Neural_Network_Projects/blob/main/RNN%20(Text%20Classification)/misc_items/hist_Conv1D.png"/>
<img src="https://github.com/Kochurovskyi/Deep_Neural_Network_Projects/blob/main/RNN%20(Text%20Classification)/misc_items/hist_GRU.png"/>
<img src="https://github.com/Kochurovskyi/Deep_Neural_Network_Projects/blob/main/RNN%20(Text%20Classification)/misc_items/hist_LSTM.png"/>

### Results and Conclusion
All these basic models showed more or less the same evaluation performance results on the test set: **accuracy 0.85**, meanwhile advanced RNN (Bidirectional LSTM and Attention) reached which more than just epmpovement
  
