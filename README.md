# _facial_keypoint_detection

### **Berkeley MIDS W207 Applied Machine Learning**
### **Fall 2021 Section 02**
### **By: Hannah Gross, Anand Patel, Frances Leung & Rumi Nakagawa**
### **[Kaggle Competition](https://www.kaggle.com/c/facial-keypoints-detection)**

Recognizing facial keypoints, such as the center of the left eye or the tip of the nose, is a fundamental building block of biometrics, tracking faces in videos, and medical diagnosis using facial signs. The goal of this project is to build a machine learning model to predict the location of 15 facial keypoints given a diverse set of 7049 facial images.

For 6 weeks, our team iterated over more than 15 convolutional neural network (CNN) models, constantly redefining our approach and model architecture to reach our best result. Our primary challenges were handling large amounts of missing facial keypoint values, and designing an architecture and strategy to minimize prediction error. The main metric we used to evaluate the "goodness" of our model was mean squared error loss (MSE Loss). This value is simply the square of the root mean squared error, the metric Kaggle uses to evaluate submissions for this competition. Here are the results of our baseline model and best model (Model 15):

**Baseline Model Results**
On Dev Data (100 observations):
On Test Data (1049 observations): 

**Model 15 Results**
On Dev Data (100 observations):
On Test Data (1049 observations): 

We believe the main factors that enabled this final result were the use of semi-supervised learning to fill in missing keypoint values, rotation augmentation, GPUs, and the SELU activation function.

There were many lessons learned during this project. The most important was prioritization. We found that the key was conducting light experimentation and then spending more time emphasizing the ideas that worked. The second major lesson is that there is always more to do. The model we produced performed extremely well compared to our baseline model. But still we are dissatisfied and believe there are still many other untapped paths to produce an even better model. 

Below is a detailed table of contents to help curious viewers navigate our work. Please note that these notebooks are meant to be viewed in Google Drive with Google Colab. Enjoy!

|- **0_Kaggle_Data**
|-   *Folder contains the original data downloaded from the Kaggle competition.*
|- **1_W207_CNN_Baseline_Final_Submission.ipynb**
|-   *Our teams initial submission to set a baseline result used to measure our additional models against.*










