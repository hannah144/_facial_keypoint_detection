# _facial_keypoint_detection
### **Berkeley MIDS W207 Applied Machine Learning Fall 2021 Section 02**
### **By: Hannah Gross, Anand Patel, Frances Leung & Rumi Nakagawa**
#### **[Link to Kaggle Competition](https://www.kaggle.com/c/facial-keypoints-detection)**  
***
### Project Overview
Recognizing facial keypoints, such as the center of the left eye or the tip of the nose, is a fundamental building block of biometrics, tracking faces in videos, and medical diagnosis using facial signs. The goal of this project is to build a machine learning model to predict the x,y coordinate location of 15 facial keypoints given a diverse set of 7049 facial images.

For 6 weeks, our team iterated over more than 15 convolutional neural network (CNN) models, constantly redefining our approach and model architecture to reach our best result. Our primary challenges were handling large amounts of missing facial keypoint values, and designing an architecture and strategy to minimize prediction error. The main metric we used to evaluate the "goodness" of our model was mean squared error loss (MSE Loss). This value is simply the square of the root mean squared error, the metric Kaggle uses to evaluate submissions for this competition. Below are the results of our baseline model and best model (Model 15).

**Baseline Model Results (MSE Loss)**  
&nbsp;&nbsp;&nbsp;&nbsp;On Dev Data (100 observations): 10.069  
&nbsp;&nbsp;&nbsp;&nbsp;On Test Data (1049 observations): 9.8437  

**Model 15 Results (MSE Loss)**  
&nbsp;&nbsp;&nbsp;&nbsp;On Dev Data (100 observations): 1.1887  
&nbsp;&nbsp;&nbsp;&nbsp;On Test Data (1049 observations): 2.1366  
***
### Lessons Learned
We believe the main factors that enabled this final result were the use of semi-supervised learning to fill in missing keypoint values, rotation augmentation, GPUs, and the SELU activation function.

There were many lessons learned during this project. The most important was prioritization. We found that the key was conducting light experimentation and then spending more time emphasizing the ideas that worked. The second major lesson is that there is always more to do. The model we produced performed extremely well compared to our baseline model. But still we are dissatisfied and believe there are still many other untapped paths to produce an even better model. 

Below is a detailed table of contents to help curious viewers navigate our work. Please note that these notebooks are meant to be viewed in [Google Drive](https://drive.google.com/drive/folders/1Xbn4FNmz9-zS-5vjDKguxGdSfRkB_7m7?usp=sharing) with Google Colab. Enjoy!  
***
### Table of Contents
|-- **[Kaggle_Data](https://drive.google.com/drive/folders/17PIoKgVfRCJIA2s4rnPV6GGBO4ptoFnk?usp=sharing)**  
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;*Folder contains the original data downloaded from the [Kaggle competition](https://www.kaggle.com/c/facial-keypoints-detection).*  
|&nbsp;  
|-- **[W207_CNN_Baseline_Final_Submission.ipynb](https://drive.google.com/file/d/1V5t75TCg5gbaBcb1MZuoTGwbqgLRnzM9/view?usp=sharing)**  
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;*Our team's initial submission setting a baseline result used to measure our subsequent models against.*  
|&nbsp;  
|-- **[Semi_Supervised_Learning](https://drive.google.com/drive/folders/1JZVwUT17xM8-S7Tv1ybR7SpLMi729h7r?usp=sharing)**  
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;*Folder contains the models, model checkpoints and data for our Semi-Supervised Learning work.*  
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|  
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|-- **[OG_Semi_Supervised_Model.ipynb](https://drive.google.com/file/d/16A103Y2KUSEEBaee-N06ZQgbwa-WFtlM/view?usp=sharing)**  
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;*Initial Semi-Supervised Model, uses the same architecture as the initial CNN model in Model 5.*  
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|  
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|-- **OG_Semi-Supervised_Model_checkpoints**  
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;*Model checkpoints from Initial Semi-Supervised Model. Use to recreate the OG Semi-Supervised Data.*  
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|  
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|-- **[Clean_Semi-Supervised_Model.ipynb](https://drive.google.com/file/d/1aVz4wO7JB2I6Ow9ReBDLdHEWIacHGYdE/view?usp=sharing)**  
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;*Second Semi-Supervised Model version, uses same architecture as the initial CNN model in Model 14.*  
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|  
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|-- **Clean_Semi-Supervised_Model_checkpoints**  
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;*Model checkpoints from Second Semi-Supervised Model. Use to recreate 2nd Semi-Supervised Data.*  
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|  
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|-- **[Best_Semi-Supervised_Model.ipynb](https://drive.google.com/file/d/1tmOlgv25cBN0Z0ykbqaj_LY-nSeMP0vf/view?usp=sharing)**  
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;*Third Semi-Supervised Model version, uses the same architecture as the initial CNN model in Model 14.*  
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|  
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|-- **Best_Semi-Supervised_Model_checkpoints**  
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;*Model checkpoints from Third Semi-Supervised Model. Use to recreate the 3rd Semi-Supervised Data.*  
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|  
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|-- **[Semi-Supervised_Data](https://drive.google.com/drive/folders/1WLvKbAoPAyJATuLANfjWzt6omOg7ZY8h?usp=sharing)**  
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;*Contains 3 Semi-Supervised Data Files, output of the above models.*  
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|  
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|-- **training_semiSup_filled.csv**  
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;*OG_Semi-Supervised_Model Output.*  
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|  
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|-- **sup_training_data.csv**  
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;*Clean_Semi-Supervised_Model Output.*  
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|  
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|-- **sup_training_data_model_15.csv**  
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;*Best_Semi-Supervised_Model Output.*  
|&nbsp;  
|-- **[W207_CNN_Final Submission_Model_14.ipynb](https://drive.google.com/file/d/1J9B29b_IdsS6PBSdYlIQ63RPsZQrLKfG/view?usp=sharing)**  
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;*Final CNN Model Class Submission.*    
|&nbsp;  
|-- **Model_14_checkpoints**  
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;*Model checkpoints from Model 14 Final CNN Model Submission. Use to recreate predictions.*  
|&nbsp;  
|-- **[W207_Final_Project_Presentation.pdf](https://drive.google.com/file/d/1_7eGl-EX89eBLrEYQCc-BrgCXGsFAQ7J/view?usp=sharing)**  
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;*Final W207 Presentation of Models.*  
|&nbsp;  
|-- **[Best_CNN_Model_15.ipynb](https://drive.google.com/file/d/19ah7Fu062tB9InEBOn-98etRmBe5ALdI/view?usp=sharing)**  
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;*Best CNN Model Produced.*  
|&nbsp;  
|-- **Model_15_checkpoints**  
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;*Model checkpoints from Model 15 Best CNN Model. Use to recreate predictions.*  
|&nbsp;  
|-- **[Kaggle_Submissions](https://drive.google.com/drive/folders/1q-L5gw7U5WHoQuy3fHrvyIbbmLo2Yrdo?usp=sharing)**  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;*Contains items needed for Kaggle Submissions.*  
 
