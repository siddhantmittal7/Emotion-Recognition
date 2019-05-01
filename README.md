## Instructions for running the code

- Download the Fer2013 dataset using https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data. Fer2013 contains over 35,000 images grouped in seven emotion category: Anger-0, Disgust-1, Fear-2, Happy-3, Sad-4, Surprise-5 and Neutral-6. 
- Download and add Dlib Shape Predictor model using http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
- Download the Haar-casade frontalface Detection XML using https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_alt.xml
- The code was developed on Anaconda environment. The following instruction are relaated to it.
- unzip the Fer2013.zip and place the Fer2013.csv file in the code folder
- Place dlib shape predictor and haar-casde frontalface XML in the code folder
- Perform the following operation in Jupiter Notebook workbook to setup the environment

```sh
!pip install numpy
!pip install argparse
!pip install sklearn
!pip install scikit-image
!pip install pandas
!pip install hyperopt
!pip install dlib
```
- Run the image_processing_and_features_extraction.py to perform the image processiong and feature extraction process to generate Fez2013_feature folder containing the traing set and test set
```sh
python image_processing_and_features_extraction.py
```
- Train and test model using emotion_classifier.py
```sh
python emotion_classifier.py
```

# What have we achieved?

Image Processing using face detection and reshaping the images to 48X48 size to uniform the alignment

Feature extraction using
- Gabor Filter
- Histogram of Oriented Gradient (HOG)
- Facial Landmarks Extraction

Support-Vector Machine Model training on training set with RBF kernel function

Validation and testing with generation of accuracy score


# Result and performance

| Type  | Value  |  
|:-:|:-:|
|  Total Samples |  35887 |
| Number of Training Samples  | 28709  | 
| Number of Validation Samples  | 3589  | 
| Number of Test Samples | 3589  | 
| Time taken in image processing stage  | 2400.6 sec  | 
| Time taken in SVM model training  | 823.9 sec  | 
| Accuracy for 7 emotion detection  | 48.4%  | 
| Accuracy for 5 emotion detection  | 55.2%  | 





