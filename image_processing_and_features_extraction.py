import numpy as np
import pandas as pd
import os
import argparse
import errno
import scipy.misc
import dlib
import cv2
import imageio
from skimage.filters import gabor_kernel
from scipy import ndimage as ndi

from skimage.feature import hog

# initialization
image_height = 48
image_width = 48
window_size = 24
window_step = 6
IMAGES_PER_LABEL = 10000
ONE_HOT_ENCODING = False
OUTPUT_FOLDER_NAME = "fer2013_features"
SELECTED_LABELS = [0,3,4,5,6]

# Target Width and Height of the face photo
W, H = 48, 48

# Target imgs folder, pre-processed imgs, "result" folder to save intermediate results
imgs_dir, pre_processed_imgs_dir, res_dir = 'target_imgs', 'pre-processed-imgs', 'result'

faceCascade = cv2.CascadeClassifier('haarcascade_frontalface.xml')

print( "preparing")
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
original_labels = [0, 1, 2, 3, 4, 5, 6]
new_labels = list(set(original_labels) & set(SELECTED_LABELS))
nb_images_per_label = list(np.zeros(len(new_labels), 'uint8'))

try:
    os.makedirs(OUTPUT_FOLDER_NAME)
except OSError as e:
    if e.errno == errno.EEXIST and os.path.isdir(OUTPUT_FOLDER_NAME):
        pass
    else:
        raise

def detect_face(img_gray):
    global faceCascade

    detected_face = None
    detected_face = faceCascade.detectMultiScale(img_gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    return detected_face


def get_landmarks(image, rects):
    # this function have been copied from http://bit.ly/2cj7Fpq
    if len(rects) > 1:
        raise BaseException("TooManyFaces")
    if len(rects) == 0:
        raise BaseException("NoFaces")
    return np.matrix([[p.x, p.y] for p in predictor(image, rects[0]).parts()])

def get_new_label(label, one_hot_encoding=False):
    if one_hot_encoding:
        new_label = new_labels.index(label)
        label = list(np.zeros(len(new_labels), 'uint8'))
        label[new_label] = 1
        return label
    else:
        return new_labels.index(label)
    
def compute_feats(image, kernels):
    feats = np.zeros((len(kernels), 2), dtype=np.double)
    for k, kernel in enumerate(kernels):
        filtered = ndi.convolve(image, kernel, mode='wrap')
        feats[k, 0] = filtered.mean()
        feats[k, 1] = filtered.var()
    return feats


def build_filter():    
    kernels = []
    for theta in range(4):
        theta = theta / 4. * np.pi
        for sigma in (1, 3):
            for frequency in (0.05, 0.25):
                kernel = np.real(gabor_kernel(frequency, theta=theta,
                                              sigma_x=sigma, sigma_y=sigma))
                kernels.append(kernel)
    return kernels

def scale(X, x_min, x_max):
    nom = (X-X.min(axis=0))*(x_max-x_min)
    denom = X.max(axis=0) - X.min(axis=0)
    denom[denom==0] = 1
    return x_min + nom/denom 

print( "importing csv file")
#data = pd.read_csv('fer2013.csv')
ata = pd.read_csv('smalltest.csv')

print(len(data))

for category in data['Usage'].unique():
    print( "converting set: " + category + "...")
    # create folder
    if not os.path.exists(category):
        try:
            os.makedirs(OUTPUT_FOLDER_NAME + '/' + category)
        except OSError as e:
            if e.errno == errno.EEXIST and os.path.isdir(OUTPUT_FOLDER_NAME):
               pass
            else:
                raise
    
    # get samples and labels of the actual category
    category_data = data[data['Usage'] == category]
    samples = category_data['pixels'].values
    labels = category_data['emotion'].values
    print('$$$$$$$$')
    print(len(samples))
    
    # get images and extract features
    images = []
    labels_list = []
    landmarks = []
    hog_features = []
    hog_images = []
    gabor_features = []
    for i in range(len(samples)):
        try:
            print(i)
            if labels[i] in SELECTED_LABELS and nb_images_per_label[get_new_label(labels[i])] < IMAGES_PER_LABEL:
            
                print('######')
                print(i)
                original_image = np.fromstring(samples[i], dtype=int, sep=" ")
                image = original_image.reshape((image_height, image_width))
                images.append(image)
                imageio.imwrite('temp.jpg', image)
                image2 = cv2.imread('temp.jpg')
                img_gray = cv2.imread('temp.jpg', 0)
                detected_face = detect_face(img_gray)
                #detected_face = True;
    
                if detected_face is not None:
                    
                    # TO print the pictures uncomment the following line of code
                    imageio.imwrite(OUTPUT_FOLDER_NAME + '/' + category + '/' + str(i) + '.jpg', image)
                   
                    # Following line of code computes the histogram 
                    # of Oriented Gradients (HOG) features for the images 
                    features, hog_image = hog(image, orientations=8, pixels_per_cell=(16, 16),
                                        cells_per_block=(1, 1), visualise=True)
                    hog_features.append(features)
                    hog_images.append(hog_image)
                    
                    #print(features)
                    
        
                    # Following line of code gets the facial features of 
                    # the image by extraction the landmarks like 
                    # eye location, lips location, nose size etc
                    
                    imageio.imwrite('temp.jpg', image)
                    image2 = cv2.imread('temp.jpg')
                    face_rects = [dlib.rectangle(left=1, top=1, right=47, bottom=47)]
                    face_landmarks = get_landmarks(image2, face_rects)
                    landmarks.append(face_landmarks)    
                    
                    #print(face_landmarks)
                    
                    # Following code extract the features based on the gabor filters
                    kernels = build_filter();
                    gabor_feat = compute_feats(image, kernels)
                    
                    gabor_feat_scaled = scale(gabor_feat, -1, 1)
                    
                    gabor_features.append(gabor_feat_scaled)
                    
                    #print(gabor_feat_scaled)
                        
                labels_list.append(get_new_label(labels[i], one_hot_encoding=ONE_HOT_ENCODING))
                nb_images_per_label[get_new_label(labels[i])] += 1
        except Exception as e:
            print( "error in image: " + str(i) + " - " + str(e))

    np.save(OUTPUT_FOLDER_NAME + '/' + category + '/images.npy', images)
    np.save(OUTPUT_FOLDER_NAME + '/' + category + '/landmarks.npy', landmarks)
    np.save(OUTPUT_FOLDER_NAME + '/' + category + '/labels.npy', labels_list)
    np.save(OUTPUT_FOLDER_NAME + '/' + category + '/hog_features.npy', hog_features)
    np.save(OUTPUT_FOLDER_NAME + '/' + category + '/hog_images.npy', hog_images)
    np.save(OUTPUT_FOLDER_NAME + '/' + category + '/gabor_features.npy', gabor_features)
