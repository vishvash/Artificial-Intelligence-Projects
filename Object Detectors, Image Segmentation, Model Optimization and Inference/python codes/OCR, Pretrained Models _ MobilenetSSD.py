#import the libraries
# pip install pytesseract

import pytesseract
import pkg_resources
import cv2

#declaring the exe path for tesseract
pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'

#printing the tesseract version
print(pkg_resources.working_set.by_key['pytesseract'].version)

#print the opencv version
print(cv2.__version__)

from PIL import Image
import pytesseract
import cv2

#loading the image from the disk
image_to_ocr = cv2.imread('C:/Users/Lenovo/Downloads/Study material/AI/Object Detectors, Image Segmentation, Model Optimization and Inference/Images/sampletxt.png')

#preprocessing the image
# step 1 : covert to grey scale
preprocessed_img = cv2.cvtColor(image_to_ocr, cv2.COLOR_BGR2GRAY)
# step 2 : Do binary and otsu thresholding
preprocessed_img = cv2.threshold(preprocessed_img,0,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
# step 3 : Smooth the image using median blur
preprocessed_img = cv2.medianBlur(preprocessed_img, 3)

#save the preprocessed image temporarily into the disk
cv2.imwrite('temp_img.jpg',preprocessed_img)

#read the temp image from disk as pil image
preprocessed_pil_img = Image.open('C:/Users/Lenovo/temp_img.jpg')

#pass the pil image to tesseract to do OCR
text_extracted = pytesseract.image_to_string(preprocessed_pil_img)

#print the text
print(text_extracted)

#display the original image
cv2.imshow("Actual Image",image_to_ocr)
cv2.waitKey(0)
cv2.destroyAllWindows()

from keras.applications import VGG16
from keras.applications import imagenet_utils
from keras.preprocessing.image import img_to_array, load_img
import numpy as np
import cv2

#loading the image to predict
img_path = 'C:/Users/Lenovo/Downloads/Study material/AI/Object Detectors, Image Segmentation, Model Optimization and Inference/Images/frog.jpg'
img = load_img(img_path)

#resize the image to 224x224 square shape
img = img.resize((224,224))
#convert the image to array
img_array = img_to_array(img)

#convert the image into a 4 dimensional Tensor
#convert from (height, width, channels), (batchsize, height, width, channels)
img_array = np.expand_dims(img_array, axis=0)

#preprocess the input image array
img_array = imagenet_utils.preprocess_input(img_array)

#Load the model from internet / computer
#approximately 530 MB
pretrained_model = VGG16(weights="imagenet")

#predict using predict() method
prediction = pretrained_model.predict(img_array)

#decode the prediction
actual_prediction = imagenet_utils.decode_predictions(prediction)

print("predicted object is:")
print(actual_prediction[0][0][1])
print("with accuracy")
print(actual_prediction[0][0][2]*100)

#display image and the prediction text over it
disp_img = cv2.imread(img_path)
#display prediction text over the image
cv2.putText(disp_img, actual_prediction[0][0][1], (100,50), cv2.FONT_HERSHEY_TRIPLEX , 0.8, (255,0,0))
#show the image
cv2.imshow("Prediction",disp_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

from keras.applications import VGG19
from keras.applications import imagenet_utils
from keras.preprocessing.image import img_to_array, load_img
import numpy as np
import cv2

#loading the image to predict
img_path = 'C:/Users/Lenovo/Downloads/Study material/AI/Object Detectors, Image Segmentation, Model Optimization and Inference/Images/plane.jpg'
img = load_img(img_path)

#resize the image to 224x224 square shape
img = img.resize((224,224))

img_array = img_to_array(img)

#convert the image into a 4 dimensional Tensor
#convert from (height, width, channels), (batchsize, height, width, channels)
img_array = np.expand_dims(img_array, axis=0)

#preprocess the input image array
img_array = imagenet_utils.preprocess_input(img_array)

#Load the model from internet / computer
#approximately 530 MB
pretrained_model = VGG19(weights="imagenet")

#predict using predict() method
prediction = pretrained_model.predict(img_array)

#decode the prediction
actual_prediction = imagenet_utils.decode_predictions(prediction)

print("predicted object is:")
print(actual_prediction[0][0][1])
print("with accuracy")
print(actual_prediction[0][0][2]*100)

#display image and the prediction text over it
disp_img = cv2.imread(img_path)
#display prediction text over the image
cv2.putText(disp_img, actual_prediction[0][0][1], (20,20), cv2.FONT_HERSHEY_TRIPLEX , 0.8, (0,0,0))
#show the image
cv2.imshow("Prediction",disp_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

from keras.applications import ResNet50
from keras.applications import imagenet_utils
from keras.preprocessing.image import img_to_array, load_img
import numpy as np
import cv2

#loading the image to predict
img_path = 'C:/Users/Lenovo/Downloads/Study material/AI/Object Detectors, Image Segmentation, Model Optimization and Inference/Images/scene.jpg'
img = load_img(img_path)

#resize the image to 224x224 square shape
img = img.resize((224,224))
#convert the image to array
img_array = img_to_array(img)

#convert the image into a 4 dimensional Tensor
#convert from (height, width, channels), (batchsize, height, width, channels)
img_array = np.expand_dims(img_array, axis=0)

#preprocess the input image array
img_array = imagenet_utils.preprocess_input(img_array)

#Load the model from internet / computer
#approximately 102 MB
pretrained_model = ResNet50(weights="imagenet")

#predict using predict() method
prediction = pretrained_model.predict(img_array)

#decode the prediction
actual_prediction = imagenet_utils.decode_predictions(prediction)

print("predicted object is:")
print(actual_prediction[0][0][1])
print("with accuracy")
print(actual_prediction[0][0][2]*100)

#display image and the prediction text over it
disp_img = cv2.imread(img_path)
#display prediction text over the image
cv2.putText(disp_img, actual_prediction[0][0][1], (20,20), cv2.FONT_HERSHEY_TRIPLEX , 0.8, (0,0,0))
#show the image
cv2.imshow("Prediction",disp_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

from keras.applications import InceptionV3
from keras.applications import imagenet_utils
from keras.preprocessing.image import img_to_array, load_img
from keras.applications.inception_v3 import preprocess_input
import numpy as np
import cv2

#loading the image to predict
img_path = 'C://Users//Bharani Kumar//Desktop//AI & DL//Object Detection//liner.jpg'
img = load_img(img_path)

#resize the image to 299x299 square shape
img = img.resize((299,299))
#convert the image to array
img_array = img_to_array(img)

#convert the image into a 4 dimensional Tensor
#convert from (height, width, channels), (batchsize, height, width, channels)
img_array = np.expand_dims(img_array, axis=0)

#preprocess the input image array
img_array = preprocess_input(img_array)

#Load the model from internet / computer
#approximately 96 MB
pretrained_model = InceptionV3(weights="imagenet")

#predict using predict() method
prediction = pretrained_model.predict(img_array)

#decode the prediction
actual_prediction = imagenet_utils.decode_predictions(prediction)

print("predicted object is:")
print(actual_prediction[0][0][1])
print("with accuracy")
print(actual_prediction[0][0][2]*100)

#display image and the prediction text over it
disp_img = cv2.imread(img_path)
#display prediction text over the image
cv2.putText(disp_img, actual_prediction[0][0][1], (20,20), cv2.FONT_HERSHEY_TRIPLEX , 0.8, (0,0,0))
#show the image
cv2.imshow("Prediction",disp_img)

from keras.applications import Xception
from keras.applications import imagenet_utils
from keras.preprocessing.image import img_to_array, load_img
from keras.applications.inception_v3 import preprocess_input
import numpy as np
import cv2

#loading the image to predict
img_path = 'C://Users//Bharani Kumar//Desktop//AI & DL//Object Detection//hen.jpg'
img = load_img(img_path)

#resize the image to 299x299 square shape
img = img.resize((299,299))
#convert the image to array
img_array = img_to_array(img)

#convert the image into a 4 dimensional Tensor
#convert from (height, width, channels), (batchsize, height, width, channels)
img_array = np.expand_dims(img_array, axis=0)

#preprocess the input image array
img_array = preprocess_input(img_array)

#Load the model from internet / computer
#approximately 91 MB
pretrained_model = Xception(weights="imagenet")

#predict using predict() method
prediction = pretrained_model.predict(img_array)

#decode the prediction
actual_prediction = imagenet_utils.decode_predictions(prediction)

print("predicted object is:")
print(actual_prediction[0][0][1])
print("with accuracy")
print(actual_prediction[0][0][2]*100)

#display image and the prediction text over it
disp_img = cv2.imread(img_path)
#display prediction text over the image
cv2.putText(disp_img, actual_prediction[0][0][1], (20,20), cv2.FONT_HERSHEY_TRIPLEX , 0.8, (0,0,0))
#show the image
cv2.imshow("Prediction",disp_img)


import numpy as  np
import cv2

# load the image to detect, get width, height 
# resize to match input size, convert to blob to pass into model
img_to_detect = cv2.imread('images/sampletext.jpg')
img_height = img_to_detect.shape[0]
img_width = img_to_detect.shape[1]
resized_img_to_detect = cv2.resize(img_to_detect,(300,300))
img_blob = cv2.dnn.blobFromImage(resized_img_to_detect,0.007843,(300,300),127.5)
#recommended scale factor is 0.007843, width,height of blob is 300,300, mean of 255 is 127.5, 

# set of 21 class labels in alphabetical order (background + rest of 20 classes)
class_labels = ["background", "aeroplane", "bicycle", "bird", "boat","bottle", "bus", "car", "cat", "chair", "cow", "diningtable","dog", "horse", "motorbike", "person", "pottedplant", "sheep","sofa", "train", "tvmonitor"]

# Loading pretrained model from prototext and caffemodel files
# input preprocessed blob into model and pass through the model
# obtain the detection predictions by the model using forward() method
mobilenetssd = cv2.dnn.readNetFromCaffe('mobilenetssd.prototext','mobilenetssd.caffemodel')
mobilenetssd.setInput(img_blob)
obj_detections = mobilenetssd.forward()
# returned obj_detections[0, 0, index, 1] , 1 => will have the prediction class index
# 2 => will have confidence, 3 to 7 => will have the bounding box co-ordinates
no_of_detections = obj_detections.shape[2]

# loop over the detections
for index in np.arange(0, no_of_detections):
    prediction_confidence = obj_detections[0, 0, index, 2]
    # take only predictions with confidence more than 20%
    if prediction_confidence > 0.20:
        
        #get the predicted label
        predicted_class_index = int(obj_detections[0, 0, index, 1])
        predicted_class_label = class_labels[predicted_class_index]
        
        #obtain the bounding box co-oridnates for actual image from resized image size
        bounding_box = obj_detections[0, 0, index, 3:7] * np.array([img_width, img_height, img_width, img_height])
        (start_x_pt, start_y_pt, end_x_pt, end_y_pt) = bounding_box.astype("int")
        
        # print the prediction in console
        predicted_class_label = "{}: {:.2f}%".format(class_labels[predicted_class_index], prediction_confidence * 100)
        print("predicted object {}: {}".format(index+1, predicted_class_label))
        
        # draw rectangle and text in the image
        cv2.rectangle(img_to_detect, (start_x_pt, start_y_pt), (end_x_pt, end_y_pt), (0,255,0), 2)
        cv2.putText(img_to_detect, predicted_class_label, (start_x_pt, start_y_pt-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

cv2.imshow("Detection Output", img_to_detect)

import numpy as  np
import cv2

#get the saved video file as stream
file_video_stream = cv2.VideoCapture('C://Users//Bharani Kumar//Desktop//AI & DL//Object Detection//video.mp4')

#create a while loop 
while (file_video_stream.isOpened):
    #get the current frame from video stream
    ret,current_frame = file_video_stream.read()
    #use the video current frame instead of image
    img_to_detect = current_frame
    img_height = img_to_detect.shape[0]
    img_width = img_to_detect.shape[1]
    # resize to match input size, convert to blob to pass into model
    resized_img_to_detect = cv2.resize(img_to_detect,(300,300))
    img_blob = cv2.dnn.blobFromImage(resized_img_to_detect,0.007843,(300,300),127.5)
    #recommended scale factor is 0.007843, width,height of blob is 300,300, mean of 255 is 127.5, 

    # set of 21 class labels in alphabetical order (background + rest of 20 classes)
    class_labels = ["background", "aeroplane", "bicycle", "bird", "boat","bottle", "bus", "car", "cat", "chair", "cow", "diningtable","dog", "horse", "motorbike", "person", "pottedplant", "sheep","sofa", "train", "tvmonitor"]
    
    # Loading pretrained model from prototext and caffemodel files
    # input preprocessed blob into model and pass through the model
    # obtain the detection predictions by the model using forward() method
    mobilenetssd = cv2.dnn.readNetFromCaffe('C://Users//Bharani Kumar//Desktop//AI & DL//Object Detection//mobilenetssd.prototext','C://Users//Bharani Kumar//Desktop//AI & DL//Object Detection//mobilenetssd.caffemodel')
    mobilenetssd.setInput(img_blob)
    obj_detections = mobilenetssd.forward()
    # returned obj_detections[0, 0, index, 1] , 1 => will have the prediction class index
    # 2 => will have confidence, 3 to 7 => will have the bounding box co-ordinates
    no_of_detections = obj_detections.shape[2]
    
    # loop over the detections
    for index in np.arange(0, no_of_detections):
        prediction_confidence = obj_detections[0, 0, index, 2]
        # take only predictions with confidence more than 20%
        if prediction_confidence > 0.20:
            
            #get the predicted label
            predicted_class_index = int(obj_detections[0, 0, index, 1])
            predicted_class_label = class_labels[predicted_class_index]
            
            #obtain the bounding box co-oridnates for actual image from resized image size
            bounding_box = obj_detections[0, 0, index, 3:7] * np.array([img_width, img_height, img_width, img_height])
            (start_x_pt, start_y_pt, end_x_pt, end_y_pt) = bounding_box.astype("int")
            
            # print the prediction in console
            predicted_class_label = "{}: {:.2f}%".format(class_labels[predicted_class_index], prediction_confidence * 100)
            print("predicted object {}: {}".format(index+1, predicted_class_label))
            
            # draw rectangle and text in the image
            cv2.rectangle(img_to_detect, (start_x_pt, start_y_pt), (end_x_pt, end_y_pt), (0,255,0), 2)
            cv2.putText(img_to_detect, predicted_class_label, (start_x_pt, start_y_pt-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
    
    
    cv2.imshow("Detection Output", img_to_detect)
    
    #terminate while loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#releasing the stream
#close all opencv windows
file_video_stream.release()
cv2.destroyAllWindows()

