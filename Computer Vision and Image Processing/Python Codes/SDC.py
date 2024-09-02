#Importing important libraries

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2

image_color = mpimg.imread('C:/Users/Bharani Kumar/Desktop/AI/SDCs/image.jpg')
plt.imshow(image_color)

image_color.shape
image_gray = cv2.cvtColor(image_color, cv2.COLOR_BGR2GRAY)
plt.imshow(image_gray, cmap = 'gray')
image_gray.shape
image_copy = np.copy(image_gray)
image_copy[ (image_copy[:,:] < 195) ] = 0 
plt.imshow(image_copy, cmap = 'gray')
plt.show()

image_copy = np.copy(image_color)
image_copy[ (image_copy[:,:,0] < 200) | (image_copy[:,:,1] < 200) | (image_copy[:,:,2] < 200) ] = 0 
plt.imshow(image_copy, cmap = 'gray')
plt.show()

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2

image = cv2.imread('C:/Users/Bharani Kumar/Desktop/AI/SDCs/test_image.jpg')
image.shape
print ('Height = ', int(image.shape[0]), 'pixels')
print ('Width = ', int(image.shape[1]), 'pixels')

cv2.imshow('Self Driving Car!', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

plt.imshow(image)
image.shape

gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow('Self Driving Car in Grayscale!', gray_img)
cv2.waitKey()
cv2.destroyAllWindows()

gray_img.shape

import cv2
image = cv2.imread('C:/Users/Bharani Kumar/Desktop/AI/SDCs/test_image.jpg')
cv2.imshow('Self Driving Car!', image)
cv2.waitKey()
cv2.destroyAllWindows()
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
cv2.imshow('HSV Image', hsv_image)
cv2.waitKey()
cv2.destroyAllWindows()

import matplotlib.pyplot as plt
plt.imshow(hsv_image[:, :, 0])
plt.title('Hue channel')

plt.imshow(hsv_image[:, :, 1])
plt.title('Saturation channel')

plt.imshow(hsv_image[:, :, 2])
plt.title('Value channel')

image = cv2.imread('C:/Users/Bharani Kumar/Desktop/AI/SDCs/test_image.jpg')
B, G, R = cv2.split(image)
B.shape
G.shape

cv2.imshow("Blue Channel!", B) 
cv2.waitKey(0)
cv2.destroyAllWindows()

zeros = np.zeros(image.shape[:2], dtype = "uint8")
cv2.imshow("Blue Channel", cv2.merge([B, zeros, zeros]))
cv2.waitKey(0)
cv2.destroyAllWindows()

image_merged = cv2.merge([B, G, R])      
cv2.imshow("Merged Image!", image_merged) 

cv2.waitKey(0)
cv2.destroyAllWindows()

image_merged = cv2.merge([B, G+100, R]) 
cv2.imshow("Merged Image with some added green!", image_merged) 

cv2.waitKey(0)
cv2.destroyAllWindows()


import cv2
import numpy as np
import matplotlib.image as mpimg
from matplotlib import pyplot as plt
%matplotlib inline

image_color = cv2.imread('C:/Users/Bharani Kumar/Desktop/AI/SDCs/test5.jpg')
cv2.imshow('Original Image', image_color)
cv2.waitKey()
cv2.destroyAllWindows()

height, width = image_color.shape[:2]
height
width

# Converting image in grayscale
image_gray = cv2.cvtColor(image_color, cv2.COLOR_BGR2GRAY)
plt.imshow(image_gray, cmap = 'gray')
#Point of interest in the image
ROI = np.array([[(0, 400),(300, 250), (450, 300), (700, height)]], dtype=np.int32)
#Define blank image with all zeros
blank = np.zeros_like(image_gray)
blank.shape
#Filling mask of interest
mask = cv2.fillPoly(blank, ROI, 255)
#Perform a bit-wise AND operation
masked_image = cv2.bitwise_and(image_gray, mask)
#Masked Image
plt.imshow(masked_image, cmap = 'gray')

#Importing openCV
import cv2

#Displaying image

image = cv2.imread('C:/Users/Bharani Kumar/Desktop/AI/SDCs/test_image_Lane.jpg')
cv2.imshow('input_image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

import cv2
import numpy as np
image = cv2.imread('C:/Users/Bharani Kumar/Desktop/AI/SDCs/test_image_Lane.jpg')
lanelines_image = np.copy(image)
gray_conversion= cv2.cvtColor(lanelines_image, cv2.COLOR_RGB2GRAY)
cv2.imshow('input_image', gray_conversion)
cv2.waitKey(0)
cv2.destroyAllWindows()

blur_conversion = cv2.GaussianBlur(gray_conversion, (5,5), 0)
cv2.imshow('input_image', blur_conversion)
cv2.waitKey(0)
cv2.destroyAllWindows()

canny_conversion = cv2.Canny(blur_conversion, 50,155)
cv2.imshow('input_image', canny_conversion)
cv2.waitKey(0)
cv2.destroyAllWindows()

#Masking Region of Interest
import cv2
import numpy as np
import matplotlib.pyplot as plt

def canny_edge(image):
          gray_conversion= cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
          blur_conversion = cv2.GaussianBlur(gray_conversion, (5,5),0)
          canny_conversion = cv2.Canny(blur_conversion, 50,150)
          return canny_conversion
      
def reg_of_interest(image):
        Image_height = image.shape[0]
        polygons = np.array([[(200, Image_height), (1100, Image_height), (550, 250)]])
        image_mask = np.zeros_like(image)
        cv2.fillPoly(image_mask, polygons, 255)
        return image_mask
    
canny_conversion = canny_edge(lanelines_image)
cv2.imshow('result', reg_of_interest(canny_conversion))
cv2.waitKey(0)
cv2.destroyAllWindows()

# Applying bitwise_and

import cv2
import numpy as np
import matplotlib.pyplot as plt

def canny_edge(image):
         gray_conversion= cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
         blur_conversion = cv2.GaussianBlur(gray_conversion, (5,5),0)
         canny_conversion = cv2.Canny(blur_conversion, 50,150)
         return canny_conversion
     
def reg_of_interest(image):
         image_height = image.shape[0]
         polygons = np.array([[(200, image_height), (1100, image_height), (551, 250)]])
         image_mask = np.zeros_like(image)
         cv2.fillPoly(image_mask, polygons, 255)
         masking_image = cv2.bitwise_and(image,image_mask)
         return masking_image
     
canny_conversion = canny_edge(lanelines_image)
cropped_image = reg_of_interest(canny_conversion)
cv2.imshow('result', cropped_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

import cv2
import numpy as np
import matplotlib.pyplot as plt

def canny_edge(image):
         gray_conversion= cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
         blur_conversion = cv2.GaussianBlur(gray_conversion, (5,5),0)
         canny_conversion = cv2.Canny(blur_conversion, 50,150)
         return canny_conversion
     
def reg_of_interest(image):
         image_height = image.shape[0]
         polygons = np.array([[(200, image_height), (1100, image_height), (551, 250)]])
         image_mask = np.zeros_like(image)
         cv2.fillPoly(image_mask, polygons, 255)
         masking_image = cv2.bitwise_and(image,image_mask)
         return masking_image
     
def show_lines(image, lines):
            lines_image = np.zeros_like(image)
            if lines is not None:
                for line in lines:
                    X1, Y1, X2, Y2 = line.reshape(4)
                    cv2.line(lines_image, (X1, Y1), (X2, Y2), (255,0,0), 10)
            return lines_image

canny_conv = canny_edge(lanelines_image)
cropped_image = reg_of_interest(canny_conv)
lane_lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength= 40, maxLineGap=5)
linelines_image = show_lines(lanelines_image, lane_lines)
cv2.imshow('result', linelines_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

#Combining with actual image

image = cv2.imread('C:/Users/Bharani Kumar/Desktop/AI/SDCs/test_image_Lane.jpg')
lane_image = np.copy(image)
canny = canny_edge(lane_image)
cropped_image = reg_of_interest(canny)
lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength= 40, maxLineGap=5)
line_image = show_lines(lane_image, lines)
combine_image = cv2.addWeighted(lane_image, 0.8, line_image, 1, 1)

cv2.imshow('result', combine_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

#Detect road marking in images
import cv2
import numpy as np
import matplotlib.pyplot as plt

def make_coordinates(image, line_parameters):
          slope, intercept = line_parameters
          y1 = image.shape[0]
          y2 = int(y1*(3/5))
          x1 = int((y1- intercept)/slope)
          x2 = int((y2 - intercept)/slope)
          return np.array([x1, y1, x2, y2])
      
def average_slope_intercept(image, lines):
          left_fit = []
          right_fit = []
          for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            parameter = np.polyfit((x1, x2), (y1, y2), 1)
            slope = parameter[0]
            intercept = parameter[1]
            if slope < 0:
              left_fit.append((slope, intercept))
            else:
              right_fit.append((slope, intercept))
          left_fit_average =np.average(left_fit, axis=0)
          right_fit_average = np.average(right_fit, axis =0)
          left_line =make_coordinates(image, left_fit_average)
          right_line = make_coordinates(image, right_fit_average)
  
          return np.array([left_line, right_line])
      
def canny_edge(image):
         gray_coversion= cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
         blur_conversion = cv2.GaussianBlur(gray_conversion, (5,5),0)
         canny_conversion = cv2.Canny(blur_conversion, 50,150)
         return canny_conversion
     
def show_lines(image, lines):
          lanelines_image = np.zeros_like(image)
          if lines is not None:
            for line in lines:
              X1, Y1, X2, Y2 = line.reshape(4)
              cv2.line(lanelines_image, (X1, Y1), (X2, Y2), (255,0,0), 10)
          return lanelines_image
      
def reg_of_interest(image):
          image_height = image.shape[0]
          polygons = np.array([[(200, image_height), (1100, image_height), (551, 250)]])
          image_mask = np.zeros_like(image)
          cv2.fillPoly(image_mask, polygons, 255)
          masking_image = cv2.bitwise_and(image,image_mask)
          return masking_image
      
image = cv2.imread('C:/Users/Bharani Kumar/Desktop/AI/SDCs/test_image_Lane.jpg')
lanelines_image = np.copy(image)
canny_image = canny_edge(lanelines_image)
cropped_image = reg_of_interest(canny_image)
lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength= 40, maxLineGap=5)
averaged_lines = average_slope_intercept(lanelines_image, lines)
line_image = show_lines(lanelines_image, averaged_lines)
combine_image = cv2.addWeighted(lanelines_image, 0.8, line_image, 1, 1)
cv2.imshow('result', combine_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

#Detecting Road Markings in Video
import cv2
import numpy as np
import matplotlib.pyplot as plt

def make_coordinates(image, line_parameters):
          try:
            slope, intercept = line_parameters
          except TypeError:
            slope, intercept = 0.001,0
          #slope, intercept = line_parameters
          y1 = image.shape[0]
          y2 = int(y1*(3/5))
          x1 = int((y1- intercept)/slope)
          x2 = int((y2 - intercept)/slope)
          return np.array([x1, y1, x2, y2])
      
def average_slope_intercept(image, lines):
          left_fit = []
          right_fit = []
          for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            parameter = np.polyfit((x1, x2), (y1, y2), 1)
            slope = parameter[0]
            intercept = parameter[1]
            if slope < 0:
              left_fit.append((slope, intercept))
            else:
              right_fit.append((slope, intercept))
          left_fit_average =np.average(left_fit, axis=0)
          right_fit_average = np.average(right_fit, axis =0)
          left_line =make_coordinates(image, left_fit_average)
          right_line = make_coordinates(image, right_fit_average)
  
          return np.array([left_line, right_line])
      
def canny_edge(image):
         gray_conversion= cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
         blur_conversion = cv2.GaussianBlur(gray_conversion, (5,5),0)
         canny_conversion = cv2.Canny(blur_conversion, 50,150)
         return canny_conversion
     
def show_lines(image, lines):
          line_image = np.zeros_like(image)
          if lines is not None:
            for line in lines:
              x1, y1, x2, y2 = line.reshape(4)
              cv2.line(line_image, (x1, y1), (x2, y2), (255,0,0), 10)
          return line_image

def reg_of_interest(image):
          image_height = image.shape[0]
          polygons = np.array([[(200, image_height), (1100, image_height), (550, 250)]])
          image_mask = np.zeros_like(image)
          cv2.fillPoly(image_mask, polygons, 255)
          masking_image = cv2.bitwise_and(image,image_mask)
          return masking_image

cap = cv2.VideoCapture("C:/Users/Bharani Kumar/Desktop/AI/SDCs/test2.mp4")
while(cap.isOpened()):
            _, frame = cap.read()
            canny_image = canny_edge(frame)
            cropped_canny = reg_of_interest(canny_image)
            lines = cv2.HoughLinesP(cropped_canny, 2, np.pi/180, 100, np.array([]), minLineLength=40,maxLineGap=5)
            averaged_lines = average_slope_intercept(frame, lines)
            line_image = show_lines(frame, averaged_lines)
            combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
            cv2.imshow("result", combo_image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
cap.release()
cv2.waitKey(0)
cv2.destroyAllWindows()