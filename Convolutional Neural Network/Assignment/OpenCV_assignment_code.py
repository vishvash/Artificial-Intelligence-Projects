import cv2
img = cv2.imread('C:/Users/Lenovo/Downloads/Study material/AI/Convolutional Neural Network/Assignment/000001.jpg')
num_rows, num_cols = img.shape[:2]
rotation_matrix = cv2.getRotationMatrix2D((num_cols/2, num_rows/2), 30, 0.7)
img_rotation = cv2.warpAffine(img, rotation_matrix, (num_cols, num_rows))
cv2.imshow('Rotation', img_rotation)
cv2.waitKey(0)


import cv2
img = cv2.imread('C:/Users/Lenovo/Downloads/Study material/AI/Convolutional Neural Network/Assignment/000456.jpg')
cv2.imshow('Original Image', img)
img_scaled = cv2.resize(img,None,fx=1.2, fy=1.2, interpolation =cv2.INTER_LINEAR)
cv2.imshow('Scaling - Linear Interpolation', img_scaled)
img_scaled = cv2.resize(img,None,fx=1.2, fy=1.2, interpolation = cv2.INTER_CUBIC)
cv2.imshow('Scaling - Cubic Interpolation', img_scaled)
img_scaled = cv2.resize(img,(450, 400), interpolation = cv2.INTER_AREA)
cv2.imshow('Scaling - Skewed Size', img_scaled)
cv2.waitKey(0)

import cv2

# Load the image
img = cv2.imread('C:/Users/Lenovo/Downloads/Study material/AI/Convolutional Neural Network/Assignment/004545.jpg')

# Perform width shift
width_shift = 50
height, width = img.shape[:2]
img_width_shift = img[:, width_shift:]  # Shift to the right
img_width_shift = cv2.resize(img_width_shift, (width, height))

# Perform height shift
height_shift = 50
img_height_shift = img[height_shift:, :]  # Shift downwards
img_height_shift = cv2.resize(img_height_shift, (width, height))

# Perform horizontal flip
img_horizontal_flip = cv2.flip(img, 1)  # Flip around y-axis

# Display the original and augmented images
cv2.imshow('Original', img)
cv2.imshow('Width Shift', img_width_shift)
cv2.imshow('Height Shift', img_height_shift)
cv2.imshow('Horizontal Flip', img_horizontal_flip)

cv2.waitKey(0)
cv2.destroyAllWindows()
