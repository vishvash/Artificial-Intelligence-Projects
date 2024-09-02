# -*- coding: utf-8 -*-
"""
Created on Wed May  8 19:47:07 2024

@author: Lenovo
"""

import cv2
face_cascade = cv2.CascadeClassifier('C:/Users/Lenovo/Downloads/Study material/AI/Computer Vision and Image Processing/Data Sets_SDC/opencv_config_files/Day 5/haarcascade_frontalface_alt.xml')
cap = cv2.VideoCapture('C:/Users/Lenovo/Downloads/Study material/AI/Computer Vision and Image Processing/Assignment/8175_pedestrians_pedestrian_footbridge_18030106BCityRoam01720p5000br.mp4')
scaling_factor = 0.5
while True:
    ret, frame = cap.read()
    if not ret:
       break
    frame = cv2.resize(frame, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
    face_rects = face_cascade.detectMultiScale(frame, scaleFactor=1.3, minNeighbors=3)
    for (x,y,w,h) in face_rects:
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 3)
    cv2.imshow('Face Detector', frame)
    c = cv2.waitKey(1)
    if c == 27:
        break
cap.release()
cv2.destroyAllWindows()