# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 01:10:25 2019

@author: rylan
"""

# Import libraries
import cv2

# Loading cascades
face_cascade = cv2.CascadeClassifier('facial-recognition-python/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('facial-recognition-python/haarcascade_eye.xml')

# Defining a function that will do detections
def detect(gray, frame):
    
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255,0,0), 2 )
        roi_gray = gray[y:y+h, x:x+w] # Region of interest
        roi_color = frame[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 3)
        
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2 )
    
    return frame

# Face recognition with the webcam
video_capture = cv2.VideoCapture(0)
while True:
    _, frame = video_capture.read() # Read in the video as a picture
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Convert to greyscale
    canvas = detect(gray, frame) # Apply the detect function (draws rectangles)
    cv2.imshow('Video', canvas)
    if cv2.waitKey(1) & 0xFF == ord('q'): #If we press on Q, break loop
        break

video_capture.release() # Stop video capture
cv2.destroyAllWindows() # Delete windows
    
# Wow! really cool!



