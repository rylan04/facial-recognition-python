# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 21:00:16 2019

@author: rylan
"""

import cv2

face_cascade = cv2.CascadeClassifier('facial-recognition-python/haarcascade_frontalface_default.xml')
smile_cascade = cv2.CascadeClassifier('facial-recognition-python/haarcascade_smile.xml')

def detect(grey, frame):
    
    faces = face_cascade.detectMultiScale(grey,1.3,5)
    
    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)
        roi_grey = grey[y:y+h,x:x+w]
        roi_frame = frame[y:y+h,x:x+w]
        smiles = smile_cascade.detectMultiScale(roi_grey, 1.7, 22)
        
        for (sx,sy,sw,sh) in smiles:
            cv2.rectangle(roi_frame, (sx,sy), (sx + sw, sy + sh), (0,0,255), 2)
    
    return frame

video_capture = cv2.VideoCapture(0)

while True:
    _,frame=video_capture.read()
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    canvas = detect(grey,frame)
    cv2.imshow('Video', canvas)
    
    if cv2.waitKey(1) & 0xFF == ord('q'): #If we press on Q, break loop
        break

video_capture.release() # Stop video capture
cv2.destroyAllWindows() # Delete windows
    