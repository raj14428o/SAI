

import os
import cv2 as cv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
img_path = os.path.join(BASE_DIR, "..", "Resources", "Photos", "group 1.jpg")
cascade_path = os.path.join(BASE_DIR, "haar_face.xml")

img = cv.imread(img_path)
if img is None:
    raise FileNotFoundError(f"Could not load image at {img_path}")

cv.imshow('Group of 5 people', img)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Gray People', gray)

haar_cascade = cv.CascadeClassifier(cascade_path)
if haar_cascade.empty():
    raise FileNotFoundError(f"Could not load cascade at {cascade_path}")


faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=1)

print(f'Number of faces found = {len(faces_rect)}')

for (x,y,w,h) in faces_rect:
    cv.rectangle(img, (x,y), (x+w,y+h), (0,255,0), thickness=2)

cv.imshow('Detected Faces', img)



cv.waitKey(0)