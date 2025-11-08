#pylint:disable=no-member

import numpy as np
import cv2 as cv
import os


haar_cascade = cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_frontalface_default.xml")

if haar_cascade.empty():
    raise FileNotFoundError("Haar cascade not loaded!")


people = ['Ben Afflek', 'Elton John', 'Jerry Seinfield', 'Madonna', 'Mindy Kaling']


model_path = r"D:\SAI\opencv\Faces\face_trained.yml"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Trained model not found at {model_path}")

face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read(model_path)


img_path = r"D:\SAI\opencv\Resources\Faces\val\elton_john\1.jpg"
if not os.path.exists(img_path):
    raise FileNotFoundError(f"Image not found at {img_path}")

img = cv.imread(img_path)
if img is None:
    raise FileNotFoundError(f"Could not load image properly at {img_path}")

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Person', gray)

# ✅ Detect face(s)
faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
if len(faces_rect) == 0:
    print("No faces detected in this image.")

for (x, y, w, h) in faces_rect:
    faces_roi = gray[y:y + h, x:x + w]

    label, confidence = face_recognizer.predict(faces_roi)
    print(f"Label = {people[label]} | Confidence = {confidence:.2f}")

    cv.putText(img, str(people[label]), (x, y - 10),
               cv.FONT_HERSHEY_COMPLEX, 1.0, (0, 255, 0), 2)
    cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

cv.imshow('Detected Face', img)
cv.waitKey(0)
cv.destroyAllWindows()

