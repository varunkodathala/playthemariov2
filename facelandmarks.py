import cv2
import dlib
import numpy as np 
from pynput.keyboard import Key, Controller
import time


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('/Users/varun/documents/autogame/shape_predictor_68_face_landmarks.dat')

cam = cv2.VideoCapture(0)

keyboard = Controller()

def key_w(ready):

    if ready:
        keyboard.press('w')
        time.sleep(0.4)
        keyboard.release('w')

def key_d(ready):

    if ready:
        keyboard.press('d')
        time.sleep(0.2)
        keyboard.release('d')

def key_a(ready):

    if ready:
        keyboard.press('a')
        time.sleep(0.2)
        keyboard.release('a')

while True:

    _,image = cam.read()
    frame = image.copy()
    image = cv2.flip(image,1)
    frame = cv2.flip(frame,1)
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    image = cv2.resize(image,(720,630))
    cv2.imshow('Original',image)

    for face in faces:

        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()

        landmarks = predictor(gray, face)

        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)

        xu = landmarks.part(21).x
        yu = landmarks.part(21).y

        xl = landmarks.part(39).x
        yl = landmarks.part(39).y

        xcx1 = landmarks.part(55).x
        xcy1 = landmarks.part(55).y

        xcx2 = landmarks.part(14).x
        xcy2 = landmarks.part(14).y

        xcx3 = landmarks.part(49).x
        xcy3 = landmarks.part(49).y

        xcx4 = landmarks.part(4).x
        xcy4 = landmarks.part(4).y


        try:
            # print((xcx1-xcx2)+(xcy1-xcy2))
            # print((xcx3-xcx4)+(xcy3-xcy4))

            right = (xcx1-xcx2)+(xcy1-xcy2)

            left = (xcx3-xcx4)+(xcy3-xcy4)

            jump = (xu-xl)+(yu-yl)

            if(jump<-20):

                key_w(1)

            if(left>70):

                key_a(1)
            
            if(right<-50):

                key_d(1)

        except Exception as e:
            pass

    frame = cv2.resize(frame,(720,630))
    cv2.imshow('Image',frame)
    

    if(cv2.waitKey(1) & 0xff == ord('q')):
        break

cv2.destroyAllWindows()