import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

image = cv2.imread('Lenna.jpg')

cv2.imshow('Static Image', image)
cv2.waitKey(0)