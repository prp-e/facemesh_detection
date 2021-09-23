import cv2
import mediapipe as mp
import time 

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh


image = cv2.imread('scarlett-johansson.jpg')

with mp_face_mesh.FaceMesh(min_detection_confidence=.6) as face_mesh:
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False 

    results = face_mesh.process(image)

    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            #mp_drawing.draw_landmarks(image=image, landmark_list=face_landmarks, connections=mp_face_mesh.FACEMESH_CONTOURS, landmark_drawing_spec=None)

            for id, landmark in enumerate(face_landmarks.landmark):
                h, w, c = image.shape
                cx, cy = int(landmark.x * w), int(landmark.y * h)
                if id == 29:
                    cv2.circle(image, (cx, cy), 5, (0, 0, 255), -1)



cv2.imshow('Static Image', image)
cv2.waitKey(0)