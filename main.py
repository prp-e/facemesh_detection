import cv2 
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

camera = cv2.VideoCapture(1)

with mp_face_mesh.FaceMesh(min_detection_confidence=.6) as face_mesh:
    while camera.isOpened():
        _, frame = camera.read()

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame.flags.writeable = False

        results = face_mesh.process(frame)

        if results.multi_face_landmarks:
            print(results)

        frame.flags.writeable = True
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        cv2.imshow('Face Mesh Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

camera.release()
cv2.destroyAllWindows()