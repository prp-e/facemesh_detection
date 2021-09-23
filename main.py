import cv2 
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

camera = cv2.VideoCapture(1)

with mp_face_mesh.FaceMesh(min_detection_confidence=.6) as face_mesh:
    while camera.isOpened():
        _, frame = camera.read()

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame.flags.writeable = False

        results = face_mesh.process(frame)

        frame.flags.writeable = True
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                for id, landmark in enumerate(face_landmarks.landmark):
                    h, w, c = frame.shape
                    cx, cy = int(landmark.x * w), int(landmark.y * h)
                    if id == 1:
                        cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
                    if id == 2:
                        cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
                    if id == 4:
                        cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
                    if id == 129:
                        cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
                    if id == 358:
                        cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

        
        
        cv2.imshow('Face Mesh Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

camera.release()
cv2.destroyAllWindows()