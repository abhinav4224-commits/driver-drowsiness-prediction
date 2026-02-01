import cv2
import mediapipe as mp
import numpy as np
import math

mp_face_mesh = mp.solutions.face_mesh

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

def eye_aspect_ratio(eye, landmarks):
    p1 = landmarks[eye[1]]
    p2 = landmarks[eye[5]]
    p3 = landmarks[eye[2]]
    p4 = landmarks[eye[4]]
    p5 = landmarks[eye[0]]
    p6 = landmarks[eye[3]]

    vertical1 = np.linalg.norm(p1 - p2)
    vertical2 = np.linalg.norm(p3 - p4)
    horizontal = np.linalg.norm(p5 - p6)

    ear = (vertical1 + vertical2) / (2.0 * horizontal)
    return ear

def analyze_drowsiness(image):
    h, w, _ = image.shape
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True)

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)

    if not result.multi_face_landmarks:
        return "No Face Detected"

    landmarks = result.multi_face_landmarks[0]
    points = np.array(
        [(int(l.x * w), int(l.y * h)) for l in landmarks.landmark]
    )

    left_ear = eye_aspect_ratio(LEFT_EYE, points)
    right_ear = eye_aspect_ratio(RIGHT_EYE, points)

    ear = (left_ear + right_ear) / 2.0

    if ear < 0.25:
        return "DROWSY"
    else:
        return "AWAKE"
