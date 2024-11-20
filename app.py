import cv2
import mediapipe as mp
import pygame
import numpy as np

# Initialize pygame for sound
pygame.mixer.init()

# Load the bell sound (replace with the path to your sound file)
pygame.mixer.music.load("bell_sound.mp3")  # Make sure to replace this with your sound file path

# Initialize MediaPipe for hand detection
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5)

# Initialize webcam
cap = cv2.VideoCapture(0)

# Placeholder for object detection (use any model, YOLO, or Haar Cascade)
# Example for using pre-trained Haar Cascade (you can replace with YOLO, TensorFlow, etc.)
object_cascade = cv2.CascadeClassifier('object_cascade.xml')  # For simplicity, change if using other models

# Define touch region for simplicity (You could replace with actual object detection logic)
object_boxes = []  # Placeholder for detected objects (replace this with actual object detection logic)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally for mirror view
    frame = cv2.flip(frame, 1)

    # Convert the frame to RGB as MediaPipe expects RGB images
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    # Detect hands using MediaPipe
    hand_found = False
    wrist_x, wrist_y = -1, -1  # Initialize wrist position

    if results.multi_hand_landmarks:
        for landmarks in results.multi_hand_landmarks:
            hand_found = True
            # Extract wrist or other relevant points
            wrist = landmarks.landmark[mp_hands.HandLandmark.WRIST]
            wrist_x = int(wrist.x * frame.shape[1])
            wrist_y = int(wrist.y * frame.shape[0])

            # Draw landmarks or a circle around wrist (just for visualization)
            cv2.circle(frame, (wrist_x, wrist_y), 5, (0, 255, 0), -1)

    # Object detection logic (example using Haar Cascade)
    # In actual use, replace this with your object detection model (YOLO, TensorFlow, etc.)
    if object_boxes:
        for (x1, y1, x2, y2) in object_boxes:
            # Draw the bounding box around detected objects
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

            # Check if the hand (wrist) is inside the object bounding box
            if hand_found and x1 < wrist_x < x2 and y1 < wrist_y < y2:
                print("Hand is touching the object!")
                # Play the bell sound when touch is detected
                pygame.mixer.music.play()

    # Display the resulting frame
    cv2.imshow('Hand and Object Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
