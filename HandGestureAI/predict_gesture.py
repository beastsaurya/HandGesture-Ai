import cv2
import mediapipe as mp
import numpy as np
import joblib
import os

MODEL_FILE = "models/gesture_model.pkl"

if not os.path.exists(MODEL_FILE):
    raise FileNotFoundError(
        "Model nahi mila. Pehle train_model.py run karo."
    )

model = joblib.load(MODEL_FILE)

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    raise RuntimeError("Webcam open nahi ho raha hai.")

print("Real-time gesture recognition started.")
print("Press Q to quit.")

while True:
    success, frame = camera.read()

    if not success:
        print("Camera frame read nahi hua.")
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    gesture_text = "No Hand Detected"

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            landmarks = hand_landmarks.landmark

            # Same feature extraction as data collection
            wrist = landmarks[0]

            x_values = [point.x - wrist.x for point in landmarks]
            y_values = [point.y - wrist.y for point in landmarks]
            z_values = [point.z - wrist.z for point in landmarks]

            features = np.array(
                x_values + y_values + z_values,
                dtype=float
            ).reshape(1, -1)

            prediction = model.predict(features)[0]

            # Prediction confidence, when supported by the model
            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(features)[0]
                confidence = float(np.max(probabilities)) * 100
                gesture_text = f"{prediction}: {confidence:.1f}%"
            else:
                gesture_text = str(prediction)

            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

    cv2.putText(
        frame,
        gesture_text,
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
        cv2.LINE_AA
    )

    cv2.imshow("Hand Gesture Recognition AI", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
hands.close()
cv2.destroyAllWindows()