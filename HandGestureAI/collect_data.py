
import cv2
import mediapipe as mp
import csv
import os
import time

# Dataset file
DATA_FILE = "dataset/gestures.csv"
os.makedirs("dataset", exist_ok=True)

# Gesture labels
GESTURES = {
    "0": "FIST",
    "1": "OPEN_PALM",
    "2": "VICTORY",
    "3": "POINTING",
    "4": "THUMBS_UP"
}

# CSV columns: 21 landmarks x 3 coordinates + label
columns = (
    [f"x{i}" for i in range(21)]
    + [f"y{i}" for i in range(21)]
    + [f"z{i}" for i in range(21)]
    + ["label"]
)

# Create CSV with header if needed
if not os.path.exists(DATA_FILE) or os.path.getsize(DATA_FILE) == 0:
    with open(DATA_FILE, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(columns)

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

# Start webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Webcam open nahi hua!")
    hands.close()
    raise SystemExit

selected_gesture = None
recording = False
sample_count = 0
last_save_time = 0

print("\n===== HAND GESTURE DATA COLLECTION =====")
print("0 = FIST")
print("1 = OPEN_PALM")
print("2 = VICTORY")
print("3 = POINTING")
print("4 = THUMBS_UP")
print("R = Start/Stop recording")
print("Q = Quit\n")

try:
    while True:
        success, frame = cap.read()

        if not success:
            print("Camera frame read nahi ho raha.")
            break

        frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        landmarks_data = None

        if results.multi_hand_landmarks:
            hand = results.multi_hand_landmarks[0]

            mp_draw.draw_landmarks(
                frame,
                hand,
                mp_hands.HAND_CONNECTIONS
            )

            points = [
                (lm.x, lm.y, lm.z)
                for lm in hand.landmark
            ]

            # Normalize coordinates relative to wrist
            wrist_x, wrist_y, wrist_z = points[0]

            points = [
                (x - wrist_x, y - wrist_y, z - wrist_z)
                for x, y, z in points
            ]

            landmarks_data = (
                [p[0] for p in points]
                + [p[1] for p in points]
                + [p[2] for p in points]
            )

        # Automatically save samples while recording
        current_time = time.time()

        if (
            recording
            and selected_gesture is not None
            and landmarks_data is not None
            and current_time - last_save_time >= 0.15
        ):
            with open(DATA_FILE, "a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(landmarks_data + [selected_gesture])

            sample_count += 1
            last_save_time = current_time

        # Display instructions
        status = "RECORDING" if recording else "PAUSED"

        cv2.putText(
            frame,
            f"Gesture: {selected_gesture or 'None'}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Status: {status}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255) if recording else (255, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"New samples: {sample_count}",
            (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        cv2.imshow("Hand Gesture AI - Data Collection", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        elif chr(key) in GESTURES:
            selected_gesture = GESTURES[chr(key)]
            recording = False
            print(f"\nSelected: {selected_gesture}")

        elif key == ord("r"):
            if selected_gesture is None:
                print("Pehle 0, 1, 2, 3 ya 4 se gesture select karo.")
            else:
                recording = not recording
                print(
                    f"{'Recording started' if recording else 'Recording stopped'}: "
                    f"{selected_gesture}"
                )

finally:
    cap.release()
    hands.close()
    cv2.destroyAllWindows()

print(f"\nData saved in: {DATA_FILE}")
print(f"New samples collected: {sample_count}")
