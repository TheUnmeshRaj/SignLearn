import os
import cv2
import mediapipe as mp

DATA_DIR = 'test'
os.makedirs(DATA_DIR, exist_ok=True)

number_of_classes = 5
dataset_size = 100
CROP_SIZE = 250

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

for j in range(number_of_classes):
    class_name = chr(ord('a') + j)
    class_path = os.path.join(DATA_DIR, class_name)
    os.makedirs(class_path, exist_ok=True)

    print('Collecting data for class {}'.format(class_name))

    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        x1, y1 = int(w * (2/3)), 0
        x2, y2 = w, h
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, 'Ready? Press "Q"', (70, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 2)
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        roi = frame[:, int(w * (2/3)):]

        rgb_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_roi)

        if results.multi_hand_landmarks:
            roi_clean = roi.copy()

            for handLms in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(roi, handLms, mp_hands.HAND_CONNECTIONS)

            cv2.imwrite(os.path.join(class_path, '{}.jpg'.format(counter)), roi)
            counter += 1
            print(f"[{class_name}] Captured {counter}/{dataset_size}")

        cv2.imshow('ROI', roi)
        if cv2.waitKey(25) & 0xFF == 27:
            break

    print('Collected {} images for class {}'.format(counter, class_name))

cap.release()
cv2.destroyAllWindows()