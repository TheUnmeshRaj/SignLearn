import tempfile

import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
import torch
from PIL import Image
from torchvision import models, transforms

st.set_page_config(page_title="ISL Learning App", layout="wide")


MODEL_CONFIGS = {
    "A–F": ("models/model_af.pth", ['a', 'b', 'c', 'd', 'e', 'f']),
    "G–L": ("models/model_gl.pth", ['g', 'h', 'i', 'j', 'k', 'l']),
    "M–S": ("models/model_ms.pth", ['m', 'n', 'o', 'p', 'q', 'r', 's']),
    "T–Z": ("models/model_tz.pth", ['t', 'u', 'v', 'w', 'x', 'y', 'z']),
}

REFERENCE_IMAGES = {
    'a': 'ref_imgs/a.jpg',
    'b': 'ref_imgs/b.jpg',
    'c': 'ref_imgs/c.jpg',
    'd': 'ref_imgs/d.jpg',
    'e': 'ref_imgs/e.jpg',
    'f': 'ref_imgs/f.jpg',
    'g': 'ref_imgs/g.jpg',
    'h': 'ref_imgs/h.jpg',
    'i': 'ref_imgs/i.jpg',
    'j': 'ref_imgs/j.jpg',
    'k': 'ref_imgs/k.jpg',
    'l': 'ref_imgs/l.jpg',
    'm': 'ref_imgs/m.jpg',
    'n': 'ref_imgs/n.jpg',
    'o': 'ref_imgs/o.jpg',
    'p': 'ref_imgs/p.jpg',
    'q': 'ref_imgs/q.jpg',
    'r': 'ref_imgs/r.jpg',
    's': 'ref_imgs/s.jpg',
    't': 'ref_imgs/t.jpg',
    'u': 'ref_imgs/u.jpg',
    'v': 'ref_imgs/v.jpg',
    'w': 'ref_imgs/w.jpg',
    'x': 'ref_imgs/x.jpg',
    'y': 'ref_imgs/y.jpg',
    'z': 'ref_imgs/z.jpg'
}

device = torch.device("cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

@st.cache_resource
def load_model(model_path, num_classes):
    model = models.resnet18(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

# Mediapipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# ==== UI ====
st.title("🇮🇳 Indian Sign Language Learning App")
tabs = st.tabs(["📚 Learn Alphabets", "🗣️ Daily Useful Phrases (Coming Soon)"])

with tabs[0]:
    st.header("Learn ISL Alphabets")
    col1, col2 = st.columns([1, 2])

    with col1:
        level = st.radio("Choose Level", list(MODEL_CONFIGS.keys()))
        model_path, class_names = MODEL_CONFIGS[level]
        model = load_model(model_path, len(class_names))

        start = st.button("Start Webcam")

    with col2:
        stframe = st.empty()
        ref_frame = st.empty()

    if start:
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            if results.multi_hand_landmarks:
                for handLms in results.multi_hand_landmarks:
                    x_list = [lm.x for lm in handLms.landmark]
                    y_list = [lm.y for lm in handLms.landmark]
                    xmin, xmax = int(min(x_list)*w), int(max(x_list)*w)
                    ymin, ymax = int(min(y_list)*h), int(max(y_list)*h)
                    x1, y1 = max(xmin - 20, 0), max(ymin - 20, 0)
                    x2, y2 = min(xmax + 20, w), min(ymax + 20, h)

                    hand_img = frame[y1:y2, x1:x2]
                    if hand_img.size != 0:
                        img = cv2.cvtColor(hand_img, cv2.COLOR_BGR2RGB)
                        img = Image.fromarray(img)
                        img = transform(img).unsqueeze(0).to(device)

                        with torch.no_grad():
                            output = model(img)
                            _, pred = torch.max(output, 1)
                            label = class_names[pred.item()]

                        cv2.putText(frame, label, (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                        if label in REFERENCE_IMAGES:
                            ref_frame.image(REFERENCE_IMAGES[label], caption=f"Reference for '{label.upper()}'")

                    mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

            stframe.image(frame, channels="BGR")

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()

with tabs[1]:
    st.header("🗣️ Daily Use Phrases")
    st.info("Coming Soon: Learn signs for 'Hello', 'Thank you', 'Sorry', and more with video examples and practice mode.")
