import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from utils import REFERENCE_IMAGES, init_mediapipe
import random

st.set_page_config(layout="wide")

# --- Setup state ---
if "game_score" not in st.session_state:
    st.session_state.game_score = 0
if "game_letter" not in st.session_state:
    st.session_state.game_letter = random.choice(list(REFERENCE_IMAGES.keys()))
if "matched" not in st.session_state:
    st.session_state.matched = False

# --- Reference Image Logic ---
ref_path = REFERENCE_IMAGES[st.session_state.game_letter]
mp_hands, hands, mp_draw = init_mediapipe()
ref_img = cv2.imread(ref_path)
ref_rgb = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB)
ref_hand_img = None

results = hands.process(ref_rgb)
if results.multi_hand_landmarks:
    for handLms in results.multi_hand_landmarks:
        x = [lm.x for lm in handLms.landmark]
        y = [lm.y for lm in handLms.landmark]
        h, w, _ = ref_rgb.shape
        x1, x2 = int(min(x)*w)-20, int(max(x)*w)+20
        y1, y2 = int(min(y)*h)-20, int(max(y)*h)+20
        ref_hand_img = ref_rgb[max(y1, 0):min(y2, h), max(x1, 0):min(x2, w)]

# --- Webcam Processing ---
class SignDetector(VideoTransformerBase):
    def __init__(self):
        self.matched = False
        self.ref = ref_hand_img

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if results.multi_hand_landmarks and self.ref is not None:
            for handLms in results.multi_hand_landmarks:
                x = [lm.x for lm in handLms.landmark]
                y = [lm.y for lm in handLms.landmark]
                h, w, _ = img.shape
                x1, x2 = int(min(x)*w)-20, int(max(x)*w)+20
                y1, y2 = int(min(y)*h)-20, int(max(y)*h)+20
                hand_img = img[max(y1,0):min(y2,h), max(x1,0):min(x2,w)]

                if hand_img.size != 0:
                    try:
                        hand_gray = cv2.cvtColor(hand_img, cv2.COLOR_BGR2GRAY)
                        ref_gray = cv2.cvtColor(self.ref, cv2.COLOR_BGR2GRAY)
                        hand_resized = cv2.resize(hand_gray, (200, 200))
                        ref_resized = cv2.resize(ref_gray, (200, 200))

                        score, _ = ssim(hand_resized, ref_resized, full=True)

                        if score > 0.4 and not st.session_state.matched:
                            st.session_state.game_score += 1
                            st.session_state.matched = True
                    except:
                        pass

        return img

# --- Layout ---
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### 📹 Live Camera Feed")
    webrtc_streamer(key="sign", video_processor_factory=SignDetector)

with col2:
    st.markdown("### 🎮 Controls & Info")

    st.markdown(f"**Sign to show:** `{st.session_state.game_letter.upper()}`")
    st.markdown(f"**Score:** `{st.session_state.game_score}`")

    if ref_hand_img is not None:
        st.image(ref_hand_img, caption="Reference Sign", width=300)
    else:
        st.image(ref_rgb, caption="Fallback: Full Reference Image", width=300)

    if st.session_state.matched:
        st.success("✅ Sign Matched!")

    colA, colB = st.columns(2)
    with colA:
        if st.button("⏭️ Next"):
            st.session_state.game_letter = random.choice(list(REFERENCE_IMAGES.keys()))
            st.session_state.matched = False
            st.rerun()
    with colB:
        if st.button("🔄 Reset Score"):
            st.session_state.game_score = 0
