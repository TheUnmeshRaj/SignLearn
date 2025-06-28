
import tempfile
import time
import json
import os
from datetime import datetime, timedelta
import threading

import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
import torch
from PIL import Image
from torchvision import models, transforms
import plotly.express as px
import plotly.graph_objects as go

# Page Configuration
st.set_page_config(
    page_title="ISL Learning Hub",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    
    .feature-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        border: 1px solid #e1e8ed;
        margin-bottom: 1rem;
        transition: all 0.3s ease;
    }
    
    .stats-container {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    
    .webcam-container {
        border: 3px solid #667eea;
        border-radius: 15px;
        padding: 1rem;
        background: #f8f9ff;
    }
    
    .reference-container {
        border: 2px solid #764ba2;
        border-radius: 10px;
        padding: 15px;
        background: white;
        text-align: center;
        margin-top: 1rem;
    }
    
    .prediction-box {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        font-size: 1.2rem;
        font-weight: bold;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Model Configurations (WORKING PATHS)
MODEL_CONFIGS = {
    "A–F": ("models/model_af.pth", ['a', 'b', 'c', 'd', 'e', 'f']),
    "G–L": ("models/model_gl.pth", ['g', 'h', 'i', 'j', 'k', 'l']),
    "M–S": ("models/model_ms.pth", ['m', 'n', 'o', 'p', 'q', 'r', 's']),
    "T–Z": ("models/model_tz.pth", ['t', 'u', 'v', 'w', 'x', 'y', 'z']),
}

# Reference Images (WORKING PATHS)
REFERENCE_IMAGES = {
    'a': 'ref_imgs/a.jpg', 'b': 'ref_imgs/b.jpg', 'c': 'ref_imgs/c.jpg',
    'd': 'ref_imgs/d.jpg', 'e': 'ref_imgs/e.jpg', 'f': 'ref_imgs/f.jpg',
    'g': 'ref_imgs/g.jpg', 'h': 'ref_imgs/h.jpg', 'i': 'ref_imgs/i.jpg',
    'j': 'ref_imgs/j.jpg', 'k': 'ref_imgs/k.jpg', 'l': 'ref_imgs/l.jpg',
    'm': 'ref_imgs/m.jpg', 'n': 'ref_imgs/n.jpg', 'o': 'ref_imgs/o.jpg',
    'p': 'ref_imgs/p.jpg', 'q': 'ref_imgs/q.jpg', 'r': 'ref_imgs/r.jpg',
    's': 'ref_imgs/s.jpg', 't': 'ref_imgs/t.jpg', 'u': 'ref_imgs/u.jpg',
    'v': 'ref_imgs/v.jpg', 'w': 'ref_imgs/w.jpg', 'x': 'ref_imgs/x.jpg',
    'y': 'ref_imgs/y.jpg', 'z': 'ref_imgs/z.jpg'
}

# Initialize session state for tracking
if 'user_progress' not in st.session_state:
    st.session_state.user_progress = {
        'learned_letters': set(),
        'practice_sessions': 0,
        'total_time': 0,
        'streak_days': 1
    }

if 'webcam_running' not in st.session_state:
    st.session_state.webcam_running = False

if 'current_prediction' not in st.session_state:
    st.session_state.current_prediction = ""

if 'current_confidence' not in st.session_state:
    st.session_state.current_confidence = 0.0

# Device and Transform Setup
device = torch.device("cpu")
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

@st.cache_resource
def load_model(model_path, num_classes):
    """Load and cache the PyTorch model"""
    model = models.resnet18(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        st.success(f"✅ Model loaded successfully: {model_path}")
    except FileNotFoundError:
        st.error(f"❌ Model file not found: {model_path}")
        st.info("Please ensure your model files are in the 'models/' directory")
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
    
    model.to(device)
    model.eval()
    return model

# MediaPipe setup
@st.cache_resource
def init_mediapipe():
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    )
    mp_draw = mp.solutions.drawing_utils
    return mp_hands, hands, mp_draw

# Header
st.markdown("""
<div class="main-header">
    <h1>🤟 Indian Sign Language Learning Hub</h1>
    <p>Master ISL with AI-powered recognition and interactive learning</p>
</div>
""", unsafe_allow_html=True)

# Sidebar - User Dashboard
with st.sidebar:
    st.markdown("### 👤 Your Learning Dashboard")
    
    learned_count = len(st.session_state.user_progress['learned_letters'])
    progress_percentage = (learned_count / 26) * 100
    
    st.markdown(f"""
    <div class="stats-container">
        <h3>📊 Your Progress</h3>
        <p><strong>{learned_count}/26</strong> Letters Learned</p>
        <p><strong>{progress_percentage:.1f}%</strong> Complete</p>
        <p><strong>{st.session_state.user_progress['practice_sessions']}</strong> Sessions</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.progress(progress_percentage / 100)
    
    if learned_count > 0:
        st.markdown("### 🎯 Recently Learned")
        recent_letters = list(st.session_state.user_progress['learned_letters'])[-5:]
        st.write(" • ".join([l.upper() for l in recent_letters]))

# Main Content Tabs
tabs = st.tabs([
    "🎯 Live Practice",
    "📚 Alphabet Reference", 
    "📊 Progress Analytics",
    "ℹ️ About ISL"
])

# Tab 1: Live Practice (WORKING VERSION)
with tabs[0]:
    st.header("🎯 AI-Powered Sign Recognition")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        st.subheader("📖 Learning Settings")
        
        # Level Selection
        level = st.radio("Choose Level", list(MODEL_CONFIGS.keys()))
        model_path, class_names = MODEL_CONFIGS[level]
        
        # Load model
        model = load_model(model_path, len(class_names))
        
        st.info(f"**Current Level:** {level}")
        st.write(f"**Letters:** {', '.join([c.upper() for c in class_names])}")
        
        # Settings
        st.subheader("⚙️ Detection Settings")
        confidence_threshold = st.slider("Confidence Threshold", 0.5, 0.95, 0.7)
        show_landmarks = st.checkbox("Show Hand Landmarks", True)
        
        # Control Buttons
        start_button = st.button("🚀 Start Webcam", type="primary")
        stop_button = st.button("⏹️ Stop Webcam")
        
        if start_button:
            st.session_state.webcam_running = True
            st.session_state.user_progress['practice_sessions'] += 1
            
        if stop_button:
            st.session_state.webcam_running = False
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Current Prediction Display
        if st.session_state.current_prediction:
            st.markdown(f"""
            <div class="prediction-box">
                🎯 Detected: {st.session_state.current_prediction.upper()}<br>
                Confidence: {st.session_state.current_confidence:.1%}
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="webcam-container">', unsafe_allow_html=True)
        
        # Webcam display
        stframe = st.empty()
        
        # Reference image display
        ref_placeholder = st.empty()
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Status display
        status_placeholder = st.empty()
    
    # WORKING WEBCAM LOGIC
    if st.session_state.webcam_running:
        mp_hands, hands, mp_draw = init_mediapipe()
        
        # Try to open webcam
        try:
            cap = cv2.VideoCapture(0)
            
            if not cap.isOpened():
                st.error("❌ Cannot access webcam. Please check your camera permissions.")
                st.session_state.webcam_running = False
            else:
                status_placeholder.success("🎥 Webcam active - Show your hand signs!")
                
                # Main webcam loop
                frame_count = 0
                while st.session_state.webcam_running:
                    ret, frame = cap.read()
                    if not ret:
                        st.error("❌ Failed to read from webcam")
                        break
                    
                    # Flip frame horizontally for mirror effect
                    frame = cv2.flip(frame, 1)
                    h, w, _ = frame.shape
                    
                    # Convert BGR to RGB for MediaPipe
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = hands.process(rgb_frame)
                    
                    # Process hand landmarks
                    if results.multi_hand_landmarks:
                        for hand_landmarks in results.multi_hand_landmarks:
                            # Extract hand region coordinates
                            x_coords = [lm.x for lm in hand_landmarks.landmark]
                            y_coords = [lm.y for lm in hand_landmarks.landmark]
                            
                            x_min, x_max = int(min(x_coords) * w), int(max(x_coords) * w)
                            y_min, y_max = int(min(y_coords) * h), int(max(y_coords) * h)
                            
                            # Add padding
                            x1 = max(x_min - 20, 0)
                            y1 = max(y_min - 20, 0)
                            x2 = min(x_max + 20, w)
                            y2 = min(y_max + 20, h)
                            
                            # Extract hand region
                            hand_img = frame[y1:y2, x1:x2]
                            
                            if hand_img.size != 0:
                                # Prepare image for model
                                img_rgb = cv2.cvtColor(hand_img, cv2.COLOR_BGR2RGB)
                                pil_img = Image.fromarray(img_rgb)
                                input_tensor = transform(pil_img).unsqueeze(0).to(device)
                                
                                # Make prediction
                                with torch.no_grad():
                                    output = model(input_tensor)
                                    probabilities = torch.nn.functional.softmax(output, dim=1)
                                    confidence, pred_idx = torch.max(probabilities, 1)
                                    
                                    confidence_score = confidence.item()
                                    predicted_letter = class_names[pred_idx.item()]
                                    
                                    # Update session state
                                    if confidence_score > confidence_threshold:
                                        st.session_state.current_prediction = predicted_letter
                                        st.session_state.current_confidence = confidence_score
                                        
                                        # Add to learned letters
                                        st.session_state.user_progress['learned_letters'].add(predicted_letter)
                                        
                                        # Draw prediction on frame
                                        cv2.putText(frame, f"{predicted_letter.upper()} ({confidence_score:.2f})", 
                                                  (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                                                  1, (0, 255, 0), 2)
                                        
                                        # Show reference image
                                        if predicted_letter in REFERENCE_IMAGES:
                                            try:
                                                ref_placeholder.image(
                                                    REFERENCE_IMAGES[predicted_letter], 
                                                    caption=f"Reference: {predicted_letter.upper()}",
                                                    width=200
                                                )
                                            except:
                                                ref_placeholder.info(f"Reference image for '{predicted_letter.upper()}' not found")
                                    
                                    # Draw bounding box
                                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                            
                            # Draw hand landmarks
                            if show_landmarks:
                                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    
                    # Display frame
                    stframe.image(frame, channels="BGR", use_column_width=True)
                    
                    # Control frame rate
                    frame_count += 1
                    if frame_count % 30 == 0:  # Update every 30 frames
                        time.sleep(0.1)
                    
                    # Check if stop button was pressed
                    if not st.session_state.webcam_running:
                        break
                
                cap.release()
                status_placeholder.info("📷 Webcam stopped")
        
        except Exception as e:
            st.error(f"❌ Webcam error: {str(e)}")
            st.session_state.webcam_running = False
    
    else:
        stframe.info("👆 Click 'Start Webcam' to begin sign recognition")

# Tab 2: Alphabet Reference
with tabs[1]:
    st.header("📚 Complete ISL Alphabet Reference")
    
    # Create alphabet grid
    cols = st.columns(6)
    alphabet = 'abcdefghijklmnopqrstuvwxyz'
    
    for i, letter in enumerate(alphabet):
        with cols[i % 6]:
            if st.button(f"{letter.upper()}", key=f"ref_{letter}"):
                st.session_state.selected_letter = letter
    
    # Display selected letter
    if 'selected_letter' in st.session_state:
        letter = st.session_state.selected_letter
        col1, col2 = st.columns([1, 1])
        
        with col1:
            if letter in REFERENCE_IMAGES:
                try:
                    st.image(REFERENCE_IMAGES[letter], caption=f"Letter: {letter.upper()}", width=300)
                except:
                    st.error(f"Reference image for '{letter.upper()}' not found")
            else:
                st.info(f"Reference image for '{letter.upper()}' not available")
        
        with col2:
            st.markdown(f"""
            ### Letter: {letter.upper()}
            
            **Practice Tips:**
            - Keep your hand steady and well-lit
            - Maintain consistent finger positioning
            - Practice the motion slowly first
            - Hold the sign for 2-3 seconds
            
            **Common Mistakes:**
            - Finger positioning too loose
            - Hand too close/far from camera
            - Poor lighting conditions
            """)

# Tab 3: Progress Analytics
with tabs[2]:
    st.header("📊 Your Learning Analytics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Learning Progress
        learned_letters = list(st.session_state.user_progress['learned_letters'])
        progress_data = {
            'Category': ['Learned', 'Remaining'],
            'Count': [len(learned_letters), 26 - len(learned_letters)]
        }
        
        fig = px.pie(progress_data, values='Count', names='Category', 
                    title="📈 Alphabet Learning Progress")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Session Stats
        st.markdown(f"""
        ### 📊 Your Statistics
        
        - **Letters Learned:** {len(learned_letters)}/26
        - **Completion:** {(len(learned_letters)/26)*100:.1f}%
        - **Practice Sessions:** {st.session_state.user_progress['practice_sessions']}
        - **Learning Streak:** {st.session_state.user_progress['streak_days']} days
        
        ### 🎯 Recently Learned Letters
        {' • '.join([l.upper() for l in learned_letters[-10:]]) if learned_letters else 'None yet'}
        """)

# Tab 4: About ISL
with tabs[3]:
    st.header("ℹ️ About Indian Sign Language")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### 🇮🇳 What is Indian Sign Language?
        
        Indian Sign Language (ISL) is the visual-spatial language used by the deaf community in India. 
        It's a complete, natural language with its own grammar and syntax.
        
        ### 🎯 Why Learn ISL?
        
        - **Inclusion**: Communicate with the deaf and hard-of-hearing community
        - **Cultural Awareness**: Understand deaf culture and perspectives  
        - **Professional Growth**: Career opportunities in interpretation
        - **Personal Development**: Enhance visual-spatial thinking
        
        ### 📚 Learning Tips
        
        1. **Practice Regularly**: Consistency builds muscle memory
        2. **Use Good Lighting**: Ensure your hands are clearly visible
        3. **Maintain Steady Position**: Keep consistent distance from camera
        4. **Be Patient**: Language learning takes time and practice
        """)
    
    with col2:
        st.markdown("""
        ### 🛠️ Technical Requirements
        
        **For Best Results:**
        - 📷 Working webcam
        - 💡 Good lighting
        - 🖥️ Stable internet connection
        - 📁 Model files in `/models/` directory
        - 🖼️ Reference images in `/ref_imgs/` directory
        
        ### 📁 Required Files
        
        **Model Files:**
        - `models/model_af.pth`
        - `models/model_gl.pth` 
        - `models/model_ms.pth`
        - `models/model_tz.pth`
        
        **Reference Images:**
        - `ref_imgs/a.jpg` to `ref_imgs/z.jpg`
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 1rem; background: #f8f9ff; border-radius: 10px;">
    <p>🤟 Made with ❤️ for the Deaf Community | Empowering communication through technology</p>
</div>
""", unsafe_allow_html=True)

# Debug information (remove in production)
if st.checkbox("🔧 Show Debug Info"):
    st.write("**Session State:**", st.session_state)
    st.write("**Current Working Directory:**", os.getcwd())
    st.write("**Available Model Files:**", [f for f in os.listdir("models/") if f.endswith('.pth')] if os.path.exists("models/") else "Models directory not found")
    st.write("**Available Reference Images:**", [f for f in os.listdir("ref_imgs/") if f.endswith('.jpg')] if os.path.exists("ref_imgs/") else "Reference images directory not found")