#conda activate sign
import json
import os
import random
import sqlite3
import string
import tempfile
import threading
import time
import pickle
from datetime import datetime, timedelta

import cv2
import mediapipe as mp
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import torch
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from streamlit_webrtc import webrtc_streamer
from torchvision import models, transforms

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
        background: linear-gradient(135deg, #FF9933 0%, white 50%, #138808 100%);
        padding: 2rem;
        border-radius: 15px;
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
    
    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 30px rgba(0,0,0,0.12);
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
    
    .achievement-badge {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        padding: 0.5rem 1rem;
        border-radius: 20px;
        display: inline-block;
        margin: 0.25rem;
        font-weight: bold;
        color: #8b4513;
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
    .sidebar-section {
        background: #f8f9ff;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        border-left: 4px solid #667eea;
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
PHRASES_CONFIG = {
    "Greetings": {
        "Hello": {"video": "phrases/hello1.mp4", "description": "Basic greeting gesture"},
        "How are you?": {"video": "phrases/how.mp4", "description": "Meeting greeting"},
        "Goodbye": {"video": "phrases/bye.mp4", "description": "Farewell gesture"}
    },
    "Basic Phrases": {
        "Name": {"video": "phrases/name.mp4", "description": "Say your name"},
        "Age": {"video": "phrases/age.mp4", "description": "Say your age"},
        "Home": {"video": "phrases/home.mp4", "description": "Say where you're from"}      
    },
    "Numbers": {
        "1-5": {"video": "phrases/1.mp4", "description": "Basic counting"},
        "6-10": {"video": "phrases/6.mp4", "description": "Extended counting"}
    }
}
# Database Setup for Progress Tracking
def init_db():
    conn = sqlite3.connect('progress.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS progress (
                    user_id TEXT PRIMARY KEY,
                    learned_letters TEXT,
                    practice_sessions INTEGER,
                    total_time INTEGER,
                    streak_days INTEGER,
                    achievements TEXT,
                    accuracy_history TEXT,
                    last_session TEXT
                )''')
    c.execute('''CREATE TABLE IF NOT EXISTS sessions (
                    session_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT,
                    start_time TEXT,
                    predictions TEXT,
                    accuracy REAL,
                    letters_practiced TEXT
                )''')
    conn.commit()
    conn.close()
def load_user_progress():
    conn = sqlite3.connect('progress.db')
    c = conn.cursor()
    c.execute("SELECT * FROM progress WHERE user_id = 'default'")
    result = c.fetchone()
    conn.close()
    
    if result:
        return {
            'learned_letters': set(json.loads(result[1])) if result[1] else set(),
            'practice_sessions': result[2] or 0,
            'total_time': result[3] or 0,
            'streak_days': result[4] or 0,
            'achievements': json.loads(result[5]) if result[5] else [],
            'accuracy_history': json.loads(result[6]) if result[6] else [],
            'last_session': result[7]
        }
    return {
        'learned_letters': set(),
        'practice_sessions': 0,
        'total_time': 0,
        'streak_days': "0",
        'achievements': [],
        'accuracy_history': [],
        'last_session': None
    }
def save_user_progress(progress):
    conn = sqlite3.connect('progress.db')
    c = conn.cursor()
    c.execute('''INSERT OR REPLACE INTO progress (
                    user_id, learned_letters, practice_sessions, total_time, 
                    streak_days, achievements, accuracy_history, last_session
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
                ('default', json.dumps(list(progress['learned_letters'])),
                 progress['practice_sessions'], progress['total_time'],
                 progress['streak_days'], json.dumps(progress['achievements']),
                 json.dumps(progress['accuracy_history']),
                 progress['last_session']))
    conn.commit()
    conn.close()
# Initialize Database
init_db()
# Initialize session state for tracking
if 'user_progress' not in st.session_state:
    st.session_state.user_progress = {
        'learned_letters': set(),
        'practice_sessions': 0,
        'total_time': 0,
        'streak_days': 1
    }
    st.session_state.user_progress = load_user_progress()
if 'current_session' not in st.session_state:
    st.session_state.current_session = {
        'start_time': None,
        'predictions': [],
        'accuracy': 0,
        'letters_practiced': set()
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
        st.success(f"Model loaded successfully: {model_path}")
    except FileNotFoundError:
        st.error(f"Model file not found: {model_path}")
        st.info("Please ensure your model files are in the 'models/' directory")
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
    
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
def update_progress(letter, accuracy):
    st.session_state.user_progress['learned_letters'].add(letter)
    st.session_state.current_session['letters_practiced'].add(letter)
    st.session_state.current_session['predictions'].append({
        'letter': letter,
        'accuracy': accuracy,
        'timestamp': datetime.now().isoformat()
    })
    st.session_state.user_progress['accuracy_history'].append(accuracy)
    save_user_progress(st.session_state.user_progress)
def render_user_progress():
    learned_count = len(st.session_state.user_progress['learned_letters'])
    progress_percentage = (learned_count / 26) * 100

    st.markdown(f"""
    <div class="stats-container">
        <h3>📊 Your Progress</h3>
        <p><strong>{learned_count}/26</strong> Letters Learned</p>
        <p><strong>{progress_percentage:.1f}%</strong> Complete</p>
    </div>
    """, unsafe_allow_html=True)

    st.progress(progress_percentage / 100)

    if learned_count > 0:
        st.markdown("### 🎯 Recently Learned")
        recent_letters = list(st.session_state.user_progress['learned_letters'])[-5:]
        st.write(" • ".join([l.upper() for l in recent_letters]))
def update_learning_streak():
    today = datetime.now().date()
    last_session = st.session_state.user_progress.get('last_session')

    if last_session:
        last_session_date = datetime.fromisoformat(last_session).date()
        if last_session_date == today - timedelta(days=1):
            st.session_state.user_progress['streak_days'] += 1
        elif last_session_date < today - timedelta(days=1):
            st.session_state.user_progress['streak_days'] = 0

    st.session_state.user_progress['last_session'] = datetime.now().isoformat()
    save_user_progress(st.session_state.user_progress)
def render_sidebar_progress():
    st.markdown(f"""
    <div class="sidebar-section">
        <h4>🔥 Learning Streak</h4>
        <p>{st.session_state.user_progress['streak_days']} days</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="sidebar-section">
        <h4>📈 Session Stats</h4>
        <p>Sessions: {st.session_state.user_progress['practice_sessions']}</p>
        <p>Total Time: {st.session_state.user_progress['total_time']//60} min</p>
    </div>
    """, unsafe_allow_html=True)
def get_progress_data():
    learned_letters = list(st.session_state.user_progress['learned_letters'])
    learned_count = len(learned_letters)
    
    progress_data = {
        'Category': ['Learned', 'Remaining'],
        'Count': [learned_count, 26 - learned_count]
    }
    dates = [datetime.now() - timedelta(days=i) for i in range(30, 0, -1)]
    progress_counts = np.minimum(np.arange(1, 31) // 3, learned_count)
    
    return progress_data, dates, progress_counts
def extract_hand_landmarks(image):
    """Extract hand landmarks using MediaPipe"""
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_image)
    
    if results.multi_hand_landmarks:
        landmarks = []
        for hand_landmarks in results.multi_hand_landmarks:
            for landmark in hand_landmarks.landmark:
                landmarks.extend([landmark.x, landmark.y, landmark.z])
        return np.array(landmarks)
    return None
def calculate_gesture_similarity(landmarks1, landmarks2):
    """Calculate similarity between two sets of hand landmarks"""
    if landmarks1 is None or landmarks2 is None:
        return 0.0
    
    # Normalize landmarks to handle scale differences
    landmarks1_norm = (landmarks1 - np.mean(landmarks1)) / (np.std(landmarks1) + 1e-8)
    landmarks2_norm = (landmarks2 - np.mean(landmarks2)) / (np.std(landmarks2) + 1e-8)
    
    # Calculate cosine similarity
    dot_product = np.dot(landmarks1_norm, landmarks2_norm)
    norm1 = np.linalg.norm(landmarks1_norm)
    norm2 = np.linalg.norm(landmarks2_norm)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    similarity = dot_product / (norm1 * norm2)
    # Convert to 0-1 range
    return (similarity + 1) / 2
def load_reference_landmarks(letter):
    """Load reference landmarks for a letter (you'll need to implement this)"""
    # This is a placeholder - you'll need to pre-compute and store reference landmarks
    # For now, return None to simulate missing reference data
    return None
def start_new_round():
    """Start a new game round"""
    if st.session_state.allowed_letters:
        import random
        st.session_state.game_letter = random.choice(st.session_state.allowed_letters)
        st.session_state.start_time = time.time()
        st.session_state.matched_this_round = False
        st.session_state.clear_camera_input = True
        st.session_state.round_start_time = time.time()
        
def extract_hand_landmarks(image):
    """Extract hand landmarks using MediaPipe"""
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_image)
    
    if results.multi_hand_landmarks:
        landmarks = []
        for hand_landmarks in results.multi_hand_landmarks:
            for landmark in hand_landmarks.landmark:
                landmarks.extend([landmark.x, landmark.y])  # Only x,y for 2D comparison
        return np.array(landmarks), results.multi_hand_landmarks
    return None, None
def draw_landmarks_on_image(image, hand_landmarks_list):
    """Draw hand landmarks on image"""
    annotated_image = image.copy()
    for hand_landmarks in hand_landmarks_list:
        mp_drawing.draw_landmarks(
            annotated_image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style()
        )
    return annotated_image
def calculate_landmark_distance(landmarks1, landmarks2):
    """Calculate normalized distance between two landmark sets"""
    if landmarks1 is None or landmarks2 is None:
        return float('inf')
    
    # Ensure same length
    min_len = min(len(landmarks1), len(landmarks2))
    landmarks1 = landmarks1[:min_len]
    landmarks2 = landmarks2[:min_len]
    
    # Calculate Euclidean distance
    distance = np.linalg.norm(landmarks1 - landmarks2)
    
    # Normalize by number of points
    normalized_distance = distance / (min_len / 2)  # Divide by 2 because we have x,y pairs
    
    return normalized_distance
def is_hand_gesture_valid(landmarks):
    """Simple validation to check if detected landmarks form a reasonable hand gesture"""
    if landmarks is None or len(landmarks) < 42:  # 21 points * 2 coordinates
        return False
    
    # Check if landmarks are spread out (not all clustered in one spot)
    landmarks_2d = landmarks.reshape(-1, 2)
    x_range = np.max(landmarks_2d[:, 0]) - np.min(landmarks_2d[:, 0])
    y_range = np.max(landmarks_2d[:, 1]) - np.min(landmarks_2d[:, 1])
    
    # Hand should have reasonable spread
    return x_range > 0.1 and y_range > 0.1
# Header
st.markdown("""
<div class="main-header">
    <h1>🙏🏽 Indian Sign Language Learning Hub</h1>
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
    </div>
    """, unsafe_allow_html=True)
    
    st.progress(progress_percentage / 100)
    
    if learned_count > 0:
        st.markdown("### 🎯 Recently Learned")
        recent_letters = list(st.session_state.user_progress['learned_letters'])[-5:]
        st.write(" • ".join([l.upper() for l in recent_letters]))
        
    
    
    today = datetime.now().date()
    last_session = st.session_state.user_progress['last_session']
    if last_session:
        last_session_date = datetime.fromisoformat(last_session).date()
        if last_session_date == today - timedelta(days=1):
            st.session_state.user_progress['streak_days'] += 1
        elif last_session_date < today - timedelta(days=1):
            st.session_state.user_progress['streak_days'] = 0
    st.session_state.user_progress['last_session'] = datetime.now().isoformat()
    save_user_progress(st.session_state.user_progress)
        
    st.markdown(f"""
    <div class="sidebar-section">
        <h4>🔥 Learning Streak</h4>
        <p>{st.session_state.user_progress['streak_days']} days</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="sidebar-section">
        <h4>📈 Session Stats</h4>
        <p>Sessions: {st.session_state.user_progress['practice_sessions']}</p>
        <p>Total Time: {st.session_state.user_progress['total_time']//60}min</p>
    </div>
    """, unsafe_allow_html=True)
# Main Content Tabs
tabs = st.tabs([
    "🎯 Interactive Learning",
    "📚 Alphabet Reference", 
    "🗣️ Common Phrases",
    "🎮 Practice Games",
    "📊 Progress Analytics",
    "ℹ️ About ISL"
])
# Tab 1: Live Practice (WORKING VERSION)
with tabs[0]:
    sub_tabs = st.tabs(["🔤 Alphabet", "🗣️ Basic Words"])
    with sub_tabs[0]: 
        st.header("🎯 AI-Powered Sign Recognition")
        col1, col2 = st.columns([1, 2], gap="large")
        
        with col1:
            # st.markdown('<div class="feature-card">', unsafe_allow_html=True)
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
            # st.markdown('<div class="webcam-container">', unsafe_allow_html=True)
            
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
                        stframe.image(frame, channels="BGR", use_container_width=True)
                        
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

    with sub_tabs[1]:  # Basic Words tab
       st.header("🗣️ AI-Powered Sign Recognition - Basic Words")
       
       col1, col2 = st.columns([1, 2], gap="large")

       with col1:
           st.subheader("📖 Recognition Settings")

           # Load traditional ML model (make sure pickle is imported)
           with open('new.p', 'rb') as f:
               word_model = pickle.load(f)['model']

           word_labels = {
               0: 'hi', 1: 'thank you', 2: 'peace', 3: 'ok', 4: 'house',
               5: 'Indian', 6: 'i', 7: 'you', 8: 'man', 9: 'woman'
           }

           st.info("Model: `new.p` - Traditional ML (DecisionTree or similar)")
           st.write("**Words:**", ", ".join(word_labels.values()))

           st.subheader("⚙️ Detection Settings")
           word_conf_thresh = st.slider("Confidence Threshold", 0.5, 1.0, 0.75)
           show_landmarks_word = st.checkbox("Show Hand Landmarks", value=True, key="basic_words_landmarks")


           # Start/Stop buttons
           
           word_start = st.button("🚀 Start Webcam (Words)", type="primary", key="start_webcam_words")
           word_stop = st.button("⏹️ Stop Webcam", key="stop_webcam_words")


           if 'word_webcam_running' not in st.session_state:
               st.session_state.word_webcam_running = False

           if word_start:
               st.session_state.word_webcam_running = True
           if word_stop:
               st.session_state.word_webcam_running = False

           # Predicted label display
           if st.session_state.get("word_prediction"):
               st.markdown(f"""
               <div class="prediction-box">
                   🗣️ Detected Word: {st.session_state.word_prediction.upper()}<br>
                   Confidence: {st.session_state.word_confidence:.1%}
               </div>
               """, unsafe_allow_html=True)

       with col2:
           # Webcam display
           word_frame = st.empty()
           ref_placeholder = st.empty()
           word_status = st.empty()

           if st.session_state.word_webcam_running:
               mp_hands = mp.solutions.hands
               hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.3)
               mp_draw = mp.solutions.drawing_utils
               mp_styles = mp.solutions.drawing_styles

               cap = cv2.VideoCapture(0)
               if not cap.isOpened():
                   word_status.error("❌ Cannot open webcam")
                   st.session_state.word_webcam_running = False
               else:
                   word_status.success("🎥 Webcam active - Show your signs!")

                   while st.session_state.word_webcam_running:
                       ret, frame = cap.read()
                       if not ret:
                           word_status.warning("Can't read from webcam")
                           break

                       frame = cv2.flip(frame, 1)
                       H, W, _ = frame.shape
                       rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                       results = hands.process(rgb)

                       if results.multi_hand_landmarks:
                           for handLms in results.multi_hand_landmarks:
                               if show_landmarks_word:
                                   mp_draw.draw_landmarks(
                                       frame, handLms,
                                       mp_hands.HAND_CONNECTIONS,
                                       mp_styles.get_default_hand_landmarks_style(),
                                       mp_styles.get_default_hand_connections_style()
                                   )

                               x_list = [lm.x for lm in handLms.landmark]
                               y_list = [lm.y for lm in handLms.landmark]

                               xmin, ymin = min(x_list), min(y_list)
                               xmax, ymax = max(x_list), max(y_list)

                               features = []
                               for lm in handLms.landmark:
                                   features.append(lm.x - xmin)
                                   features.append(lm.y - ymin)

                               x1 = int(xmin * W) - 15
                               y1 = int(ymin * H) - 15
                               x2 = int(xmax * W) + 15
                               y2 = int(ymax * H) + 15

                               pred = word_model.predict([np.asarray(features)])
                               label = word_labels[int(pred[0])]

                               st.session_state.word_prediction = label
                               st.session_state.word_confidence = 1.0  # Classic ML model has no probability output

                               overlay = frame.copy()
                               alpha = 0.6
                               box_width = 180
                               box_height = 40
                               cv2.rectangle(overlay, (x1, y1 - box_height), (x1 + box_width, y1), (30, 30, 30), -1)
                               frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

                               # Rounded prediction box
                               cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 150), 2)
                               cv2.putText(frame, label, (x1 + 10, y1 - 10),
                                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                       word_frame.image(frame, channels="BGR", use_container_width=True)

                   cap.release()
                   word_status.info("📷 Webcam stopped")

           else:
               word_frame.info("👆 Click 'Start Webcam (Words)' to begin sign recognition")

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

with tabs[2]:
    st.header("🗣️ Essential ISL Phrases")
    st.markdown("Learn practical signs for daily communication")
    
    selected_category = st.selectbox("Choose Category:", list(PHRASES_CONFIG.keys()))
    
    phrases = PHRASES_CONFIG[selected_category]
    
    phrase_cols = st.columns(2)
    
    for i, (phrase, config) in enumerate(phrases.items()):
        with phrase_cols[i % 2]:
            st.markdown(f"""
            <div class="feature-card">
                <h3>{phrase}</h3>
                <p>{config['description']}</p>
            """, unsafe_allow_html=True)
            video_path = config['video']
            if os.path.exists(video_path):
                st.video(video_path)
            else:
                st.markdown(f"*Video for '{phrase}' not found.*")
            st.markdown("</div>", unsafe_allow_html=True)
            
            if st.button(f"Practice {phrase}", key=f"practice_{phrase}"):
                st.success(f"Starting practice session for '{phrase}'")

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.4,
    min_tracking_confidence=0.5
)

with tabs[3]:
    st.header("🎮 Interactive Learning Games")
    allowed_letters = ['a', 'c', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 's', 't', 'u']
    for key, default in {
        "challenge_active": False,
        "game_score": 0,
        "game_letter": None,
        "matched_this_round": False,
        "last_detection_time": 0
    }.items():
        if key not in st.session_state:
            st.session_state[key] = default
    
    def start_new_round():
        st.session_state.game_letter = random.choice(allowed_letters)
        st.session_state.matched_this_round = False
        st.session_state.last_detection_time = 0
            
    if "game_letter" not in st.session_state or not st.session_state.get("challenge_active"):
        st.session_state.game_letter = random.choice(allowed_letters)


    game_col1, game_col2 = st.columns(2)

    with game_col1:
        st.markdown("""
        <div class="feature-card">
            <h3>🎯 Sign Challenge</h3>
            <p>Random letters appear - show the correct sign!</p>
            <ul>
                <li>⏱️ Time-based challenges</li>
                <li>🏆 Score tracking</li>
                <li>📈 Difficulty progression</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            color: white;
            margin: 20px 0;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        ">
            <h2 style="margin: 0; font-size: 2em;">🏆</h2>
            <h3 style="margin: 5px 0;">Current Score</h3>
            <h1 style="margin: 0; font-size: 3em; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">{st.session_state.game_score}</h1>
        </div>
        """, unsafe_allow_html=True)
        

        if st.button("Start Sign Challenge", key="sign_challenge"):
            st.session_state.game_mode = "challenge"
            st.session_state.game_score = 0
            st.session_state.game_letter = random.choice(allowed_letters)
            st.session_state.challenge_active = True
            st.rerun()

    with game_col2:
        if st.session_state.get("challenge_active"):
            chosen_letter = random.choice(allowed_letters)
            ref_img_path = REFERENCE_IMAGES[chosen_letter]
            
            st.subheader(f"👋 Make the sign for: **{chosen_letter.upper()}**")
            st.session_state.game_letter = chosen_letter

            mp_hands, hands, mp_draw = init_mediapipe()

            ref_img_bgr = cv2.imread(ref_img_path)
            ref_img_rgb = cv2.cvtColor(ref_img_bgr, cv2.COLOR_BGR2RGB)
            ref_results = hands.process(ref_img_rgb)

            ref_hand_img = None
            if ref_results.multi_hand_landmarks:
                for handLms in ref_results.multi_hand_landmarks:
                    x_list = [lm.x for lm in handLms.landmark]
                    y_list = [lm.y for lm in handLms.landmark]
                    xmin, xmax = int(min(x_list) * ref_img_rgb.shape[1]), int(max(x_list) * ref_img_rgb.shape[1])
                    ymin, ymax = int(min(y_list) * ref_img_rgb.shape[0]), int(max(y_list) * ref_img_rgb.shape[0])
                    x1, y1 = max(xmin - 20, 0), max(ymin - 20, 0)
                    x2, y2 = min(xmax + 20, ref_img_rgb.shape[1]), min(ymax + 20, ref_img_rgb.shape[0])
                    ref_hand_img = ref_img_rgb[y1:y2, x1:x2]

            stframe = st.empty()
            cap = cv2.VideoCapture(0)

            success = False
            timeout = time.time() + 10

            while time.time() < timeout:
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
                        xmin, xmax = int(min(x_list) * w), int(max(x_list) * w)
                        ymin, ymax = int(min(y_list) * h), int(max(y_list) * h)
                        x1, y1 = max(xmin - 20, 0), max(ymin - 20, 0)
                        x2, y2 = min(xmax + 20, w), min(ymax + 20, h)

                        hand_img = frame[y1:y2, x1:x2]
                        if hand_img.size != 0 and ref_hand_img is not None:
                            try:
                                hand_gray = cv2.cvtColor(hand_img, cv2.COLOR_BGR2GRAY)
                                ref_gray = cv2.cvtColor(ref_hand_img, cv2.COLOR_BGR2GRAY)

                                hand_resized = cv2.resize(hand_gray, (200, 200))
                                ref_resized = cv2.resize(ref_gray, (200, 200))

                                similarity_score, _ = ssim(hand_resized, ref_resized, full=True)

                                if similarity_score > 0.4:
                                    st.success(f"✅ Sign Matched!")
                                    st.balloons()
                                    st.session_state.game_score += 1
                                    success = True
                                    break
                                
                            except Exception as e:
                                st.error(f"Error comparing images: {e}")

                        if 'show_landmarks' in locals() and show_landmarks:
                            mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

                stframe.image(frame, channels="BGR")
                if success:
                    break

            cap.release()
            st.markdown(f"### 🎉 Your Score: **{st.session_state.game_score}**")

            if st.button("Next Challenge"):
                st.session_state.game_letter = random.choice(allowed_letters)
                st.rerun()

            if st.button("End Game"):
                st.session_state.challenge_active = False
                st.session_state.game_mode = None
          

                   
with tabs[4]:  # or wherever your analytics tab is
    st.header("📊 Your Learning Analytics")

    progress_data, sample_dates, sample_progress = get_progress_data()

    col1, col2 = st.columns(2)
    
    with col1:
        fig_pie = px.pie(progress_data, values='Count', names='Category', 
                         title="📈 Alphabet Learning Progress")
        fig_line = px.line(x=sample_dates, y=sample_progress,
                           title="📈 Learning Progress Over Time",
                           labels={'x': 'Date', 'y': 'Letters Learned'})
        fig_line.update_layout(plot_bgcolor='rgba(0,0,0,0)',
                               paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_pie, use_container_width=True)
        st.plotly_chart(fig_line, use_container_width=True)

    
    with col2:
        # Accuracy by letter from predictions
        letter_accuracies = {letter: [] for letter in 'abcdefghijklmnopqrstuvwxyz'}
        for pred in st.session_state.current_session.get('predictions', []):
            letter_accuracies[pred['letter']].append(pred['accuracy'])
        y_vals = [np.mean(letter_accuracies[letter]) if letter_accuracies[letter] else 0 for letter in 'abcdefghijklmnopqrstuvwxyz']

        fig_acc = px.bar(
            x=[chr(i) for i in range(ord('a'), ord('z')+1)],
            y=y_vals,
            title="🎯 Accuracy by Letter",
            labels={'x': 'Letter', 'y': 'Accuracy (%)'}
        )
        fig_acc.update_layout(plot_bgcolor='rgba(0,0,0,0)',
                              paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_acc, use_container_width=True)
        
        # Generate streak data for last 14 days (simulate based on last_session)
        today = datetime.now().date()
        dates = [today - timedelta(days=i) for i in range(13, -1, -1)]
        active = [0] * 14
        
        # Simulated: mark last streak_days as active
        streak_len = int(st.session_state.user_progress.get('streak_days', 0))
        for i in range(1, min(streak_len + 1, 14) + 1):
            active[-i] = 1
        
        fig_streaks = px.bar(
            x=[d.strftime("%b %d") for d in dates],
            y=active,
            title="🔥 Streak Activity (Past 14 Days)",
            labels={'x': 'Date', 'y': 'Active (1=Practiced)'},
            color=active,
            color_continuous_scale='reds'
        )
        fig_streaks.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            showlegend=False
        )
        
        st.plotly_chart(fig_streaks, use_container_width=True)

        
    st.markdown("<h3 style='text-align: center;'>📅 This Week's Learning Stats</h3>", unsafe_allow_html=True)


    stats = {
        "🧠 Total Letters Learned": len(st.session_state.user_progress['learned_letters']),
        "✋ Letters Practiced": len(st.session_state.user_progress['learned_letters']),
        "📈 Sessions Completed": st.session_state.user_progress['practice_sessions'],
        "🆕 New Letters Learned": len(st.session_state.user_progress['learned_letters']),
        "🎯 Average Accuracy": f"{np.mean(st.session_state.user_progress.get('accuracy_history', [0])):.1f}%",
        "⏱️ Total Practice Time": f"{st.session_state.user_progress['total_time']//60} min",
        "🔥 Longest Streak": f"{st.session_state.user_progress['streak_days']} days"
    }
    
    cols = st.columns(3)
    for idx, (label, value) in enumerate(stats.items()):
        with cols[idx % 3]:
            st.metric(label=label, value=value)  
# Tab 4: About ISL
with tabs[5]:
    st.header("ℹ️ About Indian Sign Language")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### 👳🏼‍♂️ What is Indian Sign Language?
        
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
        
        ### 🤝 Deaf Culture Etiquette
        
        - Make eye contact when signing
        - Tap gently on shoulder to get attention
        - Don't talk while someone is signing
        - Learn about deaf culture and history
        - Be respectful and patient
        """)
    
    
    with col2:
        st.markdown("""
                    
        ### 📱 App Features
        <div class="feature-card">
            <ul>
                <li>🤖 AI-powered recognition</li>
                <li>📊 Progress tracking</li>
                <li>📚 Complete reference</li>
                <li>🗣️ Common phrases</li>
            </ul>
        </div>
        
        ### 🛠️ Technical Requirements
        
        <div class="feature-card">
        
        **For Best Results:**
        - 📷 Working webcam
        - 💡 Good lighting
        - 🖥️ Stable internet connection
        - 📁 Model files in `/models/`
        - 🖼️ Reference images in `/ref_imgs/`
        </div>
        
        ### 📁 Required Files
        <div class="feature-card">
        
        **Model Files:**
        - `models/model_af.pth`
        - `models/model_gl.pth` 
        - `models/model_ms.pth`
        - `models/model_tz.pth`
        </div>
        
        
        ### 📞 Support
        <div class="feature-card">
            <p>Need help? Contact us:</p>
            <ul>
                <li>📧 support@isllearninghub.com</li>
                <li>📱 WhatsApp: +91-7783009847</li>
            </ul>
        </div>

        """, unsafe_allow_html=True)
# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 1rem; background: #f8f9ff; border-radius: 10px;">
    <p>Made with ❤️ for the Deaf Community 🙏🏽</p>
</div>
""", unsafe_allow_html=True)