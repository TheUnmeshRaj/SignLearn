
import json
import os
import random
import sqlite3
import tempfile
import threading
import time
from datetime import datetime, timedelta

import cv2
import mediapipe as mp
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import torch
from PIL import Image
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


# Model Configurations
# MODEL_CONFIGS = {
    # "Beginner (A–F)": {
        # "path": "models/model_af.pth",
        # "classes": ['a', 'b', 'c', 'd', 'e', 'f'],
        # "description": "Perfect for starting your ISL journey",
        # "difficulty": "🟢 Easy"
    # },
    # "Elementary (G–L)": {
        # "path": "models/model_gl.pth",
        # "classes": ['g', 'h', 'i', 'j', 'k', 'l'],
        # "description": "Build upon your basic knowledge",
        # "difficulty": "🟡 Medium"
    # },
    # "Intermediate (M–S)": {
        # "path": "models/model_ms.pth",
        # "classes": ['m', 'n', 'o', 'p', 'q', 'r', 's'],
        # "description": "Challenge yourself with more letters",
        # "difficulty": "🟠 Hard"
    # },
    # "Advanced (T–Z)": {
        # "path": "models/model_tz.pth",
        # "classes": ['t', 'u', 'v', 'w', 'x', 'y', 'z'],
        # "description": "Master the complete alphabet",
        # "difficulty": "🔴 Expert"
    # }
# }



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
        "Hello": {"video": "phrases/hello.mp4", "description": "Basic greeting gesture"},
        "Good Morning": {"video": "phrases/good_morning.mp4", "description": "Morning greeting"},
        "Good Evening": {"video": "phrases/good_evening.mp4", "description": "Evening greeting"},
        "Goodbye": {"video": "phrases/goodbye.mp4", "description": "Farewell gesture"}
    },
    "Courtesy": {
        "Thank You": {"video": "phrases/thank_you.mp4", "description": "Express gratitude"},
        "Please": {"video": "phrases/please.mp4", "description": "Polite request"},
        "Sorry": {"video": "phrases/sorry.mp4", "description": "Apologize gesture"},
        "Welcome": {"video": "phrases/welcome.mp4", "description": "Welcoming gesture"}
    },
    "Basic Needs": {
        "Water": {"video": "phrases/water.mp4", "description": "Ask for water"},
        "Food": {"video": "phrases/food.mp4", "description": "Ask for food"},
        "Help": {"video": "phrases/help.mp4", "description": "Request assistance"},
        "Yes": {"video": "phrases/yes.mp4", "description": "Affirmative response"},
        "No": {"video": "phrases/no.mp4", "description": "Negative response"}
    },
    "Numbers": {
        "1-5": {"video": "phrases/numbers_1_5.mp4", "description": "Basic counting"},
        "6-10": {"video": "phrases/numbers_6_10.mp4", "description": "Extended counting"}
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
    st.header("🎯 AI-Powered Sign Recognition")
    
    col1, col2 = st.columns([1, 2])
    
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
        
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)
        mp_draw = mp.solutions.drawing_utils
        st.title("Sign Language Detection (Image Snapshot)")
        
        img_file = st.camera_input("Show your hand sign here")
        
        if img_file:
            img = Image.open(img_file).convert("RGB")
            img_np = np.array(img)
        
            # Process image with MediaPipe
            results = hands.process(img_np)
        
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_draw.draw_landmarks(img_np, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        
                # Get bounding box of hand landmarks
                h, w, _ = img_np.shape
                x_coords = [lm.x for lm in hand_landmarks.landmark]
                y_coords = [lm.y for lm in hand_landmarks.landmark]
        
                x_min, x_max = int(min(x_coords) * w), int(max(x_coords) * w)
                y_min, y_max = int(min(y_coords) * h), int(max(y_coords) * h)
        
                # Add padding and clip
                pad = 20
                x1 = max(x_min - pad, 0)
                y1 = max(y_min - pad, 0)
                x2 = min(x_max + pad, w)
                y2 = min(y_max + pad, h)
        
                hand_img = img_np[y1:y2, x1:x2]
        
                if hand_img.size != 0:
                    pil_hand = Image.fromarray(hand_img)
                    st.image(pil_hand, caption="Cropped Hand Region", width=200)
        
                    # Prepare tensor and predict
                    input_tensor = transform(pil_hand).unsqueeze(0).to(device)
                    with torch.no_grad():
                        output = model(input_tensor)
                        probs = torch.nn.functional.softmax(output, dim=1)
                        conf, pred_idx = torch.max(probs, 1)
                        predicted_letter = class_names[pred_idx.item()]
                        confidence = conf.item()
        
                    st.markdown(f"### Prediction: **{predicted_letter.upper()}** (Confidence: {confidence:.2f})")
                else:
                    st.warning("Hand region too small for prediction.")
        
                # Show image with landmarks
                st.image(img_np, caption="Hand Landmarks Overlay", channels="RGB")
        
            else:
                st.warning("No hand detected. Try again.")

    
    # WORKING WEBCAM LOGIC
 
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
                

# Tab 4: Practice Games
with tabs[3]:
    st.header("🎮 Interactive Learning Games")

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

        if st.button("Start Sign Challenge", key="sign_challenge"):
            st.session_state.game_mode = "challenge"
            st.session_state.game_score = 0
            st.session_state.game_letter = random.choice(list(REFERENCE_IMAGES.keys()))
            st.session_state.challenge_active = True
            st.rerun()

    with game_col2:
        if st.session_state.get("challenge_active"):
            st.subheader(f"👋 Make the sign for: **{st.session_state.game_letter.upper()}**")
            ref_img_path = REFERENCE_IMAGES[st.session_state.game_letter]
            st.image(ref_img_path, caption=f"Reference Sign for '{st.session_state.game_letter.upper()}'", width=300)

            # Webcam input
            stframe = st.empty()
            cap = cv2.VideoCapture(0)
            mp_hands, hands, mp_draw = init_mediapipe()

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
                                predicted = class_names[pred.item()]

                                if predicted == st.session_state.game_letter:
                                    st.success("✅ Correct Sign!")
                                    st.session_state.game_score += 1
                                    success = True
                                    break

                        if show_landmarks:
                            mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

                stframe.image(frame, channels="BGR")
                if success:
                    break

            cap.release()
            st.markdown(f"### 🎉 Your Score: **{st.session_state.game_score}**")
            if st.button("Next Challenge"):
                st.session_state.game_letter = random.choice(list(REFERENCE_IMAGES.keys()))
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
        streak_len = st.session_state.user_progress.get('streak_days', 0)
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
        - 📁 Model files in `/models/` directory
        - 🖼️ Reference images in `/ref_imgs/` directory
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
                <li>📧 islleanringhub@gmail.com</li>
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