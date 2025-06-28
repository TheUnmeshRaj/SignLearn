import json
import os
import random
import sqlite3
import time
from datetime import datetime, timedelta

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
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
    
    .achievement-badge {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        padding: 0.5rem 1rem;
        border-radius: 20px;
        display: inline-block;
        margin: 0.25rem;
        font-weight: bold;
        color: #8b4513;
    }
    
    .webcam-container {
        border: 3px solid #667eea;
        border-radius: 15px;
        padding: 1rem;
        background: #f8f9ff;
    }
    
    .reference-image {
        border: 2px solid #764ba2;
        border-radius: 10px;
        padding: 10px;
        background: white;
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

# Model Configurations
MODEL_CONFIGS = {
    "Beginner (A–F)": {
        "path": "models/model_af.pth",
        "classes": ['a', 'b', 'c', 'd', 'e', 'f'],
        "description": "Perfect for starting your ISL journey",
        "difficulty": "🟢 Easy"
    },
    "Elementary (G–L)": {
        "path": "models/model_gl.pth",
        "classes": ['g', 'h', 'i', 'j', 'k', 'l'],
        "description": "Build upon your basic knowledge",
        "difficulty": "🟡 Medium"
    },
    "Intermediate (M–S)": {
        "path": "models/model_ms.pth",
        "classes": ['m', 'n', 'o', 'p', 'q', 'r', 's'],
        "description": "Challenge yourself with more letters",
        "difficulty": "🟠 Hard"
    },
    "Advanced (T–Z)": {
        "path": "models/model_tz.pth",
        "classes": ['t', 'u', 'v', 'w', 'x', 'y', 'z'],
        "description": "Master the complete alphabet",
        "difficulty": "🔴 Expert"
    }
}

# Reference Images Dictionary
REFERENCE_IMAGES = {chr(i): f'ref_imgs/{chr(i)}.jpg' for i in range(ord('a'), ord('z')+1)}

# Common Phrases Configuration
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

# Initialize session state
if 'user_progress' not in st.session_state:
    st.session_state.user_progress = load_user_progress()

if 'current_session' not in st.session_state:
    st.session_state.current_session = {
        'start_time': None,
        'predictions': [],
        'accuracy': 0,
        'letters_practiced': set()
    }

# Device and Transform Setup
device = torch.device("cpu")
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Caching Functions
@st.cache_resource
def load_model(model_path, num_classes):
    try:
        model = models.resnet18(pretrained=False)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=device))
        else:
            st.error(f"Model file {model_path} not found. Please ensure model files are in the 'models' directory.")
        model.to(device)
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None
def init_mediapipe():
    try:
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        mp_draw = mp.solutions.drawing_utils
        return mp_hands, hands, mp_draw
    except Exception as e:
        st.error(f"Error initializing MediaPipe: {str(e)}")
        return None, None, None

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

def calculate_achievements():
    achievements = []
    learned = len(st.session_state.user_progress['learned_letters'])
    
    if learned >= 5:
        achievements.append("🌟 First Steps - Learned 5 letters!")
    if learned >= 13:
        achievements.append("📚 Alphabet Explorer - Learned half the alphabet!")
    if learned >= 26:
        achievements.append("🏆 Alphabet Master - Learned all letters!")
    if st.session_state.user_progress['practice_sessions'] >= 10:
        achievements.append("💪 Dedicated Learner - 10 practice sessions!")
    
    st.session_state.user_progress['achievements'] = achievements
    save_user_progress(st.session_state.user_progress)
    return achievements

# Header
st.markdown("""
<div class="main-header">
    <h1>🤟 Indian Sign Language Learning Hub</h1>
    <p>Master ISL with AI-powered recognition, interactive lessons, and comprehensive tracking</p>
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
    
    achievements = calculate_achievements()
    if achievements:
        st.markdown("### 🏆 Achievements")
        for achievement in achievements[-3:]:
            st.markdown(f'<div class="achievement-badge">{achievement}</div>', unsafe_allow_html=True)
    
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

# Tab 1: Interactive Learning
with tabs[0]:
    st.header("🎯 AI-Powered Sign Recognition")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        st.subheader("📖 Learning Options")
        
        selected_level = st.selectbox(
            "Choose Your Level:",
            list(MODEL_CONFIGS.keys()),
            help="Select difficulty level based on your comfort"
        )
        
        level_config = MODEL_CONFIGS[selected_level]
        st.info(f"**{level_config['difficulty']}** - {level_config['description']}")
        
        model = load_model(level_config['path'], len(level_config['classes']))
        if model is None:
            st.error("Failed to load model. Please check model files.")
        
        practice_mode = st.radio(
            "Practice Mode:",
            ["Free Practice", "Guided Learning", "Quiz Mode"],
            help="Choose how you want to practice"
        )
        
        st.subheader("⚙️ Settings")
        confidence_threshold = st.slider("Confidence Threshold", 0.5, 0.95, 0.7)
        show_landmarks = st.checkbox("Show Hand Landmarks", True)
        
        col_start, col_stop = st.columns(2)
        with col_start:
            start_session = st.button("🚀 Start Practice", type="primary")
        with col_stop:
            stop_session = st.button("⏹️ Stop Session")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        if practice_mode == "Guided Learning":
            if 'target_letter' not in st.session_state:
                st.session_state.target_letter = level_config['classes'][0]
            
            st.markdown(f"""
            <div class="feature-card">
                <h3>🎯 Practice This Letter:</h3>
                <h1 style="text-align: center; color: #667eea;">{st.session_state.target_letter.upper()}</h1>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="webcam-container">', unsafe_allow_html=True)
        webcam_placeholder = st.empty()
        st.markdown('</div>', unsafe_allow_html=True)
        
        reference_placeholder = st.empty()
        
        stats_col1, stats_col2, stats_col3 = st.columns(3)
        with stats_col1:
            accuracy_placeholder = st.empty()
        with stats_col2:
            prediction_placeholder = st.empty()
        with stats_col3:
            confidence_placeholder = st.empty()
    
    # Webcam Processing Logic
    if start_session and model is not None:
        st.session_state.current_session['start_time'] = time.time()
        st.session_state.user_progress['practice_sessions'] += 1
        save_user_progress(st.session_state.user_progress)
        
        mp_hands, hands, mp_draw = init_mediapipe()
        if hands is None:
            st.error("Failed to initialize MediaPipe. Webcam feature unavailable.")
        else:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.error("Failed to access webcam.")
            else:
                st.session_state.webcam_active = True
                while st.session_state.get('webcam_active', False):
                    ret, frame = cap.read()
                    if not ret:
                        st.error("Failed to capture video frame.")
                        break
                    
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = hands.process(frame_rgb)
                    
                    if results.multi_hand_landmarks and show_landmarks:
                        for hand_landmarks in results.multi_hand_landmarks:
                            mp_draw.draw_landmarks(frame_rgb, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    
                    if results.multi_hand_landmarks:
                        hand_img = Image.fromarray(frame_rgb)
                        hand_img = transform(hand_img).unsqueeze(0).to(device)
                        with torch.no_grad():
                            outputs = model(hand_img)
                            probabilities = torch.nn.functional.softmax(outputs, dim=1)
                            confidence, predicted = torch.max(probabilities, 1)
                            confidence = confidence.item()
                            predicted_letter = level_config['classes'][predicted.item()]
                            
                            if confidence >= confidence_threshold:
                                update_progress(predicted_letter, confidence)
                                prediction_placeholder.markdown(f"**Predicted Letter:** {predicted_letter.upper()}")
                                confidence_placeholder.markdown(f"**Confidence:** {confidence:.2%}")
                                if practice_mode == "Guided Learning" and predicted_letter == st.session_state.target_letter:
                                    accuracy_placeholder.markdown(f"**Accuracy:** Correct!")
                                    if len(st.session_state.current_session['letters_practiced']) < len(level_config['classes']):
                                        remaining_letters = [l for l in level_config['classes'] if l not in st.session_state.current_session['letters_practiced']]
                                        st.session_state.target_letter = random.choice(remaining_letters) if remaining_letters else level_config['classes'][0]
                                elif practice_mode == "Guided Learning":
                                    accuracy_placeholder.markdown(f"**Accuracy:** Incorrect, try {st.session_state.target_letter.upper()}")
                                else:
                                    accuracy_placeholder.markdown(f"**Accuracy:** {confidence:.2%}")
                    
                    # Display webcam feed
                    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                    webcam_placeholder.image(frame_bgr, channels="BGR")
                    
                    # Display reference image
                    ref_img_path = REFERENCE_IMAGES.get(st.session_state.get('target_letter', 'a'))
                    if os.path.exists(ref_img_path):
                        reference_placeholder.image(ref_img_path, caption=f"Reference: {st.session_state.get('target_letter', 'a').upper()}")
                    else:
                        reference_placeholder.markdown(f"*Reference image for {st.session_state.get('target_letter', 'a').upper()} not found.*")
                    
                    if stop_session:
                        st.session_state.webcam_active = False
                        break
                
                cap.release()
                hands.close()
    
    if stop_session:
        if st.session_state.get('webcam_active', False):
            st.session_state.webcam_active = False
            if st.session_state.current_session['start_time']:
                session_time = time.time() - st.session_state.current_session['start_time']
                st.session_state.user_progress['total_time'] += int(session_time)
                save_user_progress(st.session_state.user_progress)
        st.session_state.current_session = {
            'start_time': None,
            'predictions': [],
            'accuracy': 0,
            'letters_practiced': set()
        }

# Tab 2: Alphabet Reference
with tabs[1]:
    st.header("📚 Complete ISL Alphabet Reference")
    st.markdown("Click on any letter to see its sign and practice tips")
    
    cols = st.columns(6)
    alphabet = 'abcdefghijklmnopqrstuvwxyz'
    
    for i, letter in enumerate(alphabet):
        with cols[i % 6]:
            if st.button(f"{letter.upper()}", key=f"letter_{letter}"):
                st.session_state.selected_letter = letter
    
    if 'selected_letter' in st.session_state:
        letter = st.session_state.selected_letter
        col1, col2 = st.columns(2)
        
        with col1:
            ref_img_path = REFERENCE_IMAGES.get(letter)
            st.markdown(f"""
            <div class="reference-image">
                <h2>Letter: {letter.upper()}</h2>
            """, unsafe_allow_html=True)
            if os.path.exists(ref_img_path):
                st.image(ref_img_path, caption=f"ISL Sign for {letter.upper()}")
            else:
                st.markdown(f"*Reference image for {letter.upper()} not found.*")
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            ### 💡 Practice Tips for '{letter.upper()}'
            - Keep your hand steady and well-lit
            - Maintain consistent finger positioning
            - Practice the motion slowly first
            - Hold the sign for 2-3 seconds
            
            ### 🎯 Common Mistakes
            - Finger positioning too loose
            - Hand too close/far from camera
            - Poor lighting conditions
            """)

# Tab 3: Common Phrases
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
            st.session_state.game_letter = random.choice(list('abcdefghijklmnopqrstuvwxyz'))
            st.rerun()
    
    with game_col2:
        st.markdown("""
        <div class="feature-card">
            <h3>🧩 Memory Match</h3>
            <p>Match letters with their corresponding signs</p>
            <ul>
                <li>🧠 Memory training</li>
                <li>🎨 Visual learning</li>
                <li>⭐ Progressive levels</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("Start Memory Game", key="memory_game"):
            st.session_state.game_mode = "memory"
            st.session_state.game_score = 0
            st.session_state.memory_cards = [(letter, f'ref_imgs/{letter}.jpg') for letter in random.sample(list('abcdefghijklmnopqrstuvwxyz'), 8)] * 2
            random.shuffle(st.session_state.memory_cards)
            st.session_state.memory_flipped = []
            st.session_state.memory_matched = []
            st.rerun()
    
    if 'game_mode' in st.session_state:
        if st.session_state.game_mode == "challenge":
            st.markdown(f"""
            <div class="feature-card">
                <h3>Sign Challenge</h3>
                <p>Show the sign for: <strong>{st.session_state.game_letter.upper()}</strong></p>
                <p>Score: {st.session_state.game_score}</p>
            </div>
            """, unsafe_allow_html=True)
            if st.button("Correct", key="challenge_correct"):
                st.session_state.game_score += 1
                st.session_state.game_letter = random.choice(list('abcdefghijklmnopqrstuvwxyz'))
                st.rerun()
            if st.button("Incorrect", key="challenge_incorrect"):
                st.session_state.game_letter = random.choice(list('abcdefghijklmnopqrstuvwxyz'))
                st.rerun()
        
        elif st.session_state.game_mode == "memory":
            st.markdown(f"""
            <div class="feature-card">
                <h3>Memory Match</h3>
                <p>Score: {st.session_state.game_score}</p>
            </div>
            """, unsafe_allow_html=True)
            cols = st.columns(4)
            for i, (letter, img_path) in enumerate(st.session_state.memory_cards):
                with cols[i % 4]:
                    if i in st.session_state.memory_matched:
                        st.markdown(f"**{letter.upper()}**")
                        if os.path.exists(img_path):
                            st.image(img_path, width=100)
                        else:
                            st.markdown(f"*Image not found*")
                    elif i in st.session_state.memory_flipped:
                        st.markdown(f"**{letter.upper()}**")
                        if os.path.exists(img_path):
                            st.image(img_path, width=100)
                        else:
                            st.markdown(f"*Image not found*")
                    else:
                        if st.button("Flip", key=f"memory_{i}"):
                            if len(st.session_state.memory_flipped) < 2:
                                st.session_state.memory_flipped.append(i)
                                if len(st.session_state.memory_flipped) == 2:
                                    card1, card2 = st.session_state.memory_flipped
                                    if st.session_state.memory_cards[card1][0] == st.session_state.memory_cards[card2][0]:
                                        st.session_state.memory_matched.extend([card1, card2])
                                        st.session_state.game_score += 1
                                        st.session_state.memory_flipped = []
                                    else:
                                        time.sleep(1)
                                        st.session_state.memory_flipped = []
                                    st.rerun()

# Tab 5: Progress Analytics
with tabs[4]:
    st.header("📊 Your Learning Analytics")
    
    sample_dates = [datetime.now() - timedelta(days=i) for i in range(30, 0, -1)]
    sample_progress = np.cumsum(np.random.randint(0, 3, 30))
    sample_accuracy = np.random.uniform(70, 95, 30)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_progress = px.line(
            x=sample_dates,
            y=sample_progress,
            title="📈 Learning Progress Over Time",
            labels={'x': 'Date', 'y': 'Letters Learned'}
        )
        fig_progress.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
        )
        st.plotly_chart(fig_progress, use_container_width=True)
        
        st.markdown(f"""
        ### 📅 This Week's Stats
        - **Sessions Completed:** {st.session_state.user_progress['practice_sessions']}
        - **New Letters Learned:** {len(st.session_state.user_progress['learned_letters'])}
        - **Average Accuracy:** {np.mean(st.session_state.user_progress['accuracy_history']):.1f}%
        - **Total Practice Time:** {st.session_state.user_progress['total_time']//60}min
        """)
    
    with col2:
        fig_accuracy = px.bar(
            x=[chr(i) for i in range(ord('a'), ord('z')+1)],
            y=[np.mean([a for a in st.session_state.user_progress['accuracy_history']]) if letter in st.session_state.user_progress['learned_letters'] else 0 for letter in 'abcdefghijklmnopqrstuvwxyz'],
            title="🎯 Accuracy by Letter",
            labels={'x': 'Letter', 'y': 'Accuracy (%)'}
        )
        fig_accuracy.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
        )
        st.plotly_chart(fig_accuracy, use_container_width=True)
        
        st.markdown("""
        ### 🏆 Achievement Progress
        """)
        for achievement in st.session_state.user_progress['achievements']:
            st.markdown(f"- {achievement}")

# Tab 6: About ISL
with tabs[5]:
    st.header("ℹ️ About Indian Sign Language")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### 🇮🇳 What is Indian Sign Language?
        
        Indian Sign Language (ISL) is the visual-spatial language used by the deaf community in India. 
        It's a complete, natural language with its own grammar and syntax, distinct from spoken Hindi or English.
        
        ### 🎯 Why Learn ISL?
        
        - **Inclusion**: Communicate with the deaf and hard-of-hearing community
        - **Cultural Awareness**: Understand deaf culture and perspectives  
        - **Professional Growth**: Career opportunities in interpretation and education
        - **Personal Development**: Enhance visual-spatial thinking and memory
        
        ### 📚 Learning Tips
        
        1. **Practice Regularly**: Consistency is key to muscle memory
        2. **Watch Native Signers**: Learn from the deaf community
        3. **Use Facial Expressions**: They're crucial for meaning
        4. **Be Patient**: Language learning takes time and practice
        5. **Immerse Yourself**: Join deaf community events when possible
        
        ### 🤝 Deaf Culture Etiquette
        
        - Make eye contact when signing
        - Tap gently on shoulder to get attention
        - Don't talk while someone is signing
        - Learn about deaf culture and history
        - Be respectful and patient
        """)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <h3>📱 App Features</h3>
            <ul>
                <li>🤖 AI-powered recognition</li>
                <li>📊 Progress tracking</li>
                <li>🎮 Interactive games</li>
                <li>📚 Complete reference</li>
                <li>🗣️ Common phrases</li>
                <li>🏆 Achievement system</li>
            </ul>
        </div>
        
        <div class="feature-card">
            <h3>🔧 Technical Info</h3>
            <ul>
                <li>Built with Streamlit</li>
                <li>PyTorch for AI models</li>
                <li>MediaPipe for hand detection</li>
                <li>OpenCV for image processing</li>
                <li>Plotly for analytics</li>
            </ul>
        </div>
        
        <div class="feature-card">
            <h3>📞 Support</h3>
            <p>Need help? Contact us:</p>
            <ul>
                <li>📧 support@islhub.com</li>
                <li>💬 Live chat available</li>
                <li>📱 WhatsApp: +91-XXXXX</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 2rem; background: #f8f9ff; border-radius: 10px; margin-top: 2rem;">
    <h3>🤟 Made with ❤️ for the Deaf Community</h3>
    <p>Empowering communication through technology | © 2024 ISL Learning Hub</p>
    <p>
        <a href="#" style="margin: 0 10px;">Privacy Policy</a> |
        <a href="#" style="margin: 0 10px;">Terms of Service</a> |
        <a href="#" style="margin: 0 10px;">Contact Us</a>
    </p>
</div>
""", unsafe_allow_html=True)