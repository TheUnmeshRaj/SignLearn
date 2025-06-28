import json
import os
import sqlite3
import tempfile
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

# Enhanced Model Configurations
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

# Initialize session state
if 'user_progress' not in st.session_state:
    st.session_state.user_progress = {
        'learned_letters': set(),
        'practice_sessions': 0,
        'total_time': 0,
        'streak_days': 0,
        'achievements': [],
        'accuracy_history': [],
        'last_session': None
    }

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
    """Load and cache the PyTorch model"""
    model = models.resnet18(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def init_mediapipe():
    """Initialize MediaPipe hands detection"""
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
    """Update user progress tracking"""
    st.session_state.user_progress['learned_letters'].add(letter)
    st.session_state.current_session['letters_practiced'].add(letter)
    st.session_state.current_session['predictions'].append({
        'letter': letter,
        'accuracy': accuracy,
        'timestamp': datetime.now()
    })

def calculate_achievements():
    """Calculate and award achievements"""
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
    
    # Progress Stats
    learned_count = len(st.session_state.user_progress['learned_letters'])
    progress_percentage = (learned_count / 26) * 100
    
    st.markdown(f"""
    <div class="stats-container">
        <h3>📊 Your Progress</h3>
        <p><strong>{learned_count}/26</strong> Letters Learned</p>
        <p><strong>{progress_percentage:.1f}%</strong> Complete</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Progress Bar
    st.progress(progress_percentage / 100)
    
    # Recent Achievements
    achievements = calculate_achievements()
    if achievements:
        st.markdown("### 🏆 Achievements")
        for achievement in achievements[-3:]:  # Show last 3
            st.markdown(f'<div class="achievement-badge">{achievement}</div>', unsafe_allow_html=True)
    
    # Learning Streak
    st.markdown(f"""
    <div class="sidebar-section">
        <h4>🔥 Learning Streak</h4>
        <p>{st.session_state.user_progress['streak_days']} days</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Quick Stats
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
        
        # Level Selection
        selected_level = st.selectbox(
            "Choose Your Level:",
            list(MODEL_CONFIGS.keys()),
            help="Select difficulty level based on your comfort"
        )
        
        level_config = MODEL_CONFIGS[selected_level]
        st.info(f"**{level_config['difficulty']}** - {level_config['description']}")
        
        # Load model
        model = load_model(level_config['path'], len(level_config['classes']))
        
        # Practice Mode Selection
        practice_mode = st.radio(
            "Practice Mode:",
            ["Free Practice", "Guided Learning", "Quiz Mode"],
            help="Choose how you want to practice"
        )
        
        # Settings
        st.subheader("⚙️ Settings")
        confidence_threshold = st.slider("Confidence Threshold", 0.5, 0.95, 0.7)
        show_landmarks = st.checkbox("Show Hand Landmarks", True)
        
        # Control Buttons
        col_start, col_stop = st.columns(2)
        with col_start:
            start_session = st.button("🚀 Start Practice", type="primary")
        with col_stop:
            stop_session = st.button("⏹️ Stop Session")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Current Target (for guided mode)
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
        
        # Reference Image Display
        reference_placeholder = st.empty()
        
        # Real-time Stats
        stats_col1, stats_col2, stats_col3 = st.columns(3)
        with stats_col1:
            accuracy_placeholder = st.empty()
        with stats_col2:
            prediction_placeholder = st.empty()
        with stats_col3:
            confidence_placeholder = st.empty()
    
    # Webcam Processing Logic
    if start_session:
        st.session_state.current_session['start_time'] = time.time()
        mp_hands, hands, mp_draw = init_mediapipe()
        
        # Webcam capture logic would go here
        st.info("🎥 Webcam feature requires local environment. In production, this would show live video feed with real-time ISL recognition.")

# Tab 2: Alphabet Reference
with tabs[1]:
    st.header("📚 Complete ISL Alphabet Reference")
    st.markdown("Click on any letter to see its sign and practice tips")
    
    # Create alphabet grid
    cols = st.columns(6)
    alphabet = 'abcdefghijklmnopqrstuvwxyz'
    
    for i, letter in enumerate(alphabet):
        with cols[i % 6]:
            if st.button(f"{letter.upper()}", key=f"letter_{letter}"):
                st.session_state.selected_letter = letter
    
    # Display selected letter info
    if 'selected_letter' in st.session_state:
        letter = st.session_state.selected_letter
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            <div class="reference-image">
                <h2>Letter: {letter.upper()}</h2>
                <p><em>Reference image would be displayed here</em></p>
            </div>
            """, unsafe_allow_html=True)
        
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
    
    # Category selection
    selected_category = st.selectbox("Choose Category:", list(PHRASES_CONFIG.keys()))
    
    phrases = PHRASES_CONFIG[selected_category]
    
    # Display phrases in grid
    phrase_cols = st.columns(2)
    
    for i, (phrase, config) in enumerate(phrases.items()):
        with phrase_cols[i % 2]:
            st.markdown(f"""
            <div class="feature-card">
                <h3>{phrase}</h3>
                <p>{config['description']}</p>
                <p><em>Video demonstration would be shown here</em></p>
            </div>
            """, unsafe_allow_html=True)
            
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
            st.rerun()
    
    # Game Implementation Placeholder
    if 'game_mode' in st.session_state:
        if st.session_state.game_mode == "challenge":
            st.info("🎮 Sign Challenge would be implemented here with real-time scoring")
        elif st.session_state.game_mode == "memory":
            st.info("🧩 Memory Match game would be implemented here")

# Tab 5: Progress Analytics
with tabs[4]:
    st.header("📊 Your Learning Analytics")
    
    # Generate sample data for demonstration
    sample_dates = [datetime.now() - timedelta(days=i) for i in range(30, 0, -1)]
    sample_progress = np.cumsum(np.random.randint(0, 3, 30))
    sample_accuracy = np.random.uniform(70, 95, 30)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Progress Chart
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
        
        # Weekly Stats
        st.markdown("""
        ### 📅 This Week's Stats
        - **Sessions Completed:** 12
        - **New Letters Learned:** 4
        - **Average Accuracy:** 87%
        - **Total Practice Time:** 2h 45min
        """)
    
    with col2:
        # Accuracy Chart
        fig_accuracy = px.bar(
            x=[chr(i) for i in range(ord('a'), ord('z')+1)],
            y=np.random.uniform(60, 100, 26),
            title="🎯 Accuracy by Letter",
            labels={'x': 'Letter', 'y': 'Accuracy (%)'}
        )
        fig_accuracy.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
        )
        st.plotly_chart(fig_accuracy, use_container_width=True)
        
        # Achievements Progress
        st.markdown("""
        ### 🏆 Achievement Progress
        - 🌟 **First Steps** ✅
        - 📚 **Alphabet Explorer** ✅  
        - 🏆 **Alphabet Master** (Progress: 85%)
        - 💪 **Dedicated Learner** ✅
        - 🔥 **Week Warrior** (Progress: 60%)
        """)

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
