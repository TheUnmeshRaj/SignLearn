
from turtle import st
import random

with tabs[3]:
    st.header("🎮 Interactive Learning Games")
    allowed_letters = ['a', 'c', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 's', 't', 'u']
    
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
          
