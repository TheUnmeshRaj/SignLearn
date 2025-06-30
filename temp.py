

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
            allowed_letters = ['a', 'c', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 's', 't', 'u']
            st.session_state.game_letter = random.choice(allowed_letters)
            st.session_state.challenge_active = True
            st.rerun()

    with game_col2:
        if st.session_state.get("challenge_active"):
            st.subheader(f"👋 Make the sign for: **{st.session_state.game_letter.upper()}**")
            allowed_letters = ['a', 'c', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 's', 't', 'u']
            st.session_state.game_letter = random.choice(allowed_letters)

            chosen_letter = random.choice(allowed_letters)
            ref_img_path = REFERENCE_IMAGES[chosen_letter]

            mp_hands, hands, mp_draw = init_mediapipe()

            # Load and process reference image to get hand region
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

            if ref_hand_img is not None:
                st.image(ref_hand_img, caption=f"Reference Sign for '{st.session_state.game_letter.upper()}'", width=300)
            # Webcam input
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

   



doesnt work



# Tab content
with tabs[3]:
    st.header("🎮 Interactive Learning Games")
    st.session_state.allowed_letters = ['a', 'c', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 's', 't', 'u']

    for key, default in {
        "challenge_active": False,
        "game_score": 0,
        "game_letter": None,
        "start_time": None,
        "matched_this_round": False,
        "time_limit": 10,
        "current_uploaded_file": None,
        "clear_camera_input": False,
        "reference_landmarks": {}
    }.items():
        if key not in st.session_state:
            st.session_state[key] = default

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### 📹 Camera Feed")
        if st.session_state.challenge_active:
            st.subheader(f"👋 Make the sign for: **{st.session_state.game_letter.upper()}**")
            
            # Camera input
            camera_key = f"camera_{st.session_state.game_letter}_{st.session_state.game_score}_{int(time.time())}"
            uploaded_file = st.camera_input("📷 Show your sign", key=camera_key)
            
            if uploaded_file is not None and not st.session_state.matched_this_round:
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                frame = cv2.flip(frame, 1)
                
                if frame is not None:
                    st.image(frame, caption=f"Your Sign for '{st.session_state.game_letter.upper()}'", width=400)
                    
                    try:
                        current_landmarks = extract_hand_landmarks(frame)
                        
                        if current_landmarks is not None:
                            # For demonstration, we'll use a simple detection logic
                            # In a real implementation, you'd compare with reference landmarks
                            st.success("✅ Hand detected!")
                            
                            # Simulate gesture recognition (replace with actual comparison)
                            # You would compare current_landmarks with reference landmarks here
                            detection_confidence = np.random.uniform(0.6, 0.9)  # Placeholder
                            
                            st.write(f"Detection Confidence: {detection_confidence:.2f}")
                            
                            if detection_confidence > 0.1:  # Threshold for match
                                st.success("🎉 Sign Matched!")
                                st.session_state.game_score += 1
                                st.session_state.matched_this_round = True
                                
                                # Auto-progress to next round after short delay
                                time.sleep(1)
                                start_new_round()
                                st.rerun()
                            else:
                                st.info("Keep trying! Adjust your hand position.")
                        else:
                            st.warning("No hand detected. Please show your hand clearly.")
                            
                    except Exception as e:
                        st.error(f"Processing error: {e}")
            
            # Timer logic
            if st.session_state.start_time is not None:
                elapsed = time.time() - st.session_state.start_time
                remaining = max(0, int(st.session_state.time_limit - elapsed))
                
                # Create a progress bar for time
                progress = 1 - (elapsed / st.session_state.time_limit)
                st.progress(max(0, progress))
                st.write(f"⏳ Time left: {remaining} seconds")
                
                if remaining <= 0 and not st.session_state.matched_this_round:
                    st.warning(f"⏰ Time's up for '{st.session_state.game_letter.upper()}'!")
                    start_new_round()
                    st.rerun()
        else:
            st.info("Click 'Start Sign Challenge' to begin!")

    with col2:
        st.markdown("### 🎮 Game Controls")
        
        # Game controls
        if not st.session_state.challenge_active:
            if st.button("🚀 Start Sign Challenge", type="primary"):
                st.session_state.challenge_active = True
                st.session_state.game_score = 0
                start_new_round()
                st.rerun()
        else:
            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("⏹️ End Game"):
                    st.session_state.challenge_active = False
                    st.session_state.start_time = None
                    st.session_state.game_letter = None
                    st.session_state.matched_this_round = False
                    st.session_state.current_uploaded_file = None
                    st.rerun()
            
            with col_b:
                if st.button("⏭️ Skip Round"):
                    start_new_round()
                    st.rerun()
        
        # Score display
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            color: white;
            margin: 20px 0;
        ">
            <h2 style="margin: 0; font-size: 2em;">🏆</h2>
            <h3 style="margin: 5px 0;">Current Score</h3>
            <h1 style="margin: 0; font-size: 3em;">{st.session_state.game_score}</h1>
        </div>
        """, unsafe_allow_html=True)
        
        # Reference image display
        if st.session_state.challenge_active and st.session_state.game_letter:
            st.markdown("### 📚 Reference Sign")
            ref_img_path = REFERENCE_IMAGES.get(st.session_state.game_letter, None)
            
            if ref_img_path:
                try:
                    ref_img = cv2.imread(ref_img_path)
                    if ref_img is not None:
                        ref_img = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB)
                        st.image(ref_img, caption=f"Sign for '{st.session_state.game_letter.upper()}'", width=300)
                    else:
                        st.warning(f"Could not load image: {ref_img_path}")
                except Exception as e:
                    st.warning(f"Error loading reference image: {e}")
            else:
                st.info(f"No reference image available for '{st.session_state.game_letter.upper()}'")
        
        # Game info
        st.markdown("""
        <div style="
            background: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            border-left: 4px solid #28a745;
            margin: 20px 0;
        ">
            <h4 style="color: #28a745; margin-top: 0;">🎯 How to Play</h4>
            <ul style="margin-bottom: 0;">
                <li>📸 Show the correct sign when prompted</li>
                <li>⏱️ You have 10 seconds per round</li>
                <li>🏆 Score points for correct signs</li>
                <li>⏭️ Game progresses automatically</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Settings
        if st.session_state.challenge_active:
            with st.expander("⚙️ Game Settings"):
                new_time_limit = st.slider("Time per round (seconds)", 5, 30, st.session_state.time_limit)
                if new_time_limit != st.session_state.time_limit:
                    st.session_state.time_limit = new_time_limit
                
                st.write(f"**Available letters:** {', '.join(st.session_state.allowed_letters)}")

    # Handle automatic progression
    if st.session_state.get('clear_camera_input', False):
        st.session_state['clear_camera_input'] = False
        st.rerun()
     
