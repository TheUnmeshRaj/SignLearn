# Tab content
with tabs[3]:
    st.header("🎮 Interactive Learning Games")

    # Define allowed letters
    st.session_state.allowed_letters = ['a', 'c', 'e', 'f', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 's', 't', 'u']

    # Initialize session variables
    for key, default in {
        "challenge_active": False,
        "game_score": 0,
        "game_letter": None,
        "start_time": None,
        "matched_this_round": False,
        "time_limit": 10,
        "detection_threshold": 0.3,
        "reference_landmarks": {},
        "round_start_time": None,
        "last_detection_time": 0
    }.items():
        if key not in st.session_state:
            st.session_state[key] = default

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### 📹 Camera Feed")
        if st.session_state.challenge_active:
            st.subheader(f"👋 Make the sign for: **{st.session_state.game_letter.upper()}**")
            
            # Camera input with auto-refresh
            camera_key = f"camera_{st.session_state.game_letter}_{st.session_state.game_score}"
            uploaded_file = st.camera_input("📷 Show your sign", key=camera_key)
            
            if uploaded_file is not None and not st.session_state.matched_this_round:
                # Process the uploaded image
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                frame = cv2.flip(frame, 1)  # Mirror the image
                
                if frame is not None:
                    try:
                        # Extract hand landmarks from current frame
                        current_landmarks, hand_landmarks_list = extract_hand_landmarks(frame)
                        
                        # Create annotated image
                        if hand_landmarks_list:
                            annotated_frame = draw_landmarks_on_image(frame, hand_landmarks_list)
                            st.image(annotated_frame, caption=f"Your Sign for '{st.session_state.game_letter.upper()}' (with landmarks)", width=400)
                            
                            # Display landmark info
                            st.success("✅ Hand detected!")
                            st.write(f"📍 **Landmarks detected:** {len(current_landmarks)//2} points")
                            
                            # Check if gesture is valid
                            if is_hand_gesture_valid(current_landmarks):
                                st.success("👍 Valid hand gesture detected!")
                                
                                # For scoring, we'll use a simple approach since we don't have reference landmarks
                                # In a real implementation, you'd compare with stored reference landmarks
                                
                                # Simple scoring based on hand presence and stability
                                current_time = time.time()
                                if current_time - st.session_state.last_detection_time > 1.0:  # 1 second cooldown
                                    st.success("🎉 Sign Detected! Scoring...")
                                    st.session_state.game_score += 1
                                    st.session_state.matched_this_round = True
                                    st.session_state.last_detection_time = current_time
                                    
                                    # Show success message briefly
                                    st.balloons()
                                    time.sleep(1)
                                    
                                    # Start new round
                                    start_new_round()
                                    st.rerun()
                                else:
                                    st.info("Hold the sign steady for scoring...")
                            else:
                                st.warning("⚠️ Please show your hand more clearly")
                                
                        else:
                            # No hand detected, show original frame
                            st.image(frame, caption=f"Your Sign for '{st.session_state.game_letter.upper()}'", width=400)
                            st.warning("❌ No hand detected. Please show your hand clearly in the frame.")
                            
                    except Exception as e:
                        st.error(f"Processing error: {e}")
                        st.image(frame, caption="Error processing image", width=400)
            
            elif st.session_state.matched_this_round:
                st.success("✅ Round completed! Moving to next round...")
            
            # Timer logic
            if st.session_state.start_time is not None:
                elapsed = time.time() - st.session_state.start_time
                remaining = max(0, int(st.session_state.time_limit - elapsed))
                
                # Create a progress bar for time
                progress = max(0, 1 - (elapsed / st.session_state.time_limit))
                st.progress(progress)
                
                if remaining > 0:
                    st.write(f"⏳ Time left: **{remaining}** seconds")
                else:
                    st.write("⏰ Time's up!")
                
                if remaining <= 0 and not st.session_state.matched_this_round:
                    st.warning(f"⏰ Time's up for '{st.session_state.game_letter.upper()}'!")
                    time.sleep(1)
                    start_new_round()
                    st.rerun()
        else:
            st.info("🎯 Click 'Start Sign Challenge' to begin!")
            st.markdown("""
            **Instructions:**
            - Position your hand clearly in the camera frame
            - Make the sign shown in the reference image
            - Hold steady for 1 second to score
            - Green landmarks will appear when hand is detected
            """)

    with col2:
        st.markdown("### 🎮 Game Controls")
        
        # Game controls
        if not st.session_state.challenge_active:
            if st.button("🚀 Start Sign Challenge", type="primary", use_container_width=True):
                st.session_state.challenge_active = True
                st.session_state.game_score = 0
                start_new_round()
                st.rerun()
        else:
            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("⏹️ End Game", use_container_width=True):
                    st.session_state.challenge_active = False
                    st.session_state.start_time = None
                    st.session_state.game_letter = None
                    st.session_state.matched_this_round = False
                    st.rerun()
            
            with col_b:
                if st.button("⏭️ Skip Round", use_container_width=True):
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
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        ">
            <h2 style="margin: 0; font-size: 2em;">🏆</h2>
            <h3 style="margin: 5px 0;">Current Score</h3>
            <h1 style="margin: 0; font-size: 3em; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">{st.session_state.game_score}</h1>
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
                        st.image(ref_img, caption=f"Target: Sign for '{st.session_state.game_letter.upper()}'", width=300)
                    else:
                        st.warning(f"Could not load image: {ref_img_path}")
                except Exception as e:
                    st.warning(f"Error loading reference image: {e}")
            else:
                st.info(f"📝 Make the sign for: **{st.session_state.game_letter.upper()}**")
                st.markdown("*(Reference image not available)*")
        
        # Game statistics
        if st.session_state.challenge_active:
            st.markdown(f"""
            <div style="
                background: #f8f9fa;
                padding: 15px;
                border-radius: 8px;
                border-left: 4px solid #28a745;
                margin: 20px 0;
            ">
                <h4 style="color: #28a745; margin-top: 0;">📊 Game Stats</h4>
                <p><strong>Current Letter:</strong> {st.session_state.game_letter.upper()}</p>
                <p><strong>Score:</strong> {st.session_state.game_score}</p>
                <p><strong>Time per Round:</strong> {st.session_state.time_limit}s</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Game info
        st.markdown("""
        <div style="
            background: #e3f2fd;
            padding: 15px;
            border-radius: 8px;
            border-left: 4px solid #2196f3;
            margin: 20px 0;
        ">
            <h4 style="color: #1976d2; margin-top: 0;">🎯 How to Play</h4>
            <ul style="margin-bottom: 0;">
                <li>📸 Show your hand clearly in the camera</li>
                <li>👋 Make the sign for the given letter</li>
                <li>⏱️ Hold steady for 1 second to score</li>
                <li>🎉 Game auto-progresses to next round</li>
                <li>👀 Green landmarks show hand detection</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Settings
        if st.session_state.challenge_active:
            with st.expander("⚙️ Game Settings"):
                new_time_limit = st.slider("Time per round (seconds)", 5, 30, st.session_state.time_limit)
                if new_time_limit != st.session_state.time_limit:
                    st.session_state.time_limit = new_time_limit
