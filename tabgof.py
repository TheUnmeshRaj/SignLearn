with tabs[3]:
    st.header("🎮 Interactive Learning Games")

    if "game_score" not in st.session_state:
        st.session_state.game_score = 0
    if "challenge_active" not in st.session_state:
        st.session_state.challenge_active = False
    if "game_letter" not in st.session_state:
        st.session_state.game_letter = None

    allowed_letters = ['a', 'c', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 's', 't', 'u']

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### 📹 Live Camera Feed")
        if st.session_state.challenge_active:
            st.subheader(f"👋 Make the sign for: **{st.session_state.game_letter.upper()}**")

            ref_img_path = REFERENCE_IMAGES.get(st.session_state.game_letter, None)
            ref_hand_img = None

            if ref_img_path:
                mp_hands, hands, mp_draw = init_mediapipe()
                ref_bgr = cv2.imread(ref_img_path)
                ref_rgb = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2RGB)
                ref_results = hands.process(ref_rgb)

                if ref_results.multi_hand_landmarks:
                    for handLms in ref_results.multi_hand_landmarks:
                        x = [lm.x for lm in handLms.landmark]
                        y = [lm.y for lm in handLms.landmark]
                        h, w, _ = ref_rgb.shape
                        x1, x2 = int(min(x)*w)-20, int(max(x)*w)+20
                        y1, y2 = int(min(y)*h)-20, int(max(y)*h)+20
                        ref_hand_img = ref_rgb[max(y1,0):min(y2,h), max(x1,0):min(x2,w)]

            stframe = st.empty()
            cap = cv2.VideoCapture(0)
            success = False
            timeout = time.time() + 7

            while time.time() < timeout:
                ret, frame = cap.read()
                if not ret:
                    continue

                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb)

                if results.multi_hand_landmarks:
                    for handLms in results.multi_hand_landmarks:
                        x = [lm.x for lm in handLms.landmark]
                        y = [lm.y for lm in handLms.landmark]
                        h, w, _ = frame.shape
                        x1, x2 = int(min(x)*w)-20, int(max(x)*w)+20
                        y1, y2 = int(min(y)*h)-20, int(max(y)*h)+20
                        hand_img = frame[max(y1,0):min(y2,h), max(x1,0):min(x2,w)]

                        if hand_img.size != 0 and ref_hand_img is not None:
                            try:
                                hand_gray = cv2.cvtColor(hand_img, cv2.COLOR_BGR2GRAY)
                                ref_gray = cv2.cvtColor(ref_hand_img, cv2.COLOR_BGR2GRAY)
                                hand_resized = cv2.resize(hand_gray, (200, 200))
                                ref_resized = cv2.resize(ref_gray, (200, 200))

                                score, _ = ssim(hand_resized, ref_resized, full=True)

                                if score > 0.5:
                                    st.success("✅ Sign Matched!")
                                    st.session_state.game_score += 1
                                    success = True
                                    break

                            except Exception as e:
                                st.error(f"SSIM Error: {e}")

                stframe.image(frame, channels="BGR")
                if success:
                    break

            cap.release()

        else:
            st.info("🎯 Click 'Start Sign Challenge' to begin!")

    with col2:
        st.markdown("### 🎮 Game Controls")

        if not st.session_state.challenge_active:
            if st.button("🚀 Start Sign Challenge", use_container_width=True):
                st.session_state.challenge_active = True
                st.session_state.game_score = 0
                st.session_state.game_letter = random.choice(allowed_letters)
                st.rerun()
        else:
            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("⏹️ End Game", use_container_width=True):
                    st.session_state.challenge_active = False
                    st.session_state.game_letter = None
                    st.rerun()
            with col_b:
                if st.button("⏭️ Skip Round", use_container_width=True):
                    st.session_state.game_letter = random.choice(allowed_letters)
                    st.rerun()

        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    padding: 20px; border-radius: 10px; text-align: center; color: white;
                    margin: 20px 0; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">
            <h2>🏆</h2>
            <h3>Current Score</h3>
            <h1>{st.session_state.game_score}</h1>
        </div>
        """, unsafe_allow_html=True)

        if st.session_state.challenge_active:
            st.markdown("### 📚 Reference Sign")
            if ref_hand_img is not None:
                st.image(ref_hand_img, caption=f"Target: Sign for '{st.session_state.game_letter.upper()}'", width=300)

            st.markdown("""
            <div style="background: #e3f2fd; padding: 15px; border-radius: 8px;
                        border-left: 4px solid #2196f3; margin: 20px 0;">
                <h4 style="color: #1976d2;">🎯 How to Play</h4>
                <ul>
                    <li>📸 Show your hand clearly in the camera</li>
                    <li>👋 Make the sign for the given letter</li>
                    <li>⏱️ Hold steady for a few seconds to match</li>
                    <li>🎉 Game auto-progresses when matched</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)


