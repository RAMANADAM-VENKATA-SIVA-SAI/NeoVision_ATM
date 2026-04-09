import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import time
import json
import os
import pandas as pd

# Data Persistence
DATA_FILE = "data.json"

def load_data():
    if not os.path.exists(DATA_FILE):
        return {}
    with open(DATA_FILE, 'r') as f:
        return json.load(f)

def save_data(data):
    with open(DATA_FILE, 'w') as f:
        json.dump(data, f, indent=4)

# Mediapipe Initializations
mp_face_mesh = mp.solutions.face_mesh
@st.cache_resource
def get_face_mesh():
    return mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

face_mesh = get_face_mesh()

# Eye Landmark Indices
LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]

def get_eye_aspect_ratio(landmarks, eye_indices):
    p2_p6 = np.linalg.norm(landmarks[eye_indices[12]] - landmarks[eye_indices[4]]) 
    p3_p5 = np.linalg.norm(landmarks[eye_indices[11]] - landmarks[eye_indices[5]]) 
    p1_p4 = np.linalg.norm(landmarks[eye_indices[8]] - landmarks[eye_indices[0]]) 
    ear = (p2_p6 + p3_p5) / (2.0 * p1_p4)
    return ear

def get_gaze_direction(landmarks, iris_indices, eye_indices):
    iris_center = np.mean(landmarks[iris_indices], axis=0)
    eye_left = landmarks[eye_indices[8]] 
    eye_right = landmarks[eye_indices[0]] 
    eye_width = np.linalg.norm(eye_left - eye_right)
    if eye_width == 0: return "Center"
    iris_pos = (iris_center[0] - eye_right[0]) / eye_width
    if iris_pos < 0.35: return "Right"
    elif iris_pos > 0.65: return "Left"
    else: return "Center"

def inject_custom_css():
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
        
        * {
            font-family: 'Outfit', sans-serif !important;
        }

        /* Force Main Background */
        .stApp, [data-testid="stAppViewContainer"], .main {
            background: linear-gradient(135deg, #10316b 0%, #0b7fab 100%) !important;
            color: #f5e3ba !important;
        }

        /* Force Header transparent */
        header[data-testid="stHeader"] {
            background: transparent !important;
        }
        
        /* Overriding all text colors */
        h1, h2, h3, h4, h5, h6, p, div, span, label {
            color: #f5e3ba !important;
        }

        /* Streamlit Alerts/Info boxes */
        [data-testid="stAlert"] {
            background-color: rgba(11, 127, 171, 0.4) !important;
            backdrop-filter: blur(10px);
            border-radius: 12px;
            border: 1px solid #0cd0ed !important;
            color: #f5e3ba !important;
        }

        .glass-card, [data-testid="column"] {
            background: rgba(16, 49, 107, 0.6);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border-radius: 16px;
            border: 1px solid rgba(12, 208, 237, 0.3);
            padding: 2rem;
            margin: 1rem 0;
            box-shadow: 0 4px 30px rgba(0, 0, 0, 0.3);
        }
        
        .premium-text {
            background: -webkit-linear-gradient(45deg, #0cd0ed 0%, #f5e3ba 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 700;
        }

        .menu-item {
            padding: 10px 15px;
            margin: 5px 0;
            border-radius: 8px;
            background: rgba(10, 31, 68, 0.5);
            border: 1px solid rgba(12, 208, 237, 0.1);
            transition: all 0.3s ease;
        }
        
        .menu-item-active {
            background: linear-gradient(90deg, #0cd0ed 0%, #0b7fab 100%);
            border-left: 4px solid #f5e3ba;
            transform: translateX(10px);
            box-shadow: 0 4px 15px rgba(12, 208, 237, 0.4);
        }

        /* Ensure active menu item text overrides general color rules for contrast */
        .menu-item-active h4, .menu-item-active span {
            color: #10316b !important;
            font-weight: bold;
        }
        
        div[data-testid="stMetricValue"] {
            font-size: 2.5rem !important;
            color: #0cd0ed !important;
        }
        </style>
    """, unsafe_allow_html=True)

def landing_page():
    inject_custom_css()
    st.markdown("""
        <div class="glass-card" style="text-align: center; margin-top: 10vh;">
            <h1 class="premium-text" style="font-size: 4rem;">🧠 NeoVision ATM</h1>
            <h3 style="font-weight: 300;">AI-Powered Multi-User Eye-Controlled Banking</h3>
            <p style="margin-top: 2rem; font-size: 1.2rem; color: #f5e3ba; opacity: 0.8;">
                Welcome to the future of banking. Fast, secure, and entirely touchless.<br/>
                Supporting over 150 members with advanced gaze tracking.
            </p>
            <hr style="border-color: rgba(245,227,186,0.2); margin: 2rem 0;"/>
            <div style="display: flex; justify-content: space-around; text-align: left; margin-bottom: 2rem;">
                <div>
                    <h4 class="premium-text">👥 150+ Members</h4>
                    <p>Secure multi-user data storage.</p>
                </div>
                <div>
                    <h4 class="premium-text">👁️ Eye Control</h4>
                    <p>Navigate effortlessly using gaze.</p>
                </div>
                <div>
                    <h4 class="premium-text">✨ Biometrics</h4>
                    <p>Blink to confirm actions securely.</p>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        if st.button("🚀 INITIATE ATM SESSION", use_container_width=True, type="primary"):
            st.session_state.started = True
            st.session_state.app_state = "ACCOUNT_SELECT"
            st.session_state.current_acc = None
            st.session_state.page_num = 0
            st.session_state.menu_index = 0
            st.rerun()

def atm_interface():
    inject_custom_css()
    if 'all_data' not in st.session_state:
        st.session_state.all_data = load_data()
    
    acc_list = list(st.session_state.all_data.keys())
    
    # Header
    st.markdown("<h2 class='premium-text'>NeoVision Control Center</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col2:
        st.subheader("👁️ Eye Monitor")
        video_placeholder = st.empty()
        status_placeholder = st.empty()

    with col1:
        # State management & menu definition
        if st.session_state.app_state == "ACCOUNT_SELECT":
            st.subheader("Select Account Profile")
            items_per_page = 5
            total_pages = (len(acc_list) + items_per_page - 1) // items_per_page
            start_idx = st.session_state.page_num * items_per_page
            end_idx = min(start_idx + items_per_page, len(acc_list))
            page_items = acc_list[start_idx:end_idx]
            
            current_menu = page_items + (["Next Page"] if st.session_state.page_num < total_pages - 1 else []) + \
                           (["Prev Page"] if st.session_state.page_num > 0 else []) + ["Exit"]
                           
        elif st.session_state.app_state == "MENU":
            user = st.session_state.all_data[st.session_state.current_acc]
            st.subheader(f"👋 Welcome, {user['account_holder']}")
            current_menu = ["Balance Enquiry", "Mini Statement", "Cash Withdrawal", "Cash Deposit", "Fund Transfer", "Analytics", "Logout", "Exit"]
            
        elif st.session_state.app_state == "BALANCE_VIEW":
            user = st.session_state.all_data[st.session_state.current_acc]
            st.subheader("💳 Account Overview")
            st.markdown(f"**Account Holder:** {user['account_holder']}  \n**Account Number:** {user['account_number']}")
            st.metric("Available Balance", f"₹ {user['balance']:,.2f}")
            current_menu = ["Back"]
            
        elif st.session_state.app_state == "MINI_STATEMENT":
            user = st.session_state.all_data[st.session_state.current_acc]
            st.subheader("📜 Last 5 Transactions")
            for tx in user['transactions'][:5]: # ensure we only show last 5
                icon = "🔻" if tx['type'] in ["Withdrawal", "Transfer Out"] else "🔺"
                color = "#ff4b4b" if tx['type'] in ["Withdrawal", "Transfer Out"] else "#00C9FF"
                st.markdown(f"""
                <div style='background: rgba(255,255,255,0.05); padding: 10px; border-radius: 8px; margin-bottom: 5px; display: flex; justify-content: space-between;'>
                    <span>{tx['date']} | {tx['type']}</span>
                    <span style='color: {color}; font-weight: bold;'>{icon} ₹{tx['amount']:,.2f}</span>
                </div>
                """, unsafe_allow_html=True)
            current_menu = ["Back"]
            
        elif st.session_state.app_state == "WITHDRAW_MENU":
            st.subheader("📤 Select Withdrawal Amount")
            current_menu = ["100", "500", "1000", "5000", "Back"]

        elif st.session_state.app_state == "DEPOSIT_MENU":
            st.subheader("📥 Select Deposit Amount")
            current_menu = ["100", "500", "1000", "5000", "Back"]
            
        elif st.session_state.app_state == "TRANSFER_SELECT_ACCOUNT":
            st.subheader("💸 Select Recipient Account")
            # Filter out current user
            transfer_acc_list = [acc for acc in acc_list if acc != st.session_state.current_acc]
            items_per_page = 5
            total_pages = (len(transfer_acc_list) + items_per_page - 1) // items_per_page
            start_idx = getattr(st.session_state, 'transfer_page_num', 0) * items_per_page
            end_idx = min(start_idx + items_per_page, len(transfer_acc_list))
            page_items = transfer_acc_list[start_idx:end_idx]
            
            current_menu = page_items + (["Next Page"] if getattr(st.session_state, 'transfer_page_num', 0) < total_pages - 1 else []) + \
                           (["Prev Page"] if getattr(st.session_state, 'transfer_page_num', 0) > 0 else []) + ["Back"]

        elif st.session_state.app_state == "TRANSFER_AMOUNT":
            recipient = st.session_state.all_data[st.session_state.transfer_recipient]
            st.subheader(f"Transferring to {recipient['account_holder']}")
            st.write("Select Amount:")
            current_menu = ["100", "500", "1000", "5000", "Back"]
            
        elif st.session_state.app_state == "ANALYTICS_VIEW":
            st.subheader("📊 Transaction Analytics")
            user = st.session_state.all_data[st.session_state.current_acc]
            if not user.get('transactions'):
                st.info("No transaction history available.")
            else:
                try:
                    df = pd.DataFrame(user['transactions'])
                    
                    # Convert 'date' to simple date string or datetime
                    df['date_only'] = df['date'].str.split(' ').str[0] 
                    
                    # Create withdrawal / deposit columns
                    df['Withdrawal'] = np.where(df['type'].isin(['Withdrawal', 'Transfer Out']), df['amount'], 0)
                    df['Deposit'] = np.where(df['type'].isin(['Deposit', 'Transfer In', 'Salary', 'Interest']), df['amount'], 0)
                    
                    # Group by date for chart
                    chart_data = df.groupby('date_only')[['Deposit', 'Withdrawal']].sum()
                    st.bar_chart(chart_data, color=["#00C9FF", "#ff4b4b"])
                except Exception as e:
                    st.error(f"Could not generate chart: {e}")
            current_menu = ["Back"]

        elif st.session_state.app_state == "SUCCESS":
            user = st.session_state.all_data[st.session_state.current_acc]
            st.success("✅ Transaction Successful!")
            st.metric("New Balance", f"₹ {user['balance']:,.2f}")
            current_menu = ["Back"]

        # Render Menu UI
        st.session_state.menu_index %= len(current_menu)
        for i, item in enumerate(current_menu):
            label = item
            if st.session_state.app_state == "ACCOUNT_SELECT" and item in acc_list:
                label = f"{st.session_state.all_data[item]['account_holder']} ({item[-4:]})"
            elif st.session_state.app_state == "TRANSFER_SELECT_ACCOUNT" and item in acc_list:
                label = f"Send to: {st.session_state.all_data[item]['account_holder']} ({item[-4:]})"
                
            if i == st.session_state.menu_index:
                st.markdown(f"<div class='menu-item menu-item-active'><h4>👉 {label}</h4></div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='menu-item'><span style='opacity: 0.7;'>{label}</span></div>", unsafe_allow_html=True)

        action_placeholder = st.empty()

    # Camera Loop
    if "cap" not in st.session_state or not st.session_state.cap.isOpened():
        st.session_state.cap = cv2.VideoCapture(0)
    
    cap = st.session_state.cap
    
    if "last_blink_time" not in st.session_state: st.session_state.last_blink_time = time.time()
    if "last_move_time" not in st.session_state: st.session_state.last_move_time = time.time()

    while st.session_state.started:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)
        gaze = "Center"
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                coords = np.array([[lm.x, lm.y, lm.z] for lm in face_landmarks.landmark])
                ear = (get_eye_aspect_ratio(coords, LEFT_EYE) + get_eye_aspect_ratio(coords, RIGHT_EYE)) / 2.0
                
                # Blink Detection
                if ear < 0.18 and (time.time() - st.session_state.last_blink_time > 1.5):
                    st.session_state.last_blink_time = time.time()
                    selection = current_menu[st.session_state.menu_index]
                    
                    if selection == "Exit":
                        st.session_state.started = False
                        st.session_state.cap.release()
                        st.rerun()
                    elif selection == "Back":
                        if st.session_state.app_state == "TRANSFER_AMOUNT":
                            st.session_state.app_state = "TRANSFER_SELECT_ACCOUNT"
                        else:
                            st.session_state.app_state = "MENU"
                        st.session_state.menu_index = 0
                    elif selection == "Logout":
                        st.session_state.app_state = "ACCOUNT_SELECT"
                        st.session_state.current_acc = None
                        st.session_state.menu_index = 0
                    elif selection == "Next Page":
                        if st.session_state.app_state == "TRANSFER_SELECT_ACCOUNT":
                            st.session_state.transfer_page_num = getattr(st.session_state, 'transfer_page_num', 0) + 1
                        else:
                            st.session_state.page_num += 1
                        st.session_state.menu_index = 0
                    elif selection == "Prev Page":
                        if st.session_state.app_state == "TRANSFER_SELECT_ACCOUNT":
                            st.session_state.transfer_page_num = max(0, getattr(st.session_state, 'transfer_page_num', 0) - 1)
                        else:
                            st.session_state.page_num = max(0, st.session_state.page_num - 1)
                        st.session_state.menu_index = 0
                    elif selection == "Balance Enquiry":
                        st.session_state.app_state = "BALANCE_VIEW"
                    elif selection == "Mini Statement":
                        st.session_state.app_state = "MINI_STATEMENT"
                    elif selection == "Cash Withdrawal":
                        st.session_state.app_state = "WITHDRAW_MENU"
                    elif selection == "Cash Deposit":
                        st.session_state.app_state = "DEPOSIT_MENU"
                    elif selection == "Fund Transfer":
                        st.session_state.app_state = "TRANSFER_SELECT_ACCOUNT"
                        st.session_state.transfer_page_num = 0
                    elif selection == "Analytics":
                        st.session_state.app_state = "ANALYTICS_VIEW"
                        
                    # Account Selection logic
                    elif st.session_state.app_state == "ACCOUNT_SELECT" and selection in acc_list:
                        st.session_state.current_acc = selection
                        st.session_state.app_state = "MENU"
                        st.session_state.menu_index = 0
                        
                    # Transfer Recipient Selection
                    elif st.session_state.app_state == "TRANSFER_SELECT_ACCOUNT" and selection in acc_list:
                        st.session_state.transfer_recipient = selection
                        st.session_state.app_state = "TRANSFER_AMOUNT"
                        st.session_state.menu_index = 0
                        
                    # Transaction execution (Withdraw, Deposit, Transfer Amount)
                    elif st.session_state.app_state in ["WITHDRAW_MENU", "DEPOSIT_MENU", "TRANSFER_AMOUNT"] and selection.isdigit():
                        amt = int(selection)
                        user = st.session_state.all_data[st.session_state.current_acc]
                        date_str = time.strftime("%Y-%m-%d %H:%M:%S")
                        
                        if st.session_state.app_state == "WITHDRAW_MENU":
                            if user['balance'] >= amt:
                                user['balance'] -= amt
                                user['transactions'].insert(0, {"date": date_str, "type": "Withdrawal", "amount": amt})
                                st.session_state.app_state = "SUCCESS"
                            else:
                                action_placeholder.error("Insufficient Funds!")
                                time.sleep(1)
                                
                        elif st.session_state.app_state == "DEPOSIT_MENU":
                            user['balance'] += amt
                            user['transactions'].insert(0, {"date": date_str, "type": "Deposit", "amount": amt})
                            st.session_state.app_state = "SUCCESS"
                            
                        elif st.session_state.app_state == "TRANSFER_AMOUNT":
                            if user['balance'] >= amt:
                                recipient = st.session_state.all_data[st.session_state.transfer_recipient]
                                user['balance'] -= amt
                                user['transactions'].insert(0, {"date": date_str, "type": "Transfer Out", "amount": amt})
                                recipient['balance'] += amt
                                recipient['transactions'].insert(0, {"date": date_str, "type": "Transfer In", "amount": amt})
                                st.session_state.app_state = "SUCCESS"
                            else:
                                action_placeholder.error("Insufficient Funds!")
                                time.sleep(1)
                                
                        save_data(st.session_state.all_data)
                        
                    st.rerun()

                # Gaze direction for navigation
                gaze = get_gaze_direction(coords, LEFT_IRIS, LEFT_EYE)
                if time.time() - st.session_state.last_move_time > 1.0:
                    if gaze == "Left":
                        st.session_state.menu_index = (st.session_state.menu_index - 1) % len(current_menu)
                        st.session_state.last_move_time = time.time()
                        st.rerun()
                    elif gaze == "Right":
                        st.session_state.menu_index = (st.session_state.menu_index + 1) % len(current_menu)
                        st.session_state.last_move_time = time.time()
                        st.rerun()

                # Draw Iris tracking standard dots
                for idx in LEFT_IRIS + RIGHT_IRIS:
                    lm = face_landmarks.landmark[idx]
                    cv2.circle(frame, (int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])), 2, (0, 255, 255), -1)

        # UI updates in video placeholder
        video_placeholder.image(frame, channels="BGR")
        status_html = f"""
        <div style='background: rgba(0,0,0,0.5); padding: 10px; border-radius: 8px;'>
            <div>👁️ Direction: <b>{gaze}</b></div>
            <div style='color: #00C9FF;'>🎯 Selected: <b>{current_menu[st.session_state.menu_index]}</b></div>
        </div>
        """
        status_placeholder.markdown(status_html, unsafe_allow_html=True)
        
        if not st.session_state.started: break
        time.sleep(0.01)

def main():
    st.set_page_config(page_title="NeoVision Multi-ATM", page_icon="🧠", layout="wide")
    if 'started' not in st.session_state: st.session_state.started = False
    if 'page_num' not in st.session_state: st.session_state.page_num = 0
    if not st.session_state.started: landing_page()
    else: atm_interface()

if __name__ == "__main__":
    main()
