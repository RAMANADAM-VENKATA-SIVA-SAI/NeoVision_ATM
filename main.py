import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import time
import json
import os

# Data Persistence
DATA_FILE = "data.json"

def load_data():
    if not os.path.exists(DATA_FILE):
        return {} # Should be populated by generate_data.py
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

def landing_page():
    st.markdown("""
# 🧠 NeoVision ATM  
### AI-Powered Multi-User Eye-Controlled Banking

Welcome to NeoVision ATM – supporting over 150 members with touchless eye-controlled navigation.

---

### 👀 Features:

- **150+ Members**: Full multi-user support with secure data storage.
- **Eye-Controlled Selection**: Navigate between accounts using eye gaze.
- **Biometric Gestures**: Blink to confirm your selection or transaction.

---
""")
    if st.button("Start ATM"):
        st.session_state.started = True
        st.session_state.app_state = "ACCOUNT_SELECT"
        st.session_state.current_acc = None
        st.session_state.page_num = 0
        st.session_state.menu_index = 0
        st.rerun()

def atm_interface():
    if 'all_data' not in st.session_state:
        st.session_state.all_data = load_data()
    
    acc_list = list(st.session_state.all_data.keys())
    
    st.title("NeoVision ATM Control Center")
    col1, col2 = st.columns([2, 1])
    
    with col2:
        st.subheader("Eye Monitor")
        video_placeholder = st.empty()
        status_placeholder = st.empty()

    with col1:
        if st.session_state.app_state == "ACCOUNT_SELECT":
            st.subheader("Select Your Account")
            items_per_page = 5
            total_pages = (len(acc_list) + items_per_page - 1) // items_per_page
            
            start_idx = st.session_state.page_num * items_per_page
            end_idx = min(start_idx + items_per_page, len(acc_list))
            page_items = acc_list[start_idx:end_idx]
            
            # Add navigation options to current page items
            current_menu = page_items + (["Next Page"] if st.session_state.page_num < total_pages - 1 else []) + \
                           (["Prev Page"] if st.session_state.page_num > 0 else []) + ["Exit"]
            
        elif st.session_state.app_state == "MENU":
            user = st.session_state.all_data[st.session_state.current_acc]
            st.subheader(f"Welcome, {user['account_holder']}")
            current_menu = ["Balance Enquiry", "Mini Statement", "Cash Withdrawal", "Logout", "Exit"]
            
        elif st.session_state.app_state == "BALANCE_VIEW":
            user = st.session_state.all_data[st.session_state.current_acc]
            st.subheader("Account Details")
            st.write(f"**Name:** {user['account_holder']}")
            st.write(f"**Account:** {user['account_number']}")
            st.markdown(f"## Balance: ₹{user['balance']}")
            current_menu = ["Back"]
            
        elif st.session_state.app_state == "MINI_STATEMENT":
            user = st.session_state.all_data[st.session_state.current_acc]
            st.subheader("Last 5 Transactions")
            for tx in user['transactions']:
                color = "red" if tx['type'] == "Withdrawal" else "green"
                st.markdown(f"{tx['date']} | {tx['type']} | <span style='color:{color}'>₹{tx['amount']}</span>", unsafe_allow_html=True)
            current_menu = ["Back"]
            
        elif st.session_state.app_state == "WITHDRAW_MENU":
            st.subheader("Select Amount")
            current_menu = ["100", "500", "1000", "5000", "Back"]

        elif st.session_state.app_state == "SUCCESS":
            user = st.session_state.all_data[st.session_state.current_acc]
            st.subheader("Transaction Successful!")
            st.write(f"New Balance: ₹{user['balance']}")
            current_menu = ["Back"]

        # Render Menu
        st.session_state.menu_index %= len(current_menu)
        for i, item in enumerate(current_menu):
            label = item
            if st.session_state.app_state == "ACCOUNT_SELECT" and item in acc_list:
                label = f"{st.session_state.all_data[item]['account_holder']} ({item})"
            
            if i == st.session_state.menu_index:
                st.markdown(f"### 👉 **{label}**")
            else:
                st.markdown(f"#### {label}")

        st.divider()
        action_placeholder = st.empty()

    # Camera Loop
    cap = cv2.VideoCapture(0)
    last_blink_time = time.time()
    last_move_time = time.time()

    while st.session_state.started:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)
        status_text = "Monitoring..."
        gaze = "Center"
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                coords = np.array([[lm.x, lm.y, lm.z] for lm in face_landmarks.landmark])
                ear = (get_eye_aspect_ratio(coords, LEFT_EYE) + get_eye_aspect_ratio(coords, RIGHT_EYE)) / 2.0
                
                if ear < 0.18 and (time.time() - last_blink_time > 1.5):
                    last_blink_time = time.time()
                    selection = current_menu[st.session_state.menu_index]
                    
                    if selection == "Exit":
                        st.session_state.started = False
                        cap.release()
                        st.rerun()
                    elif selection == "Back":
                        st.session_state.app_state = "MENU"
                    elif selection == "Logout":
                        st.session_state.app_state = "ACCOUNT_SELECT"
                        st.session_state.current_acc = None
                    elif selection == "Next Page":
                        st.session_state.page_num += 1
                        st.session_state.menu_index = 0
                    elif selection == "Prev Page":
                        st.session_state.page_num -= 1
                        st.session_state.menu_index = 0
                    elif selection == "Balance Enquiry":
                        st.session_state.app_state = "BALANCE_VIEW"
                    elif selection == "Mini Statement":
                        st.session_state.app_state = "MINI_STATEMENT"
                    elif selection == "Cash Withdrawal":
                        st.session_state.app_state = "WITHDRAW_MENU"
                    elif st.session_state.app_state == "ACCOUNT_SELECT" and selection in acc_list:
                        st.session_state.current_acc = selection
                        st.session_state.app_state = "MENU"
                        st.session_state.menu_index = 0
                    elif st.session_state.app_state == "WITHDRAW_MENU" and selection.isdigit():
                        amt = int(selection)
                        user = st.session_state.all_data[st.session_state.current_acc]
                        if user['balance'] >= amt:
                            user['balance'] -= amt
                            user['transactions'].insert(0, {"date": time.strftime("%Y-%m-%d"), "type": "Withdrawal", "amount": amt})
                            user['transactions'] = user['transactions'][:5]
                            save_data(st.session_state.all_data)
                            st.session_state.app_state = "SUCCESS"
                        else:
                            action_placeholder.error("Insufficient Funds!")
                    
                    st.rerun()

                gaze = get_gaze_direction(coords, LEFT_IRIS, LEFT_EYE)
                if time.time() - last_move_time > 1.0:
                    if gaze == "Left":
                        st.session_state.menu_index = (st.session_state.menu_index - 1) % len(current_menu)
                        last_move_time = time.time()
                        st.rerun()
                    elif gaze == "Right":
                        st.session_state.menu_index = (st.session_state.menu_index + 1) % len(current_menu)
                        last_move_time = time.time()
                        st.rerun()

                for idx in LEFT_IRIS + RIGHT_IRIS:
                    lm = face_landmarks.landmark[idx]
                    cv2.circle(frame, (int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])), 1, (0, 255, 0), -1)

        video_placeholder.image(frame, channels="BGR")
        status_placeholder.write(f"Gaze: `{gaze}` | Selected: `{current_menu[st.session_state.menu_index]}`")
        if not st.session_state.started: break
        time.sleep(0.01)
    cap.release()

def main():
    st.set_page_config(page_title="NeoVision Multi-ATM", page_icon="🧠", layout="wide")
    if 'started' not in st.session_state: st.session_state.started = False
    if not st.session_state.started: landing_page()
    else: atm_interface()

if __name__ == "__main__":
    main()
