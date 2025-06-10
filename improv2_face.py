import cv2
import streamlit as st
import numpy as np
from threading import Thread

# Initialize cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize session state
if 'detection_active' not in st.session_state:
    st.session_state.detection_active = False
if 'frame' not in st.session_state:
    st.session_state.frame = None

def detection_thread():
    cap = cv2.VideoCapture(0)
    while st.session_state.detection_active:
        ret, frame = cap.read()
        if not ret:
            st.warning("Can't receive frame")
            break
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray, 
            scaleFactor=st.session_state.scale_factor, 
            minNeighbors=st.session_state.min_neighbors
        )
        
        color = tuple(int(st.session_state.rect_color.lstrip('#')[i:i+2], 16) for i in (4, 2, 0))
        
        for i, (x, y, w, h) in enumerate(faces):
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            if st.session_state.save_image:
                face_img = frame[y:y+h, x:x+w]
                cv2.imwrite(f"face_{i}.jpg", face_img)
        
        st.session_state.frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    cap.release()
    if hasattr(st.session_state, 'save_image') and st.session_state.save_image and len(faces) > 0:
        st.success(f"Saved {len(faces)} face images")

def app():
    st.title("Face Detection using Viola-Jones Algorithm")
    
    # Sidebar controls
    with st.sidebar:
        st.session_state.rect_color = st.color_picker("Rectangle Color", "#00FF00")
        st.session_state.min_neighbors = st.slider("minNeighbors", 1, 10, 5)
        st.session_state.scale_factor = st.slider("scaleFactor", 1.1, 2.0, 1.3, 0.1)
        st.session_state.save_image = st.checkbox("Save detected faces")
        
        if st.button("Start Detection") and not st.session_state.detection_active:
            st.session_state.detection_active = True
            Thread(target=detection_thread).start()
            
        if st.button("Stop Detection") and st.session_state.detection_active:
            st.session_state.detection_active = False
    
    # Main display
    if st.session_state.detection_active:
        placeholder = st.empty()
        while st.session_state.detection_active:
            if st.session_state.frame is not None:
                placeholder.image(st.session_state.frame)
    else:
        st.write("Press 'Start Detection' to begin")

if __name__ == "__main__":
    app()