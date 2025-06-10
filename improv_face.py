import cv2
import streamlit as st
import numpy as np

# Initialize cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# User controls in sidebar
with st.sidebar:
    rect_color = st.color_picker("Rectangle Color", "#00FF00")
    min_neighbors = st.slider("minNeighbors", 1, 10, 5)
    scale_factor = st.slider("scaleFactor", 1.1, 2.0, 1.3, 0.1)
    save_image = st.checkbox("Save detected faces")

def detect_faces():
    FRAME_WINDOW = st.image([])
    stop_button = st.button("Stop Detection")
    
    # Convert hex color to BGR
    color = tuple(int(rect_color.lstrip('#')[i:i+2], 16) for i in (4, 2, 0))
    
    cap = cv2.VideoCapture(0)
    while cap.isOpened() and not stop_button:
        ret, frame = cap.read()
        if not ret:
            st.warning("Can't receive frame")
            break
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 
                                            scaleFactor=scale_factor, 
                                            minNeighbors=min_neighbors)
        
        for i, (x, y, w, h) in enumerate(faces):
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            if save_image:
                face_img = frame[y:y+h, x:x+w]
                cv2.imwrite(f"face_{i}.jpg", face_img)
        
        FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
    cap.release()
    if save_image and len(faces) > 0:
        st.success(f"Saved {len(faces)} face images")

def app():
    st.title("Face Detection using Viola-Jones Algorithm")
    st.write("Press the button below to start detecting faces from your webcam")
    
    if st.button("Start Face Detection"):
        detect_faces()

if __name__ == "__main__":
    app()