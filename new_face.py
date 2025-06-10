import cv2
import streamlit as st
import numpy as np

# Initialize cascade classifier with error handling
try:
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
except Exception as e:
    st.error(f"Error loading cascade file: {e}")
    st.stop()

def detect_faces():
    FRAME_WINDOW = st.image([])  # Streamlit image placeholder
    stop_button = st.button("Stop Detection")
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Could not open webcam")
        return
        
    while not stop_button:
        ret, frame = cap.read()
        if not ret:
            st.warning("Can't receive frame")
            break
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Convert BGR to RGB for Streamlit
        FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
    cap.release()
    st.success("Detection stopped")

def app():
    st.title("Face Detection using Viola-Jones Algorithm")
    st.write("Press the button below to start detecting faces from your webcam")
    
    if st.button("Start Face Detection"):
        detect_faces()

if __name__ == "__main__":
    app()

# Use the cv2.imwrite() function to save the images.
# Use the st.color_picker() function to allow the user to choose the color of the rectangles.
# Use the st.slider() function to allow the user to adjust the minNeighbors parameter.
# Use the st.slider() function to allow the user to adjust the scaleFactor parameter.