import streamlit as st
import cv2 as cv
import numpy as np
from PIL import Image

# Load the cascades
face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_eye.xml')

def detect_faces_and_eyes(image_pil):
    """Detect faces and eyes in an image."""
    image = np.array(image_pil)  # Convert PIL image to NumPy array
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = image[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.3, 5)
        for (ex, ey, ew, eh) in eyes:
            cv.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
    return image  # This is now a NumPy array; you can return it directly for display


# Streamlit UI
st.title("Face and Eye Detection")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='Uploaded Image', use_column_width=True)
    if st.button('Detect Faces and Eyes'):
        result_image = detect_faces_and_eyes(image)
        st.image(result_image, caption='Processed Image', use_column_width=True)
