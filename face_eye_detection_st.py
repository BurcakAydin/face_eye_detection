import streamlit as st
import cv2 as cv
import numpy as np
from PIL import Image

# Ensure cascades are loaded correctly. It might be necessary to specify the full path if not found.
face_cascade_path = cv.data.haarcascades + 'haarcascade_frontalface_default.xml'
eye_cascade_path = cv.data.haarcascades + 'haarcascade_eye.xml'
face_cascade = cv.CascadeClassifier(face_cascade_path)
eye_cascade = cv.CascadeClassifier(eye_cascade_path)

def detect_faces_and_eyes(image_pil):
    """Detect faces and eyes in an image using OpenCV."""
    # Convert PIL image to a NumPy array for OpenCV to process
    image = np.array(image_pil.convert('RGB'))  # Convert to RGB if not already
    gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)  # Convert to grayscale for face detection

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        cv.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = image[y:y + h, x:x + w]

        # Detect eyes within face region
        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.3, minNeighbors=5)
        for (ex, ey, ew, eh) in eyes:
            cv.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

    # Convert the result back to a PIL image
    result_image_pil = Image.fromarray(image)
    return result_image_pil  # Return as PIL image for compatibility with Streamlit

# Streamlit UI setup
st.title("Face and Eye Detection")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    if st.button('Detect Faces and Eyes'):
        # Detect faces and eyes in the image
        result_image = detect_faces_and_eyes(image)
        st.image(result_image, caption='Processed Image', use_column_width=True)
