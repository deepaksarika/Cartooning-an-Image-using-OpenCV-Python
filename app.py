import streamlit as st
import cv2
import numpy as np
from PIL import Image

# Function to load and process image
def cartoonize_image(image_path):
    # Read image
    img = cv2.imread(image_path)

    # Edges
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                  cv2.THRESH_BINARY, 9, 9)

    # Cartoonization
    color = cv2.bilateralFilter(img, 9, 250, 250)
    cartoon = cv2.bitwise_and(color, color, mask=edges)
    
    return img, edges, cartoon

# Streamlit app
st.title("Cartoonizer App")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert the file to an opencv image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)
    
    # Save uploaded image temporarily
    cv2.imwrite('temp_image.jpg', opencv_image)
    
    # Process the image
    img, edges, cartoon = cartoonize_image('temp_image.jpg')
    
    # Convert images to RGB for display
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    cartoon_rgb = cv2.cvtColor(cartoon, cv2.COLOR_BGR2RGB)
    
    # Display images
    st.image(img_rgb, caption='Original Image', use_column_width=True)
    st.image(edges_rgb, caption='Edges', use_column_width=True)
    st.image(cartoon_rgb, caption='Cartoon Image', use_column_width=True)
