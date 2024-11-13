import numpy as np
import streamlit as st
from sklearn.decomposition import PCA
from skimage import color, img_as_ubyte
from PIL import Image
import io as io_module

# Image compression function
def compress_image(img):
    gray_image = color.rgb2gray(img)
    flattened_img = gray_image.reshape(gray_image.shape[0], -1)
    pc = PCA(n_components=0.9)  # Adjust this if you want more or less compression
    compressed_data = pc.fit_transform(flattened_img)
    reconstructed_image = pc.inverse_transform(compressed_data)
    
    # Normalize for display
    compressed_data_normalize = (
        (reconstructed_image - reconstructed_image.min()) / 
        (reconstructed_image.max() - reconstructed_image.min())
    )
    compressed_img_bit = img_as_ubyte(compressed_data_normalize)
    return compressed_img_bit

# Streamlit App
st.title("Image Compression App using PCA")

# Image upload
uploaded_image = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # Read and display the original image
    img = Image.open(uploaded_image)
    st.image(img, caption="Original Image", use_container_width=True)

    # Convert to numpy array
    img_np = np.array(img)

    # Compress the image
    compressed_img_np = compress_image(img_np)

    # Convert back to displayable format
    compressed_img = Image.fromarray(compressed_img_np)

    # Display the compressed image
    st.image(compressed_img, caption="Compressed Image", use_container_width=True)

    # Download option for the compressed image
    buf = io_module.BytesIO()
    compressed_img.save(buf, format="JPEG")
    byte_img = buf.getvalue()

    st.download_button(
        label="Download Compressed Image",
        data=byte_img,
        file_name="compressed_image.jpg",
        mime="image/jpeg"
    )
