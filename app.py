import numpy as np
import streamlit as st
from sklearn.decomposition import PCA
from skimage import color, img_as_ubyte
from PIL import Image
import io as io_module

# Image compression function
def compress_image(img, compression_rate):
    # Ensure image is in RGB
    img_rgb = img.convert("RGB")
    
    # Convert to grayscale
    gray_image = color.rgb2gray(np.array(img_rgb))
    
    # Flatten the image
    flattened_img = gray_image.reshape(gray_image.shape[0], -1)
    
    # PCA compression with the chosen compression rate (in percentage)
    pc = PCA(n_components=compression_rate / 100)  # Convert to fraction (e.g., 90% -> 0.90)
    compressed_data = pc.fit_transform(flattened_img)
    
    # Reconstruct the image
    reconstructed_image = pc.inverse_transform(compressed_data)
    
    # Normalize for display
    compressed_data_normalize = (
        (reconstructed_image - reconstructed_image.min()) /
        (reconstructed_image.max() - reconstructed_image.min())
    )
    
    # Convert to uint8 (image format)
    compressed_img_bit = img_as_ubyte(compressed_data_normalize)
    
    return compressed_img_bit

# Streamlit App
st.title("Image Compression App using PCA")

# Image upload
uploaded_image = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # Show the image upload status and options
    # Select Compression Rate (percentage)
    compression_rate = st.selectbox(
        "Select Compression Rate:",
        [90, 95, 99],  # Compression rates in percentage
        index=0  # Default to 90% compression
    )

    # Show the selected compression percentage
    st.write(f"Selected Compression Rate: {compression_rate}%")

    # Read and display the original image
    img = Image.open(uploaded_image)
    st.image(img, caption="Original Image", use_container_width=True)

    # Compress the image
    compressed_img_np = compress_image(img, compression_rate)

    # Convert back to displayable format
    compressed_img = Image.fromarray(compressed_img_np)

    # Display the compressed image
    st.image(compressed_img, caption=f"Compressed Image at {compression_rate}%", use_container_width=True)

    # Download option for the compressed image
    buf = io_module.BytesIO()
    compressed_img.save(buf, format="JPEG")
    byte_img = buf.getvalue()

    st.download_button(
        label="Download Compressed Image",
        data=byte_img,
        file_name=f"compressed_image_{compression_rate}.jpg",
        mime="image/jpeg"
    )
