import streamlit as st
from sklearn.decomposition import PCA
from skimage import io, color
from skimage.util import img_as_ubyte

# Function to compress image using PCA
def compress_image(img, variance_ratio):
  gray_image = color.rgb2gray(img)
  flatten_img = gray_image.reshape(gray_image.shape[0], -1)
  pca = PCA(n_components=variance_ratio)
  compressed_data = pca.fit_transform(flatten_img)
  reconstructed_image = pca.inverse_transform(compressed_data)
  normalized_image = (reconstructed_image - reconstructed_image.min()) / (reconstructed_image.max() - reconstructed_image.min())
  compressed_img_8bit = img_as_ubyte(normalized_image)
  return compressed_img_8bit

# Streamlit app
st.title("PCA Image Compression App")

# File upload
uploaded_file = st.file_uploader("Choose an image to compress")

# User input for compression level
compression_ratio = st.slider("Compression Ratio (0% - 100%)", min_value=0, max_value=100, step=1, value=50) / 100

if uploaded_file is not None:
  # Read the uploaded image
  img = io.imread(uploaded_file)

  # Perform compression
  compressed_image = compress_image(img, compression_ratio)

  # Display original and compressed images
  col1, col2 = st.columns(2)
  with col1:
    st.subheader("Original Image")
    st.image(img, caption="Original")
  with col2:
    st.subheader("Compressed Image")
    st.image(compressed_image, caption="Compressed")

  # Download buttons for original and compressed images (optional)
  # st.download_button("Download Original", img, mime="image/jpeg")
  # st.download_button("Download Compressed", compressed_image, mime="image/jpeg")

# Run the app
if __name__ == "__main__":
  st.balloons()  # Optional: Show a success message on startup