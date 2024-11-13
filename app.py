import os
from flask import Flask, render_template, request, redirect
from sklearn.decomposition import PCA
from skimage import io, color
from skimage.util import img_as_ubyte

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'

# Ensure the upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def compress_image(img, variance_ratio):
    gray_image = color.rgb2gray(img)
    flatten_img = gray_image.reshape(gray_image.shape[0], -1)
    pca = PCA(n_components=variance_ratio)
    compressed_data = pca.fit_transform(flatten_img)
    reconstructed_image = pca.inverse_transform(compressed_data)
    normalized_image = (reconstructed_image - reconstructed_image.min()) / (reconstructed_image.max() - reconstructed_image.min())
    compressed_img_8bit = img_as_ubyte(normalized_image)
    return compressed_img_8bit

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # File upload
        if "file" not in request.files:
            return redirect(request.url)
        file = request.files["file"]
        if file.filename == "":
            return redirect(request.url)

        # Compression level from the form
        variance_ratio = float(request.form.get("compression")) / 100

        # Load image and compress
        img = io.imread(file)
        compressed_image = compress_image(img, variance_ratio)

        # Save original and compressed images
        original_path = os.path.join(app.config['UPLOAD_FOLDER'], "original.jpg")
        compressed_path = os.path.join(app.config['UPLOAD_FOLDER'], "compressed.jpg")
        io.imsave(original_path, img_as_ubyte(color.rgb2gray(img)))
        io.imsave(compressed_path, compressed_image)

        return render_template("index.html", original=original_path, compressed=compressed_path)
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True,host="0.0.0.0",port=5050)
