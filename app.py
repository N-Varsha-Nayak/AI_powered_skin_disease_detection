import os
import base64
from flask import Flask, render_template, request, send_file
from werkzeug.utils import secure_filename
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np
import cv2
from fpdf import FPDF
from datetime import datetime
import io

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Disease classes
CLASSES = [
    'Acne and Rosacea Photos',
    'Eczema Photos',
    'Melanoma Skin Cancer Nevi and Moles',
    'Hair Loss Photos Alopecia and other Hair Diseases',
    'Psoriasis pictures Lichen Planus and related diseases',
    'Normal'
]

# Load the trained model
model_path = 'models/best_model.h5'
if not os.path.exists(model_path):
    raise FileNotFoundError("Trained model not found. Please train the model first.")

model = tf.keras.models.load_model(model_path)


# ------------------------- GRAD-CAM ---------------------------- #

def generate_gradcam(img_array, model, last_conv_layer_name="conv5_block3_out"):
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)

        if isinstance(predictions, (list, tuple)):
            predictions = predictions[0]

        class_idx = tf.argmax(predictions[0])
        loss = tf.gather(predictions, indices=class_idx, axis=1)

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()


# ------------------------- IMAGE PROCESSING ---------------------------- #

def process_image(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x


# ------------------------- PDF GENERATION ---------------------------- #

def create_pdf(data):
    pdf = FPDF()
    pdf.add_page()

    # Title
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Skin Disease Diagnosis Report", ln=True, align="C")

    pdf.ln(8)
    pdf.set_font("Arial", size=12)

    # Patient info
    pdf.cell(0, 10, f"Name: {data['name']}", ln=True)
    pdf.cell(0, 10, f"Age: {data['age']}", ln=True)
    pdf.cell(0, 10, f"Gender: {data['gender']}", ln=True)
    pdf.cell(0, 10, f"Date: {data['date']}", ln=True)

    pdf.ln(5)

    # Prediction results
    pdf.cell(0, 10, f"Predicted Disease: {data['predicted_class']}", ln=True)
    pdf.cell(0, 10, f"Confidence: {data['confidence']}", ln=True)

    pdf.ln(10)

    # Original Image
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "Uploaded Image:", ln=True)
    try:
        pdf.image(data['image_path'], w=80)
    except:
        pdf.cell(0, 8, "Error loading image.", ln=True)

    pdf.ln(10)

    # Grad-CAM Image
    pdf.cell(0, 8, "Grad-CAM Heatmap:", ln=True)
    try:
        pdf.image(data['gradcam_path'], w=80)
    except:
        pdf.cell(0, 8, "Error loading heatmap.", ln=True)

    # Return PDF bytes
    pdf_bytes = pdf.output(dest="S").encode("latin-1")
    return io.BytesIO(pdf_bytes)


# ------------------------- ROUTES ---------------------------- #

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    captured_image_data = request.form.get('captured_image')

    # ---------------- Webcam image ---------------- #
    if captured_image_data and captured_image_data.startswith('data:image'):
        header, encoded = captured_image_data.split(",", 1)
        image_data = base64.b64decode(encoded)

        filename = f"webcam_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        with open(filepath, 'wb') as f:
            f.write(image_data)

    # ---------------- File upload ---------------- #
    elif 'image' in request.files:
        file = request.files['image']
        if file.filename == "":
            return "No selected file", 400

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

    else:
        return "No image uploaded", 400

    # ---------------- Prediction ---------------- #
    img_array = process_image(filepath)
    predictions = model.predict(img_array)

    idx = np.argmax(predictions[0])
    confidence = float(predictions[0][idx])
    predicted_class = CLASSES[idx]

    # ---------------- Grad-CAM ---------------- #
    heatmap = generate_gradcam(img_array, model)

    img = cv2.imread(filepath)
    img = cv2.resize(img, (224, 224))

    heatmap = cv2.resize(heatmap, (224, 224))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    superimposed_img = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

    gradcam_filename = f"gradcam_{filename}"
    gradcam_filepath = os.path.join(app.config['UPLOAD_FOLDER'], gradcam_filename)
    cv2.imwrite(gradcam_filepath, superimposed_img)

    # ---------------- Patient details ---------------- #
    patient_data = {
        'name': request.form.get('name', 'Not provided'),
        'age': request.form.get('age', 'Not provided'),
        'gender': request.form.get('gender', 'Not provided'),
        'date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'predicted_class': predicted_class,
        'confidence': f"{confidence * 100:.2f}%",
        'image_path': filepath,
        'gradcam_path': gradcam_filepath
    }

    return render_template('result.html', **patient_data)


@app.route('/download_report', methods=['POST'])
def download_report():
    patient_data = {
        'name': request.form.get('name'),
        'age': request.form.get('age'),
        'gender': request.form.get('gender'),
        'date': request.form.get('date'),
        'predicted_class': request.form.get('predicted_class'),
        'confidence': request.form.get('confidence'),
        'image_path': request.form.get('image_path'),
        'gradcam_path': request.form.get('gradcam_path'),
    }

    pdf_stream = create_pdf(patient_data)

    return send_file(
        pdf_stream,
        download_name="skin_disease_report.pdf",
        as_attachment=True,
        mimetype="application/pdf"
    )


# ------------------------- MAIN ---------------------------- #

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080)
