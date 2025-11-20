app.py :
import os
import base64
from flask import Flask, render_template, request, send_file
from werkzeug.utils import secure_filename
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np
import cv2
from xhtml2pdf import pisa
from datetime import datetime
import io

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Disease classes
CLASSES = ['Acne and Rosacea Photos', 'Eczema Photos', 'Melanoma Skin Cancer Nevi and Moles', 'Hair Loss Photos Alopecia and other Hair Diseases', 'Psoriasis pictures Lichen Planus and related diseases','Normal']

# Load the trained model
model_path = 'models/best_model.h5'
if not os.path.exists(model_path):
    raise FileNotFoundError(
        "Trained model not found. Please run train_model.py first to train the model."
    )
model = tf.keras.models.load_model(model_path)

def generate_gradcam(img_array, model, last_conv_layer_name="conv5_block3_out"):
    grad_model = tf.keras.models.Model(
        [model.inputs], 
        [model.get_layer(last_conv_layer_name).output, model.output]
    )
    
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        # Handle possible list/tuple outputs from the model
        if isinstance(predictions, (list, tuple)):
            predictions = predictions[0]
        # Select top predicted class for the batch item
        class_idx = tf.argmax(predictions[0])
        # Gather the logit/probability of the selected class across the batch
        loss = tf.gather(predictions, indices=class_idx, axis=1)

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def process_image(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

def create_pdf(data, template_name):
    html = render_template(template_name, **data)
    pdf = io.BytesIO()
    pisa.CreatePDF(html, dest=pdf)
    pdf.seek(0)
    return pdf

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Check if we have a captured webcam image
    captured_image_data = request.form.get('captured_image')
    
    if captured_image_data and captured_image_data.startswith('data:image'):
        # Handle webcam capture - convert data URL to file
        from io import BytesIO
        
        # Extract the base64 data from the data URL
        header, encoded = captured_image_data.split(",", 1)
        image_data = base64.b64decode(encoded)
        
        # Create a file-like object
        file = BytesIO(image_data)
        file.filename = f"webcam_capture_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        
        # Save the captured image
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        with open(filepath, 'wb') as f:
            f.write(image_data)
            
    elif 'image' in request.files:
        # Handle regular file upload
        file = request.files['image']
        if file.filename == '':
            return 'No selected file', 400
            
        # Save the uploaded image
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
    else:
        return 'No image uploaded', 400

    # Process image and get prediction
    img_array = process_image(filepath)
    predictions = model.predict(img_array)
    predicted_class_idx = np.argmax(predictions[0])
    confidence = float(predictions[0][predicted_class_idx])
    predicted_class = CLASSES[predicted_class_idx]

    # Generate Grad-CAM
    heatmap = generate_gradcam(img_array, model)
    
    # Overlay heatmap on original image
    img = cv2.imread(filepath)
    img = cv2.resize(img, (224, 224))
    heatmap = cv2.resize(heatmap, (224, 224))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
    
    # Save the Grad-CAM image
    gradcam_filename = f"gradcam_{filename}"
    gradcam_filepath = os.path.join(app.config['UPLOAD_FOLDER'], gradcam_filename)
    cv2.imwrite(gradcam_filepath, superimposed_img)

    # Get patient information
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
        'gradcam_path': request.form.get('gradcam_path')
    }
    
    pdf = create_pdf(patient_data, 'pdf_template.html')
    return send_file(
        pdf,
        download_name='skin_disease_report.pdf',
        as_attachment=True,
        mimetype='application/pdf'
    )

if __name__ == '__main__':
    app.run(debug=True)
