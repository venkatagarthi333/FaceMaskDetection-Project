from flask import Flask, request, render_template, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

# Load the saved model
model = load_model('Model1.h5')

# Ensure the uploads directory exists
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def prepare_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Normalize the pixel values
    return img_array

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if an image was uploaded
        if 'file' not in request.files:
            return "No file part"
        file = request.files['file']
        if file.filename == '':
            return "No selected file"
        
        if file:
            file_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(file_path)
            
            # Preprocess the image and make a prediction
            img_array = prepare_image(file_path)
            prediction = model.predict(img_array)
            
            # Interpret the result
            if prediction[0][0] >= 0.5:
                result = f"Prediction: Not wearing a mask (Probability: {prediction[0][0]:.4f})"
            else:
                result = f"Prediction: Wearing a mask (Probability: {prediction[0][0]:.4f})"
            
            # Remove the uploaded file
            os.remove(file_path)
            
            # Redirect to the result page with the result as a query parameter
            return redirect(url_for('result', prediction=result))
    
    return render_template('index.html')

@app.route('/result')
def result():
    prediction = request.args.get('prediction')
    return render_template('result.html', result=prediction)

if __name__ == '__main__':
    app.run(debug=True)
