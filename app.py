"""
Flask Backend for Deepfake Detection
=====================================
This server handles image uploads and returns predictions using our CNN model.

How it works:
1. Receives an image from the frontend via POST request
2. Preprocesses the image (resize, normalize)
3. Passes it through our CNN model
4. Returns prediction (Real/Fake) with confidence score
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import sys

# Add the backend directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model.deepfake_model import DeepfakeDetector

app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing for frontend communication

# Initialize the deepfake detector model
print("Loading Deepfake Detection Model...")
detector = DeepfakeDetector()
print("Model loaded successfully!")


@app.route('/health', methods=['GET'])
def health_check():
    """
    Health check endpoint to verify server is running.
    """
    return jsonify({
        'status': 'healthy',
        'message': 'Deepfake Detection API is running'
    })


@app.route('/predict', methods=['POST'])
def predict():
    """
    Main prediction endpoint.
    
    Accepts: Image file via multipart/form-data
    Returns: JSON with prediction and confidence score
    
    The CNN model analyzes the image for:
    - Facial artifacts (blurring around edges)
    - Texture inconsistencies (unnatural skin patterns)
    - Lighting anomalies (mismatched shadows)
    - Compression artifacts (JPEG noise patterns)
    """
    # Check if image was sent
    if 'image' not in request.files:
        return jsonify({
            'error': 'No image file provided',
            'success': False
        }), 400
    
    image_file = request.files['image']
    
    # Validate file
    if image_file.filename == '':
        return jsonify({
            'error': 'No image selected',
            'success': False
        }), 400
    
    # Check file extension
    allowed_extensions = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}
    file_ext = image_file.filename.rsplit('.', 1)[-1].lower()
    
    if file_ext not in allowed_extensions:
        return jsonify({
            'error': f'Invalid file type. Allowed: {", ".join(allowed_extensions)}',
            'success': False
        }), 400
    
    try:
        # Read image bytes
        image_bytes = image_file.read()
        
        # Get prediction from model
        result = detector.predict(image_bytes)
        
        return jsonify({
            'success': True,
            'prediction': result['prediction'],
            'confidence': result['confidence'],
            'details': result['details']
        })
        
    except Exception as e:
        return jsonify({
            'error': f'Error processing image: {str(e)}',
            'success': False
        }), 500


@app.route('/model-info', methods=['GET'])
def model_info():
    """
    Returns information about the model architecture and detection methods.
    Useful for educational purposes in your video.
    """
    return jsonify({
        'model_name': 'MobileNetV2-based Deepfake Detector',
        'architecture': 'Convolutional Neural Network (CNN)',
        'base_model': 'MobileNetV2 (pretrained on ImageNet)',
        'input_size': '224x224 pixels',
        'detection_features': [
            'Facial boundary artifacts',
            'Texture inconsistencies in skin regions',
            'Lighting and shadow anomalies',
            'Compression artifact patterns',
            'Color distribution abnormalities',
            'Frequency domain irregularities'
        ],
        'accuracy_note': 'This is a demonstration model. Production systems use larger datasets and ensemble methods.'
    })


if __name__ == '__main__':
    print("\n" + "="*50)
    print("DEEPFAKE DETECTION SERVER")
    print("="*50)
    print("Server starting on [localhost](http://localhost:5000)")
    print("Open the frontend/index.html in your browser")
    print("="*50 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
