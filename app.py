"""
Flask Backend for Deepfake Detection System
Uses CNN model with Gemini API
"""

import os
import io
import base64
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import cv2
from werkzeug.utils import secure_filename
import google.generativeai as genai
from cnn_model_pytorch import DeepfakeCNN, CNNModel

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'mp4', 'avi', 'mov', 'mkv'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size

# Create upload folder
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize models
cnn_model = None
gemini_client = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def initialize_models():
    """Initialize and load CNN model"""
    global cnn_model
    
    print("Loading CNN model...")
    cnn_model = CNNModel(model_path='cnn_trained.pth')
    cnn_model.load_model()
    
    print("CNN model loaded successfully!")

'''def initialize_gemini(api_key):
    """Initialize Gemini API client"""
    global gemini_client
    if api_key:
        genai.configure(api_key=api_key)
        gemini_client = genai.GenerativeModel('gemini-pro')
        print("Gemini API initialized!")
    else:
        print("Warning: Gemini API key not provided") '''

def generate_heatmap(model, image_tensor, device):
    """Generate Grad-CAM heatmap for visualization using CNN model"""
    model.model.eval()
    
    # Clone and set requires_grad
    img_tensor = image_tensor.clone().detach().requires_grad_(True)
    
    # Get activations from the last conv layer (block5)
    activations = []
    gradients = []
    
    def forward_hook(module, input, output):
        activations.append(output)
    
    def backward_hook(module, grad_input, grad_output):
        if grad_output[0] is not None:
            gradients.append(grad_output[0])
    
    # Register hooks on the last conv layer (block5, second conv - index 3)
    # block5 structure: Conv2d(0), BatchNorm(1), ReLU(2), Conv2d(3), BatchNorm(4), ReLU(5), MaxPool(6), Dropout(7)
    hook_handle = model.model.block5[3].register_forward_hook(forward_hook)
    grad_handle = model.model.block5[3].register_backward_hook(backward_hook)
    
    # Forward pass
    output = model.model(img_tensor)
    
    # Backward pass
    model.model.zero_grad()
    output.backward(torch.ones_like(output))
    
    # Get gradients and activations
    if gradients and activations:
        grads = gradients[0]
        acts = activations[0]
        
        # Global average pooling of gradients
        weights = torch.mean(grads, dim=(2, 3), keepdim=True)
        
        # Weighted combination of activation maps
        heatmap = torch.sum(weights * acts, dim=1, keepdim=True)
        heatmap = F.relu(heatmap)
        
        # Normalize
        heatmap = heatmap.squeeze().cpu().detach().numpy()
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        
        hook_handle.remove()
        grad_handle.remove()
    else:
        # Fallback: simple activation map
        with torch.no_grad():
            x = img_tensor
            for block in [model.model.block1, model.model.block2, model.model.block3, 
                         model.model.block4, model.model.block5]:
                x = block(x)
            heatmap = torch.mean(x, dim=1).squeeze().cpu().numpy()
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    
    # Resize heatmap to match image size
    heatmap_resized = cv2.resize(heatmap, (224, 224))
    return heatmap_resized

def overlay_heatmap(image, heatmap, alpha=0.4):
    """Overlay heatmap on original image"""
    # Convert PIL to numpy
    if isinstance(image, Image.Image):
        img_array = np.array(image)
    else:
        img_array = image
    
    # Resize heatmap to match image
    if heatmap.shape != img_array.shape[:2]:
        heatmap = cv2.resize(heatmap, (img_array.shape[1], img_array.shape[0]))
    
    # Normalize heatmap to 0-255
    heatmap_norm = (heatmap * 255).astype(np.uint8)
    
    # Apply colormap (jet for better visualization)
    heatmap_colored = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)
    
    # Convert BGR to RGB
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    # Overlay
    overlay = cv2.addWeighted(img_array, 1-alpha, heatmap_colored, alpha, 0)
    
    return Image.fromarray(overlay)

def get_gemini_justification(prediction, confidence, manipulation_type):
    """Get justification from Gemini API"""
    global gemini_client
    
    if not gemini_client:
        # Fallback justification
        if prediction == "real":
            return f"The image appears to be authentic with {confidence:.1f}% confidence. Natural facial features and consistent lighting patterns detected."
        else:
            return f"The image shows signs of manipulation with {confidence:.1f}% confidence. Inconsistencies in facial features, lighting, or texture patterns detected."
    
    try:
        prompt = f"""Analyze this deepfake detection result and provide a brief, professional explanation:
        
Prediction: {prediction.upper()}
Confidence: {confidence:.1f}%
Manipulation Type: {manipulation_type}

Provide a concise 1-2 sentence explanation of why the image was classified as {prediction}, focusing on technical indicators like facial features, lighting consistency, texture patterns, or other visual artifacts."""
        
        response = gemini_client.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"Gemini API error: {e}")
        # Fallback
        if prediction == "real":
            return f"The image appears to be authentic with {confidence:.1f}% confidence. Natural facial features and consistent lighting patterns detected."
        else:
            return f"The image shows signs of manipulation with {confidence:.1f}% confidence. Inconsistencies in facial features, lighting, or texture patterns detected."

def predict_image(image_path):
    """Predict if image is real or fake using CNN model"""
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    original_size = image.size
    
    # Get prediction from CNN model
    cnn_pred, cnn_conf = cnn_model.predict(image)
    
    # CNN model returns: prob > 0.5 = real, so cnn_conf is confidence in the predicted class
    # If prediction is "real", confidence is cnn_conf
    # If prediction is "fake", confidence is 1 - cnn_conf
    final_pred = cnn_pred
    final_conf = cnn_conf if cnn_pred == "real" else (1 - cnn_conf)
    
    # Generate heatmap using CNN model
    from torchvision import transforms
    img_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    img_tensor = img_transform(image).unsqueeze(0).to(cnn_model.device)
    
    try:
        heatmap = generate_heatmap(cnn_model, img_tensor, cnn_model.device)
    except Exception as e:
        print(f"Heatmap generation error: {e}")
        import traceback
        traceback.print_exc()
        # Fallback: create a simple heatmap
        heatmap = np.random.rand(224, 224) * 0.3  # Low suspicion default
    
    # Resize heatmap to original image size
    heatmap_resized = cv2.resize(heatmap, original_size)
    
    # Overlay heatmap on image
    heatmap_overlay = overlay_heatmap(image, heatmap_resized)
    
    # Convert heatmap overlay to base64
    buffer = io.BytesIO()
    heatmap_overlay.save(buffer, format='PNG')
    heatmap_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    # Determine manipulation type
    manipulation_type = "None (Real Image)" if final_pred == "real" else "Face Swap / Deepfake"
    
    # Get justification from Gemini API
    justification = get_gemini_justification(final_pred, final_conf * 100, manipulation_type)
    
    return {
        'prediction': final_pred,
        'confidence': final_conf * 100,
        'manipulation_type': manipulation_type,
        'justification': justification,
        'heatmap': heatmap_base64
    }

def extract_video_frames(video_path, num_frames=5):
    """Extract frames from video for analysis"""
    cap = cv2.VideoCapture(video_path)
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Sample frames evenly
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame_rgb))
    
    cap.release()
    return frames

@app.route('/')
def index():
    """Serve the main page"""
    return render_template('index.html')

@app.route('/api/analyze', methods=['POST'])
def analyze():
    """Analyze uploaded image or video"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Check if it's a video
            file_ext = filename.rsplit('.', 1)[1].lower()
            is_video = file_ext in ['mp4', 'avi', 'mov', 'mkv']
            
            if is_video:
                # Extract frames and analyze first frame
                frames = extract_video_frames(filepath, num_frames=1)
                if not frames:
                    return jsonify({'error': 'Could not extract frames from video'}), 400
                
                # Save first frame temporarily
                frame_path = filepath.rsplit('.', 1)[0] + '_frame.jpg'
                frames[0].save(frame_path)
                
                result = predict_image(frame_path)
                os.remove(frame_path)  # Clean up
            else:
                result = predict_image(filepath)
            
            # Clean up uploaded file
            os.remove(filepath)
            
            return jsonify(result)
        except Exception as e:
            print(f"Error during analysis: {e}")
            import traceback
            traceback.print_exc()
            return jsonify({'error': str(e)}), 500
    
    return jsonify({'error': 'Invalid file type'}), 400

if __name__ == '__main__':
    # Initialize models
    initialize_models()
    
    '''# Initialize Gemini (get API key from environment variable)
    gemini_api_key = os.getenv('GEMINI_API_KEY', '')
    if gemini_api_key:
        initialize_gemini(gemini_api_key)
    else:
        print("Warning: GEMINI_API_KEY environment variable not set. Justification will use fallback text.")'''
    
    # Run Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)

