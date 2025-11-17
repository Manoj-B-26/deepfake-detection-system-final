# Deepfake Detection System - TruthLens AI

A web-based deepfake detection system that uses CNN models and Gemini API to analyze images and videos for authenticity.

## Features

- **Image & Video Analysis**: Upload images or videos for deepfake detection
- **CNN Model**: Uses trained CNN model for classification
- **Confidence Scores**: Provides confidence percentage for predictions
- **Heatmap Visualization**: Shows suspicious regions in the image
- **Modern UI**: Beautiful, responsive web interface

## Installation

1. **Clone or navigate to the project directory**

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Ensure model files are in the root directory**:
   - `cnn_trained.pth` - Trained CNN model

## Usage

1. **Start the Flask server**:
```bash
python app.py
```

2. **Open your browser** and navigate to:
```
http://localhost:5000
```

3. **Upload an image or video**:
   - Click the upload area or drag and drop a file
   - Click "Analyze File" to start analysis
   - View results including prediction, confidence, justification, and heatmap

## Project Structure

```
dds_cursor/
├── app.py                      # Flask backend application
├── cnn_model_pytorch.py        # CNN model definition
├── cnn_trained.pth             # Trained CNN model weights
├── templates/
│   └── index.html              # Frontend HTML
├── static/
│   ├── style.css               # Frontend styles
│   └── script.js               # Frontend JavaScript
├── uploads/                    # Temporary upload directory (auto-created)
├── requirements.txt             # Python dependencies
└── README.md                   # This file
```

## API Endpoints

### POST `/api/analyze`
Analyzes an uploaded image or video file.

**Request**: 
- Form data with `file` field containing the image/video

**Response**:
```json
{
    "prediction": "real" | "fake",
    "confidence": 84.4,
    "manipulation_type": "None (Real Image)" | "Face Swap / Deepfake",
    "justification": "Detailed explanation...",
    "heatmap": "base64-encoded-image"
}
```

## Model Information

- **CNN Model**: Custom deep CNN architecture with 5 convolutional blocks
- **Input Size**: 224x224 pixels
- **Output**: Binary classification (Real/Fake) with confidence score

## Notes

- The system processes videos by extracting frames for analysis
- Heatmaps are generated using Grad-CAM visualization
- If Gemini API key is not provided, the system uses fallback justification text
- Maximum file size: 100MB

## Troubleshooting

1. **Model not loading**: Ensure `cnn_trained.pth` is in the root directory
2. **Heatmap not showing**: Check browser console for errors
3. **File upload issues**: Ensure file size is under 100MB and format is supported

## Supported Formats

- **Images**: PNG, JPG, JPEG, GIF
- **Videos**: MP4, AVI, MOV, MKV

## License

This project is for educational and research purposes.

