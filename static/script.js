// File upload and analysis functionality

const fileInput = document.getElementById('file-input');
const uploadBox = document.getElementById('upload-box');
const uploadPlaceholder = document.getElementById('upload-placeholder');
const previewImage = document.getElementById('preview-image');
const fileInfo = document.getElementById('file-info');
const fileName = document.getElementById('file-name');
const fileSize = document.getElementById('file-size');
const analyzeBtn = document.getElementById('analyze-btn');
const resetBtn = document.getElementById('reset-btn');
const resultsContent = document.getElementById('results-content');
const noResults = document.getElementById('no-results');

let selectedFile = null;

// Upload box click handler
uploadBox.addEventListener('click', () => {
    fileInput.click();
});

// Drag and drop handlers
uploadBox.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadBox.style.borderColor = '#8b5cf6';
    uploadBox.style.background = '#f3f4f6';
});

uploadBox.addEventListener('dragleave', (e) => {
    e.preventDefault();
    uploadBox.style.borderColor = '#cbd5e0';
    uploadBox.style.background = '#f7fafc';
});

uploadBox.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadBox.style.borderColor = '#cbd5e0';
    uploadBox.style.background = '#f7fafc';
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleFileSelect(files[0]);
    }
});

// File input change handler
fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        handleFileSelect(e.target.files[0]);
    }
});

// Handle file selection
function handleFileSelect(file) {
    selectedFile = file;
    
    // Check if it's an image
    if (file.type.startsWith('image/')) {
        const reader = new FileReader();
        reader.onload = (e) => {
            previewImage.src = e.target.result;
            previewImage.style.display = 'block';
            uploadPlaceholder.style.display = 'none';
        };
        reader.readAsDataURL(file);
    } else {
        // For videos, show placeholder
        uploadPlaceholder.innerHTML = `
            <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <polygon points="23 7 16 12 23 17 23 7"></polygon>
                <rect x="1" y="5" width="15" height="14" rx="2" ry="2"></rect>
            </svg>
            <p>Video file selected</p>
        `;
        previewImage.style.display = 'none';
    }
    
    // Show file info
    fileName.textContent = file.name;
    fileSize.textContent = formatFileSize(file.size);
    fileInfo.style.display = 'flex';
    
    // Enable analyze button
    analyzeBtn.disabled = false;
}

// Format file size
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
}

// Reset button handler
resetBtn.addEventListener('click', () => {
    selectedFile = null;
    fileInput.value = '';
    previewImage.src = '';
    previewImage.style.display = 'none';
    uploadPlaceholder.style.display = 'block';
    uploadPlaceholder.innerHTML = `
        <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
            <polyline points="17 8 12 3 7 8"></polyline>
            <line x1="12" y1="3" x2="12" y2="15"></line>
        </svg>
        <p>Click to upload or drag and drop</p>
    `;
    fileInfo.style.display = 'none';
    analyzeBtn.disabled = true;
    resultsContent.style.display = 'none';
    noResults.style.display = 'block';
});

// Analyze button handler
analyzeBtn.addEventListener('click', async () => {
    if (!selectedFile) return;
    
    // Show loading state
    analyzeBtn.disabled = true;
    analyzeBtn.textContent = 'Analyzing...';
    resultsContent.style.display = 'none';
    noResults.style.display = 'block';
    noResults.innerHTML = '<div class="loading">Analyzing file</div>';
    
    // Create form data
    const formData = new FormData();
    formData.append('file', selectedFile);
    
    try {
        const response = await fetch('/api/analyze', {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            throw new Error('Analysis failed');
        }
        
        const result = await response.json();
        
        // Display results
        displayResults(result);
        
    } catch (error) {
        console.error('Error:', error);
        noResults.innerHTML = `<p style="color: #ef4444;">Error: ${error.message}. Please try again.</p>`;
    } finally {
        analyzeBtn.disabled = false;
        analyzeBtn.textContent = 'Analyze File';
    }
});

// Display analysis results
function displayResults(result) {
    // Update result summary
    const resultIcon = document.getElementById('result-icon');
    const resultLabel = document.getElementById('result-label');
    const confidenceValue = document.getElementById('confidence-value');
    const manipulationBadge = document.getElementById('manipulation-badge');
    const justificationText = document.getElementById('justification-text');
    const heatmapImage = document.getElementById('heatmap-image');
    
    // Set prediction result
    const isReal = result.prediction === 'real';
    
    resultIcon.className = `result-icon ${isReal ? '' : 'fake'}`;
    resultIcon.innerHTML = `<span>${isReal ? '✓' : '✗'}</span>`;
    
    resultLabel.textContent = isReal ? 'Real' : 'Fake';
    resultLabel.className = `result-label ${isReal ? '' : 'fake'}`;
    
    confidenceValue.textContent = `${result.confidence.toFixed(1)}%`;
    
    // Update manipulation type
    manipulationBadge.textContent = result.manipulation_type;
    
    // Update justification
    justificationText.textContent = result.justification;
    
    // Update heatmap
    if (result.heatmap) {
        heatmapImage.src = `data:image/png;base64,${result.heatmap}`;
        heatmapImage.style.display = 'block';
    }
    
    // Show results
    resultsContent.style.display = 'block';
    noResults.style.display = 'none';
}

// Navigation handlers
document.getElementById('analyze-nav').addEventListener('click', (e) => {
    e.preventDefault();
    // Already on analyze page
});

document.getElementById('history-nav').addEventListener('click', (e) => {
    e.preventDefault();
    alert('History feature coming soon!');
});

