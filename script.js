/**
 * Deepfake Detector - Frontend JavaScript
 * ========================================
 * 
 * This script handles:
 * 1. Image upload (drag & drop and file selection)
 * 2. Image preview
 * 3. API communication with Flask backend
 * 4. Result display and animations
 */

// API Configuration
const API_BASE_URL = 'http://localhost:5000';

// DOM Elements
const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');
const previewContainer = document.getElementById('previewContainer');
const imagePreview = document.getElementById('imagePreview');
const fileName = document.getElementById('fileName');
const removeBtn = document.getElementById('removeBtn');
const detectBtn = document.getElementById('detectBtn');
const resultsSection = document.getElementById('resultsSection');
const resultCard = document.getElementById('resultCard');
const resultIcon = document.getElementById('resultIcon');
const resultLabel = document.getElementById('resultLabel');
const confidenceValue = document.getElementById('confidenceValue');
const confidenceFill = document.getElementById('confidenceFill');
const confidenceMarker = document.getElementById('confidenceMarker');
const detailsGrid = document.getElementById('detailsGrid');
const loadingOverlay = document.getElementById('loadingOverlay');

// Current selected file
let selectedFile = null;

/**
 * Initialize Event Listeners
 */
function initializeEventListeners() {
    // File input change
    fileInput.addEventListener('change', handleFileSelect);
    
    // Drop zone events
    dropZone.addEventListener('click', () => fileInput.click());
    dropZone.addEventListener('dragover', handleDragOver);
    dropZone.addEventListener('dragleave', handleDragLeave);
    dropZone.addEventListener('drop', handleDrop);
    
    // Remove button
    removeBtn.addEventListener('click', removeImage);
    
    // Detect button
    detectBtn.addEventListener('click', analyzeImage);
}

/**
 * Handle file selection from input
 */
function handleFileSelect(event) {
    const file = event.target.files[0];
    if (file) {
        processFile(file);
    }
}

/**
 * Handle drag over event
 */
function handleDragOver(event) {
    event.preventDefault();
    dropZone.classList.add('drag-over');
}

/**
 * Handle drag leave event
 */
function handleDragLeave(event) {
    event.preventDefault();
    dropZone.classList.remove('drag-over');
}

/**
 * Handle file drop
 */
function handleDrop(event) {
    event.preventDefault();
    dropZone.classList.remove('drag-over');
    
    const file = event.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) {
        processFile(file);
    } else {
        showError('Please drop a valid image file');
    }
}

/**
 * Process the selected file
 */
function processFile(file) {
    // Validate file type
    if (!file.type.startsWith('image/')) {
        showError('Please select a valid image file');
        return;
    }
    
    // Validate file size (max 10MB)
    if (file.size > 10 * 1024 * 1024) {
        showError('File size must be less than 10MB');
        return;
    }
    
    selectedFile = file;
    
    // Create preview
    const reader = new FileReader();
    reader.onload = (e) => {
        imagePreview.src = e.target.result;
        fileName.textContent = file.name;
        previewContainer.style.display = 'block';
        dropZone.style.display = 'none';
        detectBtn.disabled = false;
        
        // Hide previous results
        resultsSection.style.display = 'none';
    };
    reader.readAsDataURL(file);
}

/**
 * Remove selected image
 */
function removeImage(event) {
    event.stopPropagation();
    
    selectedFile = null;
    imagePreview.src = '';
    fileName.textContent = '';
    fileInput.value = '';
    
    previewContainer.style.display = 'none';
    dropZone.style.display = 'block';
    detectBtn.disabled = true;
    resultsSection.style.display = 'none';
}

/**
 * Analyze the image using the API
 */
async function analyzeImage() {
    if (!selectedFile) {
        showError('Please select an image first');
        return;
    }
    
    // Show loading
    loadingOverlay.style.display = 'flex';
    detectBtn.disabled = true;
    
    try {
        // Create form data
        const formData = new FormData();
        formData.append('image', selectedFile);
        
        // Send to API
        const response = await fetch(`${API_BASE_URL}/predict`, {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (data.success) {
            displayResults(data);
        } else {
            showError(data.error || 'Analysis failed');
        }
        
    } catch (error) {
        console.error('API Error:', error);
        showError('Could not connect to the server. Make sure the backend is running on localhost:5000');
    } finally {
        // Hide loading
        loadingOverlay.style.display = 'none';
        detectBtn.disabled = false;
    }
}

/**
 * Display analysis results
 */
function displayResults(data) {
    const { prediction, confidence, details } = data;
    
    // Show results section
    resultsSection.style.display = 'block';
    
    // Update result card
    resultCard.className = 'result-card ' + prediction.toLowerCase();
    
    if (prediction === 'REAL') {
        resultIcon.textContent = '✓';
        resultLabel.textContent = 'REAL IMAGE';
    } else {
        resultIcon.textContent = '⚠';
        resultLabel.textContent = 'LIKELY DEEPFAKE';
    }
    
    // Update confidence
    confidenceValue.textContent = confidence.toFixed(1);
    
    // Update confidence bar marker
    // For REAL: marker should be on left side (low fake probability)
    // For FAKE: marker should be on right side (high fake probability)
    let markerPosition;
    if (prediction === 'REAL') {
        markerPosition = ((100 - confidence) / 100) * 100;
    } else {
        markerPosition = (confidence / 100) * 100;
    }
    confidenceMarker.style.left = `${markerPosition}%`;
    
    // Display analysis details
    displayDetails(details);
    
    // Scroll to results
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

/**
 * Display detailed analysis
 */
function displayDetails(details) {
    detailsGrid.innerHTML = '';
    
    // Main metrics
    const metrics = [
        {
            label: 'Fake Probability',
            value: `${(details.fake_probability * 100).toFixed(1)}%`
        },
        {
            label: 'Indicators Found',
            value: `${details.indicators_found.toFixed(1)} / ${details.total_checks.toFixed(1)}`
        }
    ];
    
    // Add statistical metrics if available
    if (details.analysis && details.analysis.statistics) {
        const stats = details.analysis.statistics;
        
        metrics.push({
            label: 'Texture Quality',
            value: stats.laplacian_variance > 300 ? 'Normal' : 
                   stats.laplacian_variance > 100 ? 'Suspicious' : 'Anomalous'
        });
        
        metrics.push({
            label: 'Noise Pattern',
            value: (stats.noise_score > 50 && stats.noise_score < 500) ? 'Natural' : 'Unusual'
        });
    }
    
    // Add deep learning metrics if available
    if (details.analysis && details.analysis.deep_learning) {
        const dl = details.analysis.deep_learning;
        
        metrics.push({
            label: 'Feature Variance',
            value: dl.feature_std > 0.8 ? 'High (suspicious)' : 'Normal'
        });
    }
    
    // Create detail items
    metrics.forEach(metric => {
        const item = document.createElement('div');
        item.className = 'detail-item';
        item.innerHTML = `
            <div class="label">${metric.label}</div>
            <div class="value">${metric.value}</div>
        `;
        detailsGrid.appendChild(item);
    });
}

/**
 * Show error message
 */
function showError(message) {
    alert(message);
}

/**
 * Check server health on page load
 */
async function checkServerHealth() {
    try {
        const response = await fetch(`${API_BASE_URL}/health`);
        const data = await response.json();
        console.log('Server status:', data.status);
    } catch (error) {
        console.warn('Server not available. Make sure to start the Flask backend.');
    }
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    initializeEventListeners();
    checkServerHealth();
});
