/**
 * Pneumonia AI — Frontend Application Logic
 * Handles image upload, API communication, results display, and history
 */

const API_BASE = '';  // Same origin

// ── DOM Elements ──
const uploadZone = document.getElementById('upload-zone');
const uploadInput = document.getElementById('upload-input');
const previewArea = document.getElementById('preview-area');
const previewThumb = document.getElementById('preview-thumb');
const previewFilename = document.getElementById('preview-filename');
const previewFilesize = document.getElementById('preview-filesize');
const analyzeBtn = document.getElementById('analyze-btn');
const resultsSection = document.getElementById('results-section');
const loadingOverlay = document.getElementById('loading-overlay');
const historyBody = document.getElementById('history-body');
const historyEmpty = document.getElementById('history-empty');
const toast = document.getElementById('toast');

let selectedFile = null;

// ── Upload Zone Events ──
uploadZone.addEventListener('click', () => uploadInput.click());

uploadZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadZone.classList.add('dragover');
});

uploadZone.addEventListener('dragleave', () => {
    uploadZone.classList.remove('dragover');
});

uploadZone.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadZone.classList.remove('dragover');
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleFileSelect(files[0]);
    }
});

uploadInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        handleFileSelect(e.target.files[0]);
    }
});

// ── File Selection ──
function handleFileSelect(file) {
    const validTypes = ['image/jpeg', 'image/png', 'image/bmp', 'image/tiff', 'image/webp'];
    if (!validTypes.includes(file.type) && !file.name.toLowerCase().endsWith('.dcm')) {
        showToast('Faqat rasm fayllari (JPEG, PNG, BMP, TIFF, WEBP) qabul qilinadi', 'error');
        return;
    }

    if (file.size > 50 * 1024 * 1024) {
        showToast('Fayl hajmi 50MB dan oshmasligi kerak', 'error');
        return;
    }

    selectedFile = file;

    // Show preview
    const reader = new FileReader();
    reader.onload = (e) => {
        previewThumb.src = e.target.result;
    };
    reader.readAsDataURL(file);

    previewFilename.textContent = file.name;
    previewFilesize.textContent = formatFileSize(file.size);
    previewArea.classList.add('active');
    analyzeBtn.disabled = false;

    // Hide previous results
    resultsSection.classList.remove('active');
}

// ── Analyze Button ──
analyzeBtn.addEventListener('click', async () => {
    if (!selectedFile) return;

    analyzeBtn.classList.add('loading');
    analyzeBtn.disabled = true;
    loadingOverlay.classList.add('active');

    try {
        const formData = new FormData();
        formData.append('file', selectedFile);

        const response = await fetch(`${API_BASE}/api/predict`, {
            method: 'POST',
            body: formData,
        });

        if (!response.ok) {
            const err = await response.json().catch(() => ({}));
            throw new Error(err.detail || `Server xatosi: ${response.status}`);
        }

        const result = await response.json();
        displayResults(result);
        showToast('Tahlil muvaffaqiyatli yakunlandi!', 'success');

        // Refresh history
        loadHistory();

    } catch (error) {
        console.error('Prediction error:', error);
        showToast(`Xato: ${error.message}`, 'error');
    } finally {
        analyzeBtn.classList.remove('loading');
        analyzeBtn.disabled = false;
        loadingOverlay.classList.remove('active');
    }
});

// ── Display Results ──
function displayResults(data) {
    resultsSection.classList.add('active');

    // Scroll to results
    setTimeout(() => {
        resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }, 100);

    // Diagnosis card
    const diagCard = document.getElementById('diagnosis-card');
    diagCard.className = `card diagnosis-card ${data.severity || 'normal'}`;

    // Badge
    const diagBadge = document.getElementById('diagnosis-badge');
    diagBadge.className = `diagnosis-badge ${data.prediction_label === 'PNEUMONIA' ? 'pneumonia' : 'normal'}`;
    diagBadge.textContent = data.prediction_label === 'PNEUMONIA' ? '⚠ PNEVMONIYA' : '✓ NORMAL';

    // Severity text
    document.getElementById('severity-text').textContent = data.severity_description || '';

    // Stats
    document.getElementById('stat-confidence').textContent = `${(data.confidence * 100).toFixed(1)}%`;
    document.getElementById('stat-confidence').style.color = data.severity_color || '#3b82f6';

    document.getElementById('stat-affected').textContent = `${(data.affected_area_percent || 0).toFixed(1)}%`;

    document.getElementById('stat-prob-normal').textContent = `${(data.prob_normal * 100).toFixed(1)}%`;
    document.getElementById('stat-prob-pneumonia').textContent = `${(data.prob_pneumonia * 100).toFixed(1)}%`;

    // Confidence bar
    const confFill = document.getElementById('confidence-fill');
    const confClass = data.prediction_label === 'PNEUMONIA' ? 'pneumonia' : 'normal';
    confFill.className = `confidence-fill ${confClass}`;
    setTimeout(() => {
        confFill.style.width = `${(data.confidence * 100).toFixed(1)}%`;
    }, 100);

    // Model info
    document.getElementById('model-name').textContent = data.model_name;

    // Images
    const originalImg = document.getElementById('original-image');
    const heatmapImg = document.getElementById('heatmap-image');

    originalImg.src = data.image_url;
    originalImg.style.display = 'block';

    if (data.heatmap_url) {
        heatmapImg.src = data.heatmap_url;
        heatmapImg.style.display = 'none';
        document.getElementById('tab-heatmap').style.display = 'block';
    } else {
        document.getElementById('tab-heatmap').style.display = 'none';
    }

    // Set original tab active
    setActiveTab('original');

    // Recommendations
    const recList = document.getElementById('rec-list');
    recList.innerHTML = '';

    if (data.recommendations && data.recommendations.length > 0) {
        data.recommendations.forEach((rec, i) => {
            const li = document.createElement('li');
            li.className = `rec-item${i === 0 && data.severity === "og'ir" ? ' urgent' : ''}`;
            li.innerHTML = `
                <span class="rec-icon">${getRecIcon(rec, data.severity)}</span>
                <span>${rec}</span>
            `;
            recList.appendChild(li);
        });
    }
}

function getRecIcon(text, severity) {
    if (text.includes('SHOSHILINCH') || text.includes('OG\'IR') || text.includes('ZUDLIK')) return '🚨';
    if (text.includes('tavsiya') || text.includes('Tavsiya')) return '💡';
    if (text.includes('tekshiruv') || text.includes('Tekshiruv')) return '🔬';
    if (text.includes('antibiotik') || text.includes('davolash')) return '💊';
    if (text.includes('stasionar') || text.includes('yotqizish')) return '🏥';
    if (text.includes('kislorod') || text.includes('ventilyatsiya')) return '🫁';
    if (text.includes('nazorat') || text.includes('kuzatuv')) return '📋';
    if (severity === 'normal') return '✅';
    return '📌';
}

// ── Image Tabs ──
function setActiveTab(tab) {
    const tabs = document.querySelectorAll('.image-tab');
    tabs.forEach(t => t.classList.remove('active'));
    document.getElementById(`tab-${tab}`).classList.add('active');

    const originalImg = document.getElementById('original-image');
    const heatmapImg = document.getElementById('heatmap-image');

    if (tab === 'original') {
        originalImg.style.display = 'block';
        heatmapImg.style.display = 'none';
    } else {
        originalImg.style.display = 'none';
        heatmapImg.style.display = 'block';
    }
}

document.getElementById('tab-original').addEventListener('click', () => setActiveTab('original'));
document.getElementById('tab-heatmap').addEventListener('click', () => setActiveTab('heatmap'));

// ── History ──
async function loadHistory() {
    try {
        const response = await fetch(`${API_BASE}/api/history?limit=20`);
        if (!response.ok) return;

        const data = await response.json();

        if (data.length === 0) {
            historyEmpty.style.display = 'block';
            historyBody.innerHTML = '';
            return;
        }

        historyEmpty.style.display = 'none';
        historyBody.innerHTML = '';

        data.forEach(item => {
            const row = document.createElement('tr');
            const date = new Date(item.created_at);
            const formattedDate = date.toLocaleString('uz-UZ', {
                year: 'numeric', month: '2-digit', day: '2-digit',
                hour: '2-digit', minute: '2-digit'
            });

            const labelClass = item.prediction_label === 'PNEUMONIA' ? 'pneumonia' : 'normal';
            const labelText = item.prediction_label === 'PNEUMONIA' ? 'Pnevmoniya' : 'Normal';

            let severityBadge = '';
            if (item.severity && item.severity !== 'normal') {
                const sevClass = item.severity === 'yengil' ? 'severity-yengil' :
                                 item.severity === "o'rta" ? 'severity-orta' : 'severity-ogir';
                severityBadge = `<span class="badge ${sevClass}">${item.severity}</span>`;
            }

            row.innerHTML = `
                <td>${formattedDate}</td>
                <td><span class="badge ${labelClass}">${labelText}</span></td>
                <td>${(item.confidence * 100).toFixed(1)}%</td>
                <td>${severityBadge || '—'}</td>
                <td>${item.model_name}</td>
            `;
            row.style.cursor = 'pointer';
            row.addEventListener('click', () => {
                displayResults(item);
                resultsSection.scrollIntoView({ behavior: 'smooth' });
            });
            historyBody.appendChild(row);
        });

    } catch (error) {
        console.error('History load error:', error);
    }
}

// ── Toast Notifications ──
function showToast(message, type = 'success') {
    toast.textContent = (type === 'success' ? '✓ ' : '✕ ') + message;
    toast.className = `toast ${type} show`;

    setTimeout(() => {
        toast.classList.remove('show');
    }, 4000);
}

// ── Utility ──
function formatFileSize(bytes) {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1048576) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / 1048576).toFixed(1) + ' MB';
}

// ── Initialize ──
document.addEventListener('DOMContentLoaded', () => {
    loadHistory();
});
