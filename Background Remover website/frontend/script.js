const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const editorUi = document.getElementById('editor-ui');
const resultImg = document.getElementById('result-image');
const imageStage = document.getElementById('image-stage');
const statusBadge = document.getElementById('status-badge');
const loaderUi = document.getElementById('loader-ui');

// Sliders
const brightnessInput = document.getElementById('brightness');
const contrastInput = document.getElementById('contrast');

let currentBgColor = 'transparent'; // Holds the actual color value

// 1. Upload Logic
dropZone.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', (e) => {
    if (e.target.files[0]) processImage(e.target.files[0]);
});

async function processImage(file) {
    dropZone.classList.add('hidden');
    
    // Show temp preview while loading
    const tempUrl = URL.createObjectURL(file);
    resultImg.src = tempUrl;
    
    editorUi.classList.remove('hidden');
    loaderUi.classList.remove('hidden');

    statusBadge.innerText = "⚡ Neural Network Active...";
    statusBadge.style.color = "#fbbf24";
    statusBadge.style.background = "rgba(251, 191, 36, 0.1)";

    const formData = new FormData();
    formData.append('file', file);

    try {
        // 🟢 THIS IS YOUR NGROK LINK (Verify this is still correct!)
        const response = await fetch('https://miller-ossiferous-onita.ngrok-free.dev/remove-bg/', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) throw new Error("Backend Error");

        const blob = await response.blob();
        
        // 🟢 IMPORTANT: Allow Cross-Origin for Canvas to work
        const url = URL.createObjectURL(blob);
        resultImg.crossOrigin = "anonymous"; 
        resultImg.src = url;
        
        loaderUi.classList.add('hidden'); 
        
        statusBadge.innerText = "✅ Extraction Complete";
        statusBadge.style.color = "#34d399";
        statusBadge.style.background = "rgba(16, 185, 129, 0.2)";

    } catch (error) {
        console.error(error);
        alert("Connection Error!\n1. Is python backend.py running?\n2. Is Ngrok running?");
        location.reload();
    }
}

// 2. Visual Filter Updates (Preview Only)
function updateFilters() {
    const b = brightnessInput.value;
    const c = contrastInput.value;
    // Apply CSS filters to the image for preview
    resultImg.style.filter = `brightness(${b}%) contrast(${c}%)`;
}

brightnessInput.addEventListener('input', updateFilters);
contrastInput.addEventListener('input', updateFilters);

// 3. Background Color Handling
function setBg(color) {
    currentBgColor = color; // Store for download
    imageStage.style.background = color; // Apply visually
    
    // Highlight selected button
    document.querySelectorAll('.color-btn').forEach(btn => btn.classList.remove('selected'));
    event.target.classList.add('selected');
}

// 4. Download Logic (The "Bake" Process)
function downloadImage() {
    // Create a virtual canvas to merge everything
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');

    // Set canvas to match the actual image size
    canvas.width = resultImg.naturalWidth;
    canvas.height = resultImg.naturalHeight;

    // A. Fill Background
    if (currentBgColor && currentBgColor !== 'transparent') {
        ctx.fillStyle = currentBgColor;
        ctx.fillRect(0, 0, canvas.width, canvas.height);
    }

    // B. Apply Filters (Brightness/Contrast)
    const b = brightnessInput.value;
    const c = contrastInput.value;
    ctx.filter = `brightness(${b}%) contrast(${c}%)`;

    // C. Draw the AI Image on top
    ctx.drawImage(resultImg, 0, 0);

    // D. Trigger Download
    const link = document.createElement('a');
    link.href = canvas.toDataURL('image/png');
    link.download = 'vision_result.png';
    link.click();
}

// 5. Reset
function resetApp() {
    location.reload();
}