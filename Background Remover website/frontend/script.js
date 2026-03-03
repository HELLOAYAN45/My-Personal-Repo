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

let currentBgColor = 'transparent';

// 1. Upload
dropZone.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', (e) => {
    if (e.target.files[0]) processImage(e.target.files[0]);
});

async function processImage(file) {
    dropZone.classList.add('hidden');
    
    // Show Preview immediately
    const tempUrl = URL.createObjectURL(file);
    resultImg.src = tempUrl;
    resultImg.classList.remove('hidden'); // 🟢 Show image now that it has data
    
    editorUi.classList.remove('hidden');
    loaderUi.classList.remove('hidden');

    statusBadge.innerText = "⚡ Processing High-Res Model...";
    statusBadge.style.color = "#fbbf24";
    statusBadge.style.background = "rgba(251, 191, 36, 0.1)";

    const formData = new FormData();
    formData.append('file', file);

    try {
        // 🟢 ENSURE THIS IS YOUR CURRENT NGROK LINK
        const response = await fetch('https://miller-ossiferous-onita.ngrok-free.dev/remove-bg/', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) throw new Error("Backend Error");

        const blob = await response.blob();
        
        const url = URL.createObjectURL(blob);
        resultImg.crossOrigin = "anonymous"; 
        resultImg.src = url;
        
        loaderUi.classList.add('hidden'); 
        
        statusBadge.innerText = "✅ HQ Extraction Complete";
        statusBadge.style.color = "#34d399";
        statusBadge.style.background = "rgba(16, 185, 129, 0.2)";

    } catch (error) {
        console.error(error);
        alert("Connection Error!\n1. Check Backend\n2. Check Ngrok");
        location.reload();
    }
}

// 2. Visual Filters
function updateFilters() {
    const b = brightnessInput.value;
    const c = contrastInput.value;
    resultImg.style.filter = `brightness(${b}%) contrast(${c}%)`;
}

brightnessInput.addEventListener('input', updateFilters);
contrastInput.addEventListener('input', updateFilters);

// 3. Background Color
function setBg(color) {
    currentBgColor = color;
    imageStage.style.background = color;
    
    document.querySelectorAll('.color-btn').forEach(btn => btn.classList.remove('selected'));
    event.target.classList.add('selected');
}

// 4. Download (Canvas Bake)
function downloadImage() {
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');

    // Use Natural size for HD download
    canvas.width = resultImg.naturalWidth;
    canvas.height = resultImg.naturalHeight;

    // A. Fill Background
    if (currentBgColor && currentBgColor !== 'transparent') {
        ctx.fillStyle = currentBgColor;
        ctx.fillRect(0, 0, canvas.width, canvas.height);
    }

    // B. Apply Filters
    const b = brightnessInput.value;
    const c = contrastInput.value;
    ctx.filter = `brightness(${b}%) contrast(${c}%)`;

    // C. Draw Image
    ctx.drawImage(resultImg, 0, 0);

    // D. Save
    const link = document.createElement('a');
    link.href = canvas.toDataURL('image/png');
    link.download = 'vision_hd_result.png';
    link.click();
}

function resetApp() {
    location.reload();
}//jfhffgfgf