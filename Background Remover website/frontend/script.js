const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const editorUi = document.getElementById('editor-ui');
const resultImg = document.getElementById('result-image');
const imageStage = document.getElementById('image-stage');
const statusBadge = document.getElementById('status-badge');
const brightnessInput = document.getElementById('brightness');
const loaderUi = document.getElementById('loader-ui');

// Upload Logic
dropZone.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', (e) => {
    if (e.target.files[0]) processImage(e.target.files[0]);
});

async function processImage(file) {
    // 1. Hide Upload Zone
    dropZone.classList.add('hidden');
    
    // 2. SHOW PREVIEW IMMEDIATELY
    // Create a temporary URL for the uploaded file so user sees it instantly
    const tempUrl = URL.createObjectURL(file);
    resultImg.src = tempUrl;
    
    // Show the editor and the loader overlay ON TOP of the image
    editorUi.classList.remove('hidden');
    loaderUi.classList.remove('hidden');

    // Update Status Badge
    statusBadge.innerText = "⚡ Neural Network Active...";
    statusBadge.style.color = "#fbbf24";
    statusBadge.style.background = "rgba(251, 191, 36, 0.1)";

    const formData = new FormData();
    formData.append('file', file);

    try {
        // --- 🟢 CRITICAL: This connects to your Laptop via Ngrok ---
        const response = await fetch('https://miller-ossiferous-onita.ngrok-free.dev/remove-bg/', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) throw new Error("Backend Error");

        const blob = await response.blob();
        
        // 3. SWAP PREVIEW WITH RESULT
        // Replace the "temp" original image with the AI cut-out result
        resultImg.src = URL.createObjectURL(blob);
        
        // Hide the loader overlay (reveal the result)
        loaderUi.classList.add('hidden'); 
        
        statusBadge.innerText = "✅ Extraction Complete";
        statusBadge.style.color = "#34d399";
        statusBadge.style.background = "rgba(16, 185, 129, 0.2)";

    } catch (error) {
        console.error(error);
        alert("Connection Error! \n\n1. Is your laptop running 'python backend.py'?\n2. Is the Ngrok window open?");
        location.reload();
    }
}

// Adjustments (Brightness)
brightnessInput.addEventListener('input', (e) => {
    resultImg.style.filter = `brightness(${e.target.value}%)`;
});

// Background Color Changer
function setBg(color) {
    imageStage.style.background = color;
}

// Download Action
function downloadImage() {
    const link = document.createElement('a');
    link.href = resultImg.src;
    link.download = 'vision_result.png';
    link.click();
}

// Reset Action
function resetApp() {
    location.reload();
}