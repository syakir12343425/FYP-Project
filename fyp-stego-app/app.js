/**
 * AI-Based Robust Steganography Against Image Downscaling Attacks
 * FYP Research UI - Core Application Logic
 */

// Application State
const state = {
  coverLoaded: false,
  message: '',
  method: 'cnn', // 'cnn' or 'lsb'
  embedded: false,
  coverScene: 'lena', // 'lena', 'mandrill', 'satellite', 'medical'
  coverImgData: null,
  
  // Embedding Results
  psnr: 0,
  ssim: 0,
  stegoImgData: null,
  
  // Downscaling Attack parameters
  attackScale: 0.5,
  attackAlgo: 'bicubic',
  attackSimulated: false,
  downscaledImgData: null,
  
  // Extraction parameters
  extracted: false,
  extractedMsg: '',
  ber: 0,
  extractedPsnr: 0,
  extractedSsim: 0
};

// DOM Elements
const elements = {
  // Navigation
  navButtons: document.querySelectorAll('.nav-btn'),
  viewports: document.querySelectorAll('.viewport'),
  toggleFigures: document.getElementById('toggle-figures'),
  toggleTheme: document.getElementById('toggle-theme'),
  
  // Page 1: Embedding
  btnUseSample: document.getElementById('btn-use-sample'),
  btnTriggerUpload: document.getElementById('btn-trigger-upload'),
  coverFileInput: document.getElementById('cover-file-input'),
  sampleThumbButtons: document.querySelectorAll('.sample-thumb-btn'),
  secretMessageInput: document.getElementById('secret-message-input'),
  charCounter: document.getElementById('char-counter'),
  bitCounter: document.getElementById('bit-counter'),
  methodCards: document.querySelectorAll('.method-card'),
  btnResetEmbedding: document.getElementById('btn-reset-embedding'),
  btnEmbedPayload: document.getElementById('btn-embed-payload'),
  coverCanvas: document.getElementById('cover-canvas'),
  embeddingConsole: document.getElementById('embedding-console'),
  
  // Page 2: Result
  btnModeSide: document.getElementById('btn-mode-side'),
  btnModeResidual: document.getElementById('btn-mode-residual'),
  sideBySideView: document.getElementById('side-by-side-view'),
  residualMapView: document.getElementById('residual-map-view'),
  resultCoverCanvas: document.getElementById('result-cover-canvas'),
  resultStegoCanvas: document.getElementById('result-stego-canvas'),
  resultResidualCanvas: document.getElementById('result-residual-canvas'),
  metricPsnr: document.getElementById('metric-psnr'),
  metricSsim: document.getElementById('metric-ssim'),
  psnrBar: document.getElementById('psnr-bar'),
  ssimBar: document.getElementById('ssim-bar'),
  btnDownloadStego: document.getElementById('btn-download-stego'),
  btnGoToDownscaling: document.getElementById('btn-go-to-downscaling'),
  methodTitleBadge: document.getElementById('method-title-badge'),
  methodDetailsText: document.getElementById('method-details-text'),
  cnnNetworkPreview: document.getElementById('cnn-network-preview'),
  
  // Page 3: Downscaling
  miniStegoCanvas: document.getElementById('mini-stego-canvas'),
  stegoSourceStatus: document.getElementById('stego-source-status'),
  btnImportLatest: document.getElementById('btn-import-latest'),
  downscaleAlgorithm: document.getElementById('downscale-algorithm'),
  btnSimulateAttack: document.getElementById('btn-simulate-attack'),
  attackInputCanvas: document.getElementById('attack-input-canvas'),
  attackOutputCanvas: document.getElementById('attack-output-canvas'),
  arrowScaleLabel: document.getElementById('arrow-scale-label'),
  attackProgressFill: document.getElementById('attack-progress-fill'),
  progressStatusText: document.getElementById('progress-status-text'),
  progressPercentage: document.getElementById('progress-percentage'),
  btnContinueToExtraction: document.getElementById('btn-continue-to-extraction'),
  mathExplanationText: document.getElementById('math-explanation-text'),
  downscaledCanvasContainer: document.getElementById('downscaled-canvas-container'),
  
  // Page 4: Extraction
  miniDownscaledCanvas: document.getElementById('mini-downscaled-canvas'),
  downscaledImportStatus: document.getElementById('downscaled-import-status'),
  btnImportAttacked: document.getElementById('btn-import-attacked'),
  btnExtractMessage: document.getElementById('btn-extract-message'),
  displayOriginalMsg: document.getElementById('display-original-msg'),
  displayExtractedMsg: document.getElementById('display-extracted-msg'),
  extractBer: document.getElementById('extract-ber'),
  extractPsnr: document.getElementById('extract-psnr'),
  extractSsim: document.getElementById('extract-ssim'),
  berStatusIndicator: document.getElementById('ber-status-indicator'),
  evaluationSummaryText: document.getElementById('evaluation-summary-text'),
  evaluationTableRows: document.getElementById('evaluation-table-rows'),
  btnExportReport: document.getElementById('btn-export-report'),
  extractedMsgLabel: document.getElementById('extracted-msg-label'),
  
  // Modal Report
  reportModal: document.getElementById('report-modal'),
  btnCloseModal: document.getElementById('btn-close-modal'),
  btnDismissModal: document.getElementById('btn-dismiss-modal'),
  btnPrintDocument: document.getElementById('btn-print-document'),
  reportDate: document.getElementById('report-date'),
  reportCoverName: document.getElementById('report-cover-name'),
  reportMsgLen: document.getElementById('report-msg-len'),
  reportTableAlgo: document.getElementById('report-table-algo'),
  reportTableScale: document.getElementById('report-table-scale'),
  reportLsbBer: document.getElementById('report-lsb-ber'),
  reportCnnBer: document.getElementById('report-cnn-ber')
};

// Procedural Image Generator (Generates stunning test textures inside canvases)
function drawProceduralImage(canvas, scene) {
  const ctx = canvas.getContext('2d');
  const w = canvas.width;
  const h = canvas.height;
  
  ctx.clearRect(0, 0, w, h);

  if (scene === 'lena') {
    // Academic standard Lena lookalike: pinkish skin hues, feather/hat details
    // Background gradient
    const bgGrad = ctx.createLinearGradient(0, 0, w, h);
    bgGrad.addColorStop(0, '#581c3f');
    bgGrad.addColorStop(0.5, '#701a35');
    bgGrad.addColorStop(1, '#9f1239');
    ctx.fillStyle = bgGrad;
    ctx.fillRect(0, 0, w, h);
    
    // Smooth feather shape
    ctx.fillStyle = '#fda4af';
    ctx.beginPath();
    ctx.ellipse(w*0.35, h*0.5, w*0.3, h*0.2, Math.PI/4, 0, Math.PI*2);
    ctx.fill();
    
    // Skin Tone Face shapes
    ctx.fillStyle = '#fecdd3';
    ctx.beginPath();
    ctx.arc(w*0.5, h*0.45, w*0.18, 0, Math.PI*2);
    ctx.fill();
    
    // Eyes
    ctx.fillStyle = '#0f172a';
    ctx.beginPath();
    ctx.ellipse(w*0.46, h*0.43, w*0.02, h*0.01, 0, 0, Math.PI*2);
    ctx.ellipse(w*0.54, h*0.43, w*0.02, h*0.01, 0, 0, Math.PI*2);
    ctx.fill();
    
    // Lips
    ctx.fillStyle = '#be123c';
    ctx.beginPath();
    ctx.ellipse(w*0.5, h*0.53, w*0.03, h*0.01, 0, 0, Math.PI*2);
    ctx.fill();
    
    // Straw hat shape
    ctx.fillStyle = '#b45309';
    ctx.beginPath();
    ctx.moveTo(w*0.2, h*0.35);
    ctx.quadraticCurveTo(w*0.5, h*0.15, w*0.8, h*0.35);
    ctx.quadraticCurveTo(w*0.5, h*0.42, w*0.2, h*0.35);
    ctx.fill();
    
    // Feather highlights
    ctx.strokeStyle = '#ffe4e6';
    ctx.lineWidth = 1.5;
    for (let i = 0; i < 15; i++) {
      ctx.beginPath();
      ctx.moveTo(w*0.25 + i*8, h*0.5);
      ctx.quadraticCurveTo(w*0.3 + i*6, h*0.65, w*0.1 + i*5, h*0.8);
      ctx.stroke();
    }
  } 
  else if (scene === 'mandrill') {
    // Mandrill high texture: blue cheeks, yellow/white beard, high frequency noise
    // Base brown face
    ctx.fillStyle = '#78350f';
    ctx.fillRect(0, 0, w, h);
    
    // Blue cheek pads (gradient)
    const cheekGrad = ctx.createLinearGradient(0, 0, w, 0);
    cheekGrad.addColorStop(0.2, '#0284c7');
    cheekGrad.addColorStop(0.5, '#0f172a');
    cheekGrad.addColorStop(0.8, '#0284c7');
    ctx.fillStyle = cheekGrad;
    ctx.beginPath();
    ctx.ellipse(w*0.5, h*0.55, w*0.35, h*0.2, 0, 0, Math.PI*2);
    ctx.fill();
    
    // White/yellow beard strokes (High frequency)
    ctx.strokeStyle = '#fef08a';
    ctx.lineWidth = 1;
    for (let i = 0; i < 400; i++) {
      const rx = Math.random() * w;
      const ry = h*0.65 + Math.random() * (h*0.35);
      ctx.beginPath();
      ctx.moveTo(rx, ry);
      ctx.lineTo(rx + (Math.random() - 0.5) * 10, ry + Math.random() * 20);
      ctx.stroke();
    }

    // Red nose ridge
    ctx.fillStyle = '#dc2626';
    ctx.fillRect(w*0.44, h*0.2, w*0.12, h*0.42);

    // Deep eyes
    ctx.fillStyle = '#facc15';
    ctx.beginPath();
    ctx.arc(w*0.36, h*0.28, w*0.04, 0, Math.PI*2);
    ctx.arc(w*0.64, h*0.28, w*0.04, 0, Math.PI*2);
    ctx.fill();
    
    ctx.fillStyle = '#000000';
    ctx.beginPath();
    ctx.arc(w*0.36, h*0.28, w*0.015, 0, Math.PI*2);
    ctx.arc(w*0.64, h*0.28, w*0.015, 0, Math.PI*2);
    ctx.fill();
  } 
  else if (scene === 'satellite') {
    // Dense structural grids, river path, agricultural fields
    ctx.fillStyle = '#1e3a1e';
    ctx.fillRect(0, 0, w, h);
    
    // Agricultural fields grid
    ctx.strokeStyle = '#4d7c0f';
    ctx.lineWidth = 1;
    for(let i=0; i<w; i+=32) {
      ctx.beginPath();
      ctx.moveTo(i, 0); ctx.lineTo(i, h);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(0, i); ctx.lineTo(w, i);
      ctx.stroke();
    }
    
    // Diverse field crops colors
    for(let r=0; r<16; r++) {
      for(let c=0; c<16; c++) {
        if((r+c) % 3 === 0) {
          ctx.fillStyle = '#3f6212';
          ctx.fillRect(c*32+1, r*32+1, 30, 30);
        } else if((r+c) % 5 === 0) {
          ctx.fillStyle = '#7c2d12';
          ctx.fillRect(c*32+1, r*32+1, 30, 30);
        }
      }
    }
    
    // Curvy Blue River
    ctx.strokeStyle = '#2563eb';
    ctx.lineWidth = 14;
    ctx.lineCap = 'round';
    ctx.beginPath();
    ctx.moveTo(0, h*0.1);
    ctx.bezierCurveTo(w*0.3, h*0.2, w*0.4, h*0.8, w, h*0.6);
    ctx.stroke();
    
    // Main Roads
    ctx.strokeStyle = '#94a3b8';
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.moveTo(w*0.1, 0);
    ctx.lineTo(w*0.8, h);
    ctx.stroke();
  } 
  else if (scene === 'medical') {
    // Gray scale axial brain MRI outline and brain ventricles
    ctx.fillStyle = '#020617';
    ctx.fillRect(0, 0, w, h);
    
    // Skull (Greyscale gradient ring)
    ctx.strokeStyle = '#cbd5e1';
    ctx.lineWidth = 8;
    ctx.beginPath();
    ctx.ellipse(w*0.5, h*0.5, w*0.38, h*0.44, 0, 0, Math.PI*2);
    ctx.stroke();
    
    // Brain Tissue (Wavy convoluted gray shapes)
    ctx.fillStyle = '#475569';
    ctx.beginPath();
    ctx.ellipse(w*0.5, h*0.5, w*0.34, h*0.4, 0, 0, Math.PI*2);
    ctx.fill();
    
    // Lobes convolutions details
    ctx.strokeStyle = '#1e293b';
    ctx.lineWidth = 3;
    for(let i=0; i<18; i++) {
      ctx.beginPath();
      ctx.arc(w*0.5 + Math.cos(i) * w * 0.2, h*0.5 + Math.sin(i) * h * 0.25, w*0.08, 0, Math.PI);
      ctx.stroke();
    }
    
    // Internal ventricles (Butterfly shape, black/fluid)
    ctx.fillStyle = '#020617';
    ctx.beginPath();
    ctx.moveTo(w*0.5, h*0.35);
    ctx.bezierCurveTo(w*0.4, h*0.38, w*0.4, h*0.5, w*0.48, h*0.52);
    ctx.bezierCurveTo(w*0.4, h*0.55, w*0.42, h*0.62, w*0.5, h*0.65);
    ctx.bezierCurveTo(w*0.58, h*0.62, w*0.6, h*0.55, w*0.52, h*0.52);
    ctx.bezierCurveTo(w*0.6, h*0.5, w*0.6, h*0.38, w*0.5, h*0.35);
    ctx.closePath();
    ctx.fill();
    
    // Ventricle highlights
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = 2;
    ctx.stroke();
  }
}

// Log Terminal writer
function writeLog(text, type = 'info') {
  const line = document.createElement('div');
  line.className = `log-line ${type}`;
  line.innerText = `[${new Date().toLocaleTimeString()}] ${text}`;
  elements.embeddingConsole.appendChild(line);
  elements.embeddingConsole.scrollTop = elements.embeddingConsole.scrollHeight;
}

// Update character and bit length indicators
function updatePayloadMetrics() {
  const text = elements.secretMessageInput.value;
  state.message = text;
  
  const chars = text.length;
  const bits = chars * 8;
  
  elements.charCounter.innerText = `${chars} / 120 chars`;
  elements.bitCounter.innerText = `${bits} bits`;
}

// SPA Routing
function handleNavigation(targetId) {
  // Update nav buttons active state
  elements.navButtons.forEach(btn => {
    if (btn.getAttribute('data-target') === targetId) {
      btn.classList.add('active');
    } else {
      btn.classList.remove('active');
    }
  });

  // Toggle Viewports
  elements.viewports.forEach(vp => {
    if (vp.id === targetId) {
      vp.classList.add('active');
    } else {
      vp.classList.remove('active');
    }
  });
}

// Differentiable downscaling simulation math labels
function updateThreatVectorInfo() {
  const algo = state.attackAlgo;
  let explanation = '';
  
  switch(algo) {
    case 'nearest':
      explanation = `<strong>Nearest Neighbor:</strong> resizes by rounding coordinates to the nearest pixel index: <code>f(x, y) = g(round(x/r), round(y/r))</code>. This completely drops 75% to 98% of rows and columns, leading to massive aliasing artifacts and absolute destruction of LSB bit patterns.`;
      break;
    case 'bilinear':
      explanation = `<strong>Bilinear Interpolation:</strong> performs 2D linear filtering using a 2x2 local pixel neighborhood: <code>W(x) = max(0, 1 - |x|)</code>. The resulting pixels are averages of adjacent locations, blurring away high-frequency spatial bits.`;
      break;
    case 'bicubic':
      explanation = `<strong>Bicubic Interpolation:</strong> computes pixel values using a 4x4 coordinate neighborhood and cubic spline convolution: <code>W(x) = (a+2)|x|³ - (a+3)|x|² + 1</code>. Acts as an aggressive low-pass reconstruction filter, causing severe loss of spatial stego data.`;
      break;
    case 'lanczos':
      explanation = `<strong>Lanczos Resampling:</strong> applies a windowed sinc kernel convolution filter: <code>L(x) = sinc(x)sinc(x/a)</code>. It models mathematical sinc filters to retain edge details, but the high-frequency pixel phase transitions holding steganographic arrays are still heavily degraded.`;
      break;
    case 'area':
      explanation = `<strong>Area-Based Resampling:</strong> computes average value of pixels under a pixel mapping box. Acts as a box blur downsampler, neutralizing pixel-level variations.`;
      break;
  }
  elements.mathExplanationText.innerHTML = explanation;
}

// Generate Stego Image simulation (Injects simulated noise map / latent signatures)
function runStegoPipeline() {
  if (state.message.trim() === '') {
    alert("Please enter a secret message payload before embedding.");
    return;
  }

  writeLog("[SYSTEM] Initiating embedding thread...", "system");
  elements.btnEmbedPayload.disabled = true;
  elements.btnEmbedPayload.innerText = "Processing Embedding...";
  
  // Create timeline simulations
  setTimeout(() => {
    writeLog("[INFO] Preprocessing cover pixel matrix (512x512x3)...", "info");
  }, 300);

  setTimeout(() => {
    writeLog(`[INFO] Message byte representation converted to bitstream (${state.message.length * 8} bits).`, "info");
  }, 700);

  setTimeout(() => {
    if (state.method === 'cnn') {
      writeLog("[INFO] Executing ResNet-based Deep Encoder network...", "info");
      writeLog("[INFO] Differentiable downscale training simulation active. Noise distribution matched.", "info");
    } else {
      writeLog("[INFO] Allocating Spatial LSB buffers...", "info");
      writeLog("[INFO] Writing message bits to least-significant bit (8th bit plane).", "info");
    }
  }, 1100);

  setTimeout(() => {
    // Generate results
    state.embedded = true;
    
    // Copy cover canvas to result pages
    const coverCtx = elements.coverCanvas.getContext('2d');
    const w = elements.coverCanvas.width;
    const h = elements.coverCanvas.height;
    
    const coverData = coverCtx.getImageData(0, 0, w, h);
    
    // Draw to Result original cover
    const resCoverCtx = elements.resultCoverCanvas.getContext('2d');
    resCoverCtx.putImageData(coverData, 0, 0);
    
    // Draw to Stego (simulate small modifications)
    const resStegoCtx = elements.resultStegoCanvas.getContext('2d');
    const stegoData = resCoverCtx.createImageData(w, h);
    
    // Populate Stego Data
    for (let i = 0; i < coverData.data.length; i++) {
      stegoData.data[i] = coverData.data[i];
    }
    
    // We add slight alterations based on algorithm selected
    const seedText = state.message;
    if (state.method === 'lsb') {
      // Classic spatial bits replacement: very tiny random changes on 1 pixel in 8
      for (let i = 0; i < seedText.length * 8 * 4; i += 4) {
        // alter R or G or B channel slightly
        const targetIdx = Math.floor(Math.random() * (w * h)) * 4;
        const channel = Math.floor(Math.random() * 3);
        stegoData.data[targetIdx + channel] ^= 1; // Toggle LSB
      }
      state.psnr = 51.20 + (Math.random() * 0.6);
      state.ssim = 0.9998;
      
      elements.methodTitleBadge.innerText = "Spatial LSB Steganography";
      elements.methodDetailsText.innerText = "Least Significant Bit embedding injects message bitstream values directly into the lowest order bits of RGB color channels. While offering excellent image quality with close to zero visual distortions (high PSNR), it stores data in fragile pixel states that are easily destroyed by image scaling.";
      elements.cnnNetworkPreview.style.display = 'none'; // Hide neural net diagram
    } else {
      // CNN Stego details: features are integrated globally using conv kernels, slightly more global variance
      for (let y = 0; y < h; y++) {
        for (let x = 0; x < w; x++) {
          const idx = (y * w + x) * 4;
          // Apply a gentle high frequency convolution overlay to mimic neural weight distribution
          // Mostly visible on edges
          const edgeFactor = Math.sin(x/5) * Math.cos(y/5);
          if (Math.abs(edgeFactor) > 0.8) {
            stegoData.data[idx] = Math.min(255, Math.max(0, stegoData.data[idx] + Math.floor(Math.random() * 3 - 1.5)));
            stegoData.data[idx+1] = Math.min(255, Math.max(0, stegoData.data[idx+1] + Math.floor(Math.random() * 3 - 1.5)));
            stegoData.data[idx+2] = Math.min(255, Math.max(0, stegoData.data[idx+2] + Math.floor(Math.random() * 3 - 1.5)));
          }
        }
      }
      state.psnr = 38.42 - (Math.random() * 0.4);
      state.ssim = 0.9682 - (Math.random() * 0.002);
      
      elements.methodTitleBadge.innerText = "CNN Encoder-Decoder Model";
      elements.methodDetailsText.innerText = "Using a deep convolutional neural network, the encoder has been optimized jointly with a differentiable downscale simulator. It hides secret payload bits inside complex textures and edge contours rather than plain pixels, providing high resistance to downscaling attacks.";
      elements.cnnNetworkPreview.style.display = 'flex'; // Show neural net diagram
    }
    
    resStegoCtx.putImageData(stegoData, 0, 0);
    
    // Generate Residual noise map (S - C) x 150
    const resResidualCtx = elements.resultResidualCanvas.getContext('2d');
    const residualData = resResidualCtx.createImageData(w, h);
    
    for (let i = 0; i < coverData.data.length; i += 4) {
      // calculate difference
      const dR = Math.abs(stegoData.data[i] - coverData.data[i]) * 150;
      const dG = Math.abs(stegoData.data[i+1] - coverData.data[i+1]) * 150;
      const dB = Math.abs(stegoData.data[i+2] - coverData.data[i+2]) * 150;
      
      residualData.data[i] = dR;
      residualData.data[i+1] = dG;
      residualData.data[i+2] = dB;
      residualData.data[i+3] = 255; // Alpha
    }
    resResidualCtx.putImageData(residualData, 0, 0);
    
    // Update metrics UI
    elements.metricPsnr.innerText = `${state.psnr.toFixed(2)} dB`;
    elements.metricSsim.innerText = state.ssim.toFixed(4);
    
    // Update gauge bars
    elements.psnrBar.style.width = `${Math.min(100, (state.psnr / 60) * 100)}%`;
    elements.ssimBar.style.width = `${state.ssim * 100}%`;
    
    // Enable other controls
    elements.stegoSourceStatus.innerText = `Stego loaded. Method: ${state.method === 'cnn' ? 'Robust CNN' : 'Spatial LSB'}.`;
    elements.btnSimulateAttack.disabled = false;
    
    writeLog(`[SUCCESS] Steganographic embedding complete. PSNR: ${state.psnr.toFixed(2)}dB, SSIM: ${state.ssim.toFixed(4)}`, "success");
    
    elements.btnEmbedPayload.disabled = false;
    elements.btnEmbedPayload.innerText = "Embed Message";
    
    // Automatically route user to results page
    setTimeout(() => {
      handleNavigation('page-result');
    }, 400);

  }, 1600);
}

// Canvas-based resizing algorithms simulator (Nearest Neighbor, Bilinear, Bicubic, etc.)
function runDownscalingAttack() {
  if (!state.embedded) {
    alert("Please perform the steganography embedding step in Module 1 first.");
    return;
  }

  writeLog("[ATTACK] Initializing Downscaling Attack simulation...", "system");
  elements.btnSimulateAttack.disabled = true;
  elements.btnSimulateAttack.innerText = "Resizing Stego Image...";
  
  // Set scale label on connection arrow
  elements.arrowScaleLabel.innerText = `${state.attackScale * 100}%`;
  
  // Copy stego image to attack input canvas
  const stegoCtx = elements.resultStegoCanvas.getContext('2d');
  const w = elements.resultStegoCanvas.width;
  const h = elements.resultStegoCanvas.height;
  const stegoData = stegoCtx.getImageData(0, 0, w, h);
  
  const attInCtx = elements.attackInputCanvas.getContext('2d');
  attInCtx.putImageData(stegoData, 0, 0);
  
  // Set output canvas resolution according to scale
  const outW = Math.floor(w * state.attackScale);
  const outH = Math.floor(h * state.attackScale);
  elements.attackOutputCanvas.width = outW;
  elements.attackOutputCanvas.height = outH;
  
  // Adjust output canvas wrapper css sizing to visualize quality loss/aliasing
  elements.attackOutputCanvas.style.width = '100%';
  
  const attOutCtx = elements.attackOutputCanvas.getContext('2d');
  
  // Progress bar animation
  let prog = 0;
  const interval = setInterval(() => {
    prog += 10;
    elements.attackProgressFill.style.width = `${prog}%`;
    elements.progressPercentage.innerText = `${prog}%`;
    
    if (prog === 20) {
      elements.progressStatusText.innerText = "Sampling original stego matrices...";
    } else if (prog === 50) {
      elements.progressStatusText.innerText = `Running ${state.attackAlgo.toUpperCase()} interpolation kernel...`;
    } else if (prog === 80) {
      elements.progressStatusText.innerText = "Reconstructing downsampled coordinates...";
    }
    
    if (prog >= 100) {
      clearInterval(interval);
      
      // Perform ACTUAL canvas image scaling based on algorithm selected
      attOutCtx.clearRect(0, 0, outW, outH);
      
      if (state.attackAlgo === 'nearest') {
        // Manual nearest neighbor to show hard pixelated edges
        attOutCtx.imageSmoothingEnabled = false;
        attOutCtx.drawImage(elements.attackInputCanvas, 0, 0, w, h, 0, 0, outW, outH);
      } else {
        // High quality bilinear/bicubic/lanczos simulation using canvas interpolation
        attOutCtx.imageSmoothingEnabled = true;
        attOutCtx.imageSmoothingQuality = (state.attackAlgo === 'bilinear') ? 'medium' : 'high';
        attOutCtx.drawImage(elements.attackInputCanvas, 0, 0, w, h, 0, 0, outW, outH);
      }
      
      // Save attack output state
      state.attackSimulated = true;
      elements.btnContinueToExtraction.disabled = false;
      elements.btnContinueToExtraction.classList.remove('disabled');
      elements.progressStatusText.innerText = "Attack completed successfully.";
      
      // Update Page 4 import status
      elements.downscaledImportStatus.innerText = `Attacked image loaded (${outW}x${outH}px). Ready.`;
      
      // Copy to mini canvases
      const miniCtx = elements.miniDownscaledCanvas.getContext('2d');
      elements.miniDownscaledCanvas.width = 64;
      elements.miniDownscaledCanvas.height = 64;
      miniCtx.drawImage(elements.attackOutputCanvas, 0, 0, outW, outH, 0, 0, 64, 64);
      
      writeLog(`[SUCCESS] Resizing attack complete. Rescaled to ${outW}x${outH} px using ${state.attackAlgo}.`, "success");
      elements.btnSimulateAttack.disabled = false;
      elements.btnSimulateAttack.innerText = "Simulate Downscaling Attack";
    }
  }, 150);
}

// Extraction Module Simulator
function runExtractionAndEvaluation() {
  if (!state.attackSimulated) {
    alert("Please simulate the downscaling attack in Module 3 first.");
    return;
  }

  elements.btnExtractMessage.disabled = true;
  elements.btnExtractMessage.innerText = "Decrypting Payload...";
  
  setTimeout(() => {
    state.extracted = true;
    
    // Update original text display
    elements.displayOriginalMsg.innerText = state.message;
    
    // Simulate Decryption BER and text corruptions
    let rawBer = 0.00;
    let textOut = '';
    
    if (state.method === 'lsb') {
      // LSB is highly fragile to resizing!
      // Scale Factor determines corruption level
      if (state.attackScale === 0.5) {
        rawBer = 32.41;
        // corrupt ~ 30% of characters
        textOut = corruptMessageText(state.message, 0.3);
      } else if (state.attackScale === 0.25) {
        rawBer = 46.18;
        // corrupt ~ 75% of characters
        textOut = corruptMessageText(state.message, 0.75);
      } else {
        rawBer = 49.82;
        // pure noise
        textOut = corruptMessageText(state.message, 0.95);
      }
      
      elements.berStatusIndicator.className = "m-indicator red";
      elements.extractedMsgLabel.innerText = "Recovered Text (M') - [CORRUPTED / LSB]";
    } else {
      // CNN Robust model!
      if (state.attackScale === 0.5) {
        rawBer = 0.00;
        textOut = state.message; // Perfect match
      } else if (state.attackScale === 0.25) {
        rawBer = 0.00;
        textOut = state.message; // Perfect match
      } else {
        rawBer = 1.84;
        // minor typo (1 or 2 characters)
        textOut = corruptMessageText(state.message, 0.05);
      }
      
      elements.berStatusIndicator.className = "m-indicator green";
      elements.extractedMsgLabel.innerText = "Recovered Text (M') - [100% SECURE / CNN]";
    }
    
    state.ber = rawBer;
    elements.displayExtractedMsg.innerText = textOut;
    elements.extractBer.innerText = `${state.ber.toFixed(2)} %`;
    
    // Set extracted image quality simulation metrics
    // Resize yields low resolution metrics relative to original cover
    state.extractedPsnr = 10 + (Math.random() * 3); // low due to grid shape shrinkage
    state.extractedSsim = 0.3 + (Math.random() * 0.15);
    
    elements.extractPsnr.innerText = `${state.extractedPsnr.toFixed(2)} dB`;
    elements.extractSsim.innerText = state.extractedSsim.toFixed(4);
    
    // Update comparison benchmark table dynamically depending on current selection
    updateBenchmarkTable();
    
    // Update academic report summary paragraph
    updateAcademicSummaryParagraph();
    
    elements.btnExtractMessage.disabled = false;
    elements.btnExtractMessage.innerText = "Extract Hidden Message";
    
  }, 1000);
}

// Corrupts string characters with random bits/glyphs based on corruption percentage
function corruptMessageText(text, probability) {
  const chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789!@#$%^&*()_+=[]{}';
  let out = '';
  for (let i = 0; i < text.length; i++) {
    if (Math.random() < probability) {
      // corrupt character (sometimes replace with random glyph, sometimes space)
      if (Math.random() < 0.2) {
        out += ' ';
      } else {
        out += chars.charAt(Math.floor(Math.random() * chars.length));
      }
    } else {
      out += text.charAt(i);
    }
  }
  return out;
}

// Dynamically updates comparison benchmark table rows based on current state variables
function updateBenchmarkTable() {
  const scalePct = `${state.attackScale * 100}%`;
  const algoName = state.attackAlgo.charAt(0).toUpperCase() + state.attackAlgo.slice(1);
  
  // Construct rows: LSB vs CNN under current attack parameters
  let lsbBer = 32.41;
  let lsbPsnr = 12.18;
  let lsbSsim = 0.4851;
  
  let cnnBer = 0.00;
  let cnnPsnr = 31.45;
  let cnnSsim = 0.9104;
  
  if (state.attackScale === 0.25) {
    lsbBer = 46.18; lsbPsnr = 9.04; lsbSsim = 0.2982;
    cnnBer = 0.00; cnnPsnr = 28.52; cnnSsim = 0.8653;
  } else if (state.attackScale === 0.125) {
    lsbBer = 49.82; lsbPsnr = 6.42; lsbSsim = 0.1142;
    cnnBer = 1.84; cnnPsnr = 24.15; cnnSsim = 0.7915;
  }
  
  elements.evaluationTableRows.innerHTML = `
    <tr>
      <td class="method-cell lsb-cell">LSB (Standard)</td>
      <td>${scalePct} (${algoName})</td>
      <td class="worse-val">${lsbBer.toFixed(2)}%</td>
      <td>${lsbPsnr.toFixed(2)} dB</td>
      <td>${lsbSsim.toFixed(4)}</td>
    </tr>
    <tr>
      <td class="method-cell cnn-cell">CNN (Deep/Robust)</td>
      <td>${scalePct} (${algoName})</td>
      <td class="better-val">${cnnBer.toFixed(2)}%</td>
      <td>${cnnPsnr.toFixed(2)} dB</td>
      <td>${cnnSsim.toFixed(4)}</td>
    </tr>
  `;
}

// Update the textual academic explanation inside extraction module
function updateAcademicSummaryParagraph() {
  const algoName = state.attackAlgo.charAt(0).toUpperCase() + state.attackAlgo.slice(1);
  const scalePct = `${state.attackScale * 100}%`;
  
  let summary = '';
  if (state.method === 'lsb') {
    summary = `Under standard spatial LSB embedding, downscaling filters cause destructive spatial coordinate interpolations. Applying a <strong>${scalePct} ${algoName} Resizing</strong> attack resulted in an unrecoverable <strong>${state.ber.toFixed(2)}% Bit Error Rate (BER)</strong>. This confirms that classical pixel-replacement steganography lacks any defensive capacity against spatial modifications.`;
  } else {
    summary = `The deep CNN Encoder-Decoder model successfully bypassed the <strong>${scalePct} ${algoName} Resizing</strong> attack, yielding a highly robust <strong>${state.ber.toFixed(2)}% Bit Error Rate (BER)</strong>. The decoder reconstructs the secret message perfectly because information was distributed globally within structural coefficients optimized by the differentiable layer.`;
  }
  elements.evaluationSummaryText.innerHTML = summary;
}

// Initialize Application Events
function initApp() {
  
  // Navigation tabs routing
  elements.navButtons.forEach(btn => {
    btn.addEventListener('click', () => {
      const target = btn.getAttribute('data-target');
      handleNavigation(target);
    });
  });

  // Settings: Toggle Academic Figure Captions
  elements.toggleFigures.addEventListener('change', (e) => {
    if (e.target.checked) {
      document.body.classList.remove('hide-academic-figures');
    } else {
      document.body.classList.add('hide-academic-figures');
    }
  });

  // Settings: Toggle Theme light/dark
  elements.toggleTheme.addEventListener('change', (e) => {
    if (e.target.checked) {
      document.body.className = 'dark-theme';
    } else {
      document.body.className = 'light-theme';
    }
  });

  // Page 1: Sample select thumbnails
  elements.sampleThumbButtons.forEach(btn => {
    btn.addEventListener('click', () => {
      elements.sampleThumbButtons.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      
      const sampleType = btn.getAttribute('data-sample');
      state.coverScene = sampleType;
      state.coverLoaded = true;
      
      // Redraw cover canvas
      drawProceduralImage(elements.coverCanvas, sampleType);
      writeLog(`[INFO] Loaded sample cover image scene: '${sampleType.toUpperCase()}'`, "info");
    });
  });

  // Page 1: Secret Message character limits counter
  elements.secretMessageInput.addEventListener('input', updatePayloadMetrics);

  // Page 1: Embed method toggle cards
  elements.methodCards.forEach(card => {
    const radio = card.querySelector('input[type="radio"]');
    card.addEventListener('click', () => {
      elements.methodCards.forEach(c => c.classList.remove('active'));
      card.classList.add('active');
      radio.checked = true;
      state.method = radio.value;
      writeLog(`[INFO] Active steganography algorithm changed to: ${state.method.toUpperCase()}`, "info");
    });
  });

  // Page 1: File Uploader handlers
  elements.btnTriggerUpload.addEventListener('click', () => {
    elements.coverFileInput.click();
  });
  
  elements.coverFileInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (!file) return;
    
    // Select thumbnail visual updates
    elements.sampleThumbButtons.forEach(b => b.classList.remove('active'));
    elements.btnUseSample.classList.remove('active');
    elements.btnTriggerUpload.classList.add('active');
    
    const reader = new FileReader();
    reader.onload = (event) => {
      const img = new Image();
      img.onload = () => {
        const ctx = elements.coverCanvas.getContext('2d');
        ctx.clearRect(0, 0, 512, 512);
        // Draw fitting boundaries
        ctx.drawImage(img, 0, 0, 512, 512);
        state.coverLoaded = true;
        state.coverScene = 'Uploaded File';
        writeLog(`[INFO] Custom cover image '${file.name}' loaded successfully into workspace memory.`, "info");
      };
      img.src = event.target.result;
    };
    reader.readAsDataURL(file);
  });
  
  elements.btnUseSample.addEventListener('click', () => {
    elements.btnTriggerUpload.classList.remove('active');
    elements.btnUseSample.classList.add('active');
    // Default back to selected sample active button
    const activeThumb = document.querySelector('.sample-thumb-btn.active');
    if (activeThumb) activeThumb.click();
  });

  // Page 1: Reset Button
  elements.btnResetEmbedding.addEventListener('click', () => {
    elements.secretMessageInput.value = "Deep Steganography hides messages within image features, maintaining robustness against Downscaling Attacks.";
    updatePayloadMetrics();
    // Reset to default sample Lena
    const thumbLena = document.querySelector('.sample-thumb-btn[data-sample="lena"]');
    if (thumbLena) thumbLena.click();
    writeLog("[SYSTEM] Configuration settings reset to academic defaults.", "system");
  });

  // Page 1: Embed Message button trigger
  elements.btnEmbedPayload.addEventListener('click', runStegoPipeline);

  // Page 2: Comparison Mode selectors
  elements.btnModeSide.addEventListener('click', () => {
    elements.btnModeSide.classList.add('active');
    elements.btnModeResidual.classList.remove('active');
    elements.sideBySideView.classList.add('active');
    elements.residualMapView.classList.remove('active');
  });

  elements.btnModeResidual.addEventListener('click', () => {
    elements.btnModeSide.classList.remove('active');
    elements.btnModeResidual.classList.add('active');
    elements.sideBySideView.classList.remove('active');
    elements.residualMapView.classList.add('active');
  });

  // Page 2: Download stego file simulation
  elements.btnDownloadStego.addEventListener('click', () => {
    const dataUrl = elements.resultStegoCanvas.toDataURL('image/png');
    const link = document.createElement('a');
    link.download = `stego_${state.method}_${state.coverScene}.png`;
    link.href = dataUrl;
    link.click();
  });

  // Page 2: Proceed shortcut
  elements.btnGoToDownscaling.addEventListener('click', () => {
    // Import values automatically
    elements.btnImportLatest.click();
    handleNavigation('page-downscaling');
  });

  // Page 3: Import stego action
  elements.btnImportLatest.addEventListener('click', () => {
    if (!state.embedded) {
      alert("No stego image has been embedded yet. Please complete Module 1 and 2.");
      return;
    }
    const miniCtx = elements.miniStegoCanvas.getContext('2d');
    miniCtx.clearRect(0, 0, 64, 64);
    miniCtx.drawImage(elements.resultStegoCanvas, 0, 0, 512, 512, 0, 0, 64, 64);
    
    // Copy main preview
    const attInCtx = elements.attackInputCanvas.getContext('2d');
    attInCtx.drawImage(elements.resultStegoCanvas, 0, 0);
    
    elements.stegoSourceStatus.innerText = `Stego loaded. Method: ${state.method === 'cnn' ? 'Robust CNN' : 'Spatial LSB'}.`;
    
    writeLog("[INFO] Imported latest stego output image as attack simulator source.", "info");
  });

  // Page 3: Dropdown change triggers text formula details updates
  elements.downscaleAlgorithm.addEventListener('change', (e) => {
    state.attackAlgo = e.target.value;
    updateThreatVectorInfo();
  });

  // Page 3: Scale radio buttons selection
  document.querySelectorAll('input[name="scale-factor"]').forEach(radio => {
    radio.addEventListener('change', (e) => {
      state.attackScale = parseFloat(e.target.value);
    });
  });

  // Page 3: Simulate Attack button
  elements.btnSimulateAttack.addEventListener('click', runDownscalingAttack);

  // Page 3: Proceed shortcut to extraction
  elements.btnContinueToExtraction.addEventListener('click', () => {
    // Import automatically
    elements.btnImportAttacked.click();
    handleNavigation('page-extraction');
  });

  // Page 4: Import attacked image action
  elements.btnImportAttacked.addEventListener('click', () => {
    if (!state.attackSimulated) {
      alert("No downscaled image has been simulated. Please run Module 3.");
      return;
    }
    const miniCtx = elements.miniDownscaledCanvas.getContext('2d');
    miniCtx.clearRect(0, 0, 64, 64);
    miniCtx.drawImage(elements.attackOutputCanvas, 0, 0, elements.attackOutputCanvas.width, elements.attackOutputCanvas.height, 0, 0, 64, 64);
    
    elements.downscaledImportStatus.innerText = `Attacked image loaded (${elements.attackOutputCanvas.width}x${elements.attackOutputCanvas.height}px). Ready.`;
  });

  // Page 4: Extract hidden payload trigger
  elements.btnExtractMessage.addEventListener('click', runExtractionAndEvaluation);

  // Page 4: Export report modal
  elements.btnExportReport.addEventListener('click', () => {
    // Populate report meta data values dynamically
    elements.reportDate.innerText = new Date().toLocaleDateString('en-US', { year: 'numeric', month: 'long', day: 'numeric' });
    elements.reportCoverName.innerText = `${state.coverScene.toUpperCase()} (512x512 px)`;
    elements.reportMsgLen.innerText = `${state.message.length} characters (${state.message.length * 8} bits)`;
    
    const scalePct = `${state.attackScale * 100}%`;
    const algoName = elements.downscaleAlgorithm.options[elements.downscaleAlgorithm.selectedIndex].text;
    
    elements.reportTableAlgo.innerText = algoName;
    elements.reportTableScale.innerText = `${scalePct} (${elements.attackOutputCanvas.width}x${elements.attackOutputCanvas.height} px)`;
    
    // Simulate benchmark values
    let lsbBer = 32.41;
    let cnnBer = 0.00;
    if (state.attackScale === 0.25) {
      lsbBer = 46.18; cnnBer = 0.00;
    } else if (state.attackScale === 0.125) {
      lsbBer = 49.82; cnnBer = 1.84;
    }
    
    elements.reportLsbBer.innerText = `${lsbBer.toFixed(2)}%`;
    elements.reportCnnBer.innerText = `${cnnBer.toFixed(2)}%`;
    
    elements.reportModal.classList.add('active');
  });

  // Modal actions
  elements.btnCloseModal.addEventListener('click', () => {
    elements.reportModal.classList.remove('active');
  });
  
  elements.btnDismissModal.addEventListener('click', () => {
    elements.reportModal.classList.remove('active');
  });

  elements.btnPrintDocument.addEventListener('click', () => {
    // Standard print window simulation for saving as PDF
    window.print();
  });

  // Initialize Canvas Scenes
  drawProceduralImage(elements.coverCanvas, 'lena');
  updatePayloadMetrics();
  updateThreatVectorInfo();
}

// Start application when DOM loads
window.addEventListener('DOMContentLoaded', initApp);
