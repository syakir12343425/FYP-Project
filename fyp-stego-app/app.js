/**
 * AI-Based Robust Steganography Against Image Downscaling Attacks
 * FYP Research UI - Core Application Logic
 *
 * This version is wired to the real inference server (server.py): the
 * embed / downscale / extract steps call the trained CNN (and real LSB)
 * over HTTP instead of simulating results client-side.
 */

// Point this at wherever server.py is running (uvicorn server:app --port 8000).
// Change to your VPS address once you deploy it there, e.g. "https://api.yourdomain.com".
const API_BASE = "http://localhost:8000"; // use your HTTPS API domain on the VPS

// Real aggregate numbers from the full eval.py benchmark
// (100 validation images x 5 algorithms x 4 sizes), used as reference
// values in the comparison table / report so the UI never shows invented numbers.
const REFERENCE = {
  cnn: { ber: 1.15, psnr: 32.13, ssim: 0.9514 },
  lsb: { ber: 48.00, psnr: 79.99, ssim: 0.9999 },
};


// ------------------------------------------------------------------ auth
// Access token lives ONLY in this JS variable (never localStorage -> not stealable via XSS storage reads).
// The refresh token is an httpOnly+SameSite=Strict cookie the JS can't even see.
let accessToken = null;
let refreshTimer = null;

function setSession(data) {
  accessToken = data.access_token;
  document.getElementById('current-user').innerText = data.username;
  document.getElementById('login-overlay').classList.remove('active');
  clearTimeout(refreshTimer);
  refreshTimer = setTimeout(silentRefresh, Math.max(10, data.expires_in - 60) * 1000);
}

function clearSession() {
  accessToken = null;
  clearTimeout(refreshTimer);
  document.getElementById('login-overlay').classList.add('active');
}

async function authPost(path, body) {
  return fetch(`${API_BASE}${path}`, {
    method: "POST",
    credentials: "include",
    headers: { "Content-Type": "application/json", "X-Requested-With": "fetch" },
    body: body ? JSON.stringify(body) : undefined,
  });
}

// Single-flight: if several requests get 401 at once, they share ONE refresh call. Two parallel
// refreshes would send the same single-use cookie twice, which the server treats as token theft
// and answers by revoking the whole session.
let refreshInFlight = null;
function silentRefresh() {
  if (!refreshInFlight) {
    refreshInFlight = doRefresh().finally(() => { refreshInFlight = null; });
  }
  return refreshInFlight;
}

async function doRefresh() {
  try {
    const res = await authPost("/api/auth/refresh");
    if (!res.ok) throw new Error("refresh failed");
    setSession(await res.json());
    return true;
  } catch (_) {
    clearSession();
    return false;
  }
}

// fetch() wrapper for protected endpoints: attaches the bearer token, retries once after a refresh.
async function apiFetch(path, options = {}) {
  const run = () => fetch(`${API_BASE}${path}`, {
    ...options,
    headers: { ...(options.headers || {}), "Authorization": `Bearer ${accessToken}` },
  });
  let res = await run();
  if (res.status === 401 && await silentRefresh()) res = await run();
  if (res.status === 401) throw new Error("Session expired - please log in again.");
  if (res.status === 429) throw new Error("Too many requests - slow down and retry shortly.");
  return res;
}

async function handleLogin(e) {
  e.preventDefault();
  const errEl = document.getElementById('login-error');
  const btn = document.getElementById('login-submit');
  errEl.innerText = '';
  btn.disabled = true;
  try {
    const res = await authPost("/api/auth/login", {
      username: document.getElementById('login-username').value,
      password: document.getElementById('login-password').value,
    });
    if (res.status === 429) throw new Error("Too many attempts. Wait a minute and try again.");
    if (!res.ok) throw new Error("Invalid credentials, or account temporarily locked.");
    document.getElementById('login-password').value = '';
    setSession(await res.json());
  } catch (err) {
    errEl.innerText = err instanceof TypeError ? describeFetchError(err) : err.message;
  } finally {
    btn.disabled = false;
  }
}

async function handleLogout() {
  try { await authPost("/api/auth/logout"); } catch (_) {}
  clearSession();
}

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
  stegoUploaded: false,      // stego image came from a file upload, not Module 2
  downscaledUploaded: false, // image for extraction came from a file upload

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
    const bgGrad = ctx.createLinearGradient(0, 0, w, h);
    bgGrad.addColorStop(0, '#581c3f');
    bgGrad.addColorStop(0.5, '#701a35');
    bgGrad.addColorStop(1, '#9f1239');
    ctx.fillStyle = bgGrad;
    ctx.fillRect(0, 0, w, h);

    ctx.fillStyle = '#fda4af';
    ctx.beginPath();
    ctx.ellipse(w*0.35, h*0.5, w*0.3, h*0.2, Math.PI/4, 0, Math.PI*2);
    ctx.fill();

    ctx.fillStyle = '#fecdd3';
    ctx.beginPath();
    ctx.arc(w*0.5, h*0.45, w*0.18, 0, Math.PI*2);
    ctx.fill();

    ctx.fillStyle = '#0f172a';
    ctx.beginPath();
    ctx.ellipse(w*0.46, h*0.43, w*0.02, h*0.01, 0, 0, Math.PI*2);
    ctx.ellipse(w*0.54, h*0.43, w*0.02, h*0.01, 0, 0, Math.PI*2);
    ctx.fill();

    ctx.fillStyle = '#be123c';
    ctx.beginPath();
    ctx.ellipse(w*0.5, h*0.53, w*0.03, h*0.01, 0, 0, Math.PI*2);
    ctx.fill();

    ctx.fillStyle = '#b45309';
    ctx.beginPath();
    ctx.moveTo(w*0.2, h*0.35);
    ctx.quadraticCurveTo(w*0.5, h*0.15, w*0.8, h*0.35);
    ctx.quadraticCurveTo(w*0.5, h*0.42, w*0.2, h*0.35);
    ctx.fill();

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
    ctx.fillStyle = '#78350f';
    ctx.fillRect(0, 0, w, h);

    const cheekGrad = ctx.createLinearGradient(0, 0, w, 0);
    cheekGrad.addColorStop(0.2, '#0284c7');
    cheekGrad.addColorStop(0.5, '#0f172a');
    cheekGrad.addColorStop(0.8, '#0284c7');
    ctx.fillStyle = cheekGrad;
    ctx.beginPath();
    ctx.ellipse(w*0.5, h*0.55, w*0.35, h*0.2, 0, 0, Math.PI*2);
    ctx.fill();

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

    ctx.fillStyle = '#dc2626';
    ctx.fillRect(w*0.44, h*0.2, w*0.12, h*0.42);

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
    ctx.fillStyle = '#1e3a1e';
    ctx.fillRect(0, 0, w, h);

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

    ctx.strokeStyle = '#2563eb';
    ctx.lineWidth = 14;
    ctx.lineCap = 'round';
    ctx.beginPath();
    ctx.moveTo(0, h*0.1);
    ctx.bezierCurveTo(w*0.3, h*0.2, w*0.4, h*0.8, w, h*0.6);
    ctx.stroke();

    ctx.strokeStyle = '#94a3b8';
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.moveTo(w*0.1, 0);
    ctx.lineTo(w*0.8, h);
    ctx.stroke();
  }
  else if (scene === 'medical') {
    ctx.fillStyle = '#020617';
    ctx.fillRect(0, 0, w, h);

    ctx.strokeStyle = '#cbd5e1';
    ctx.lineWidth = 8;
    ctx.beginPath();
    ctx.ellipse(w*0.5, h*0.5, w*0.38, h*0.44, 0, 0, Math.PI*2);
    ctx.stroke();

    ctx.fillStyle = '#475569';
    ctx.beginPath();
    ctx.ellipse(w*0.5, h*0.5, w*0.34, h*0.4, 0, 0, Math.PI*2);
    ctx.fill();

    ctx.strokeStyle = '#1e293b';
    ctx.lineWidth = 3;
    for(let i=0; i<18; i++) {
      ctx.beginPath();
      ctx.arc(w*0.5 + Math.cos(i) * w * 0.2, h*0.5 + Math.sin(i) * h * 0.25, w*0.08, 0, Math.PI);
      ctx.stroke();
    }

    ctx.fillStyle = '#020617';
    ctx.beginPath();
    ctx.moveTo(w*0.5, h*0.35);
    ctx.bezierCurveTo(w*0.4, h*0.38, w*0.4, h*0.5, w*0.48, h*0.52);
    ctx.bezierCurveTo(w*0.4, h*0.55, w*0.42, h*0.62, w*0.5, h*0.65);
    ctx.bezierCurveTo(w*0.58, h*0.62, w*0.6, h*0.55, w*0.52, h*0.52);
    ctx.bezierCurveTo(w*0.6, h*0.5, w*0.6, h*0.38, w*0.5, h*0.35);
    ctx.closePath();
    ctx.fill();

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
  elements.navButtons.forEach(btn => {
    if (btn.getAttribute('data-target') === targetId) {
      btn.classList.add('active');
    } else {
      btn.classList.remove('active');
    }
  });

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
      explanation = `<strong>Lanczos Resampling:</strong> applies a windowed sinc kernel convolution filter: <code>L(x) = sinc(x)sinc(x/a)</code>. Held out of CNN training entirely — used here as a zero-shot generalization test of the decoder.`;
      break;
    case 'area':
      explanation = `<strong>Area-Based Resampling:</strong> computes average value of pixels under a pixel mapping box. Acts as a box blur downsampler, neutralizing pixel-level variations.`;
      break;
  }
  elements.mathExplanationText.innerHTML = explanation;
}

// --------------------------------------------------------------- server utils

// Draws a (possibly data:) image URL into a canvas, resizing the canvas to
// match the image's natural dimensions. Returns a Promise so callers can await it.
function drawDataUrlToCanvas(canvas, dataUrl) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => {
      canvas.width = img.naturalWidth;
      canvas.height = img.naturalHeight;
      const ctx = canvas.getContext('2d');
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(img, 0, 0);
      resolve();
    };
    img.onerror = reject;
    img.src = dataUrl;
  });
}

// Builds the (S - C) x150 residual visualisation from two same-size canvases.
function drawResidualMap(coverCanvas, stegoCanvas, outCanvas) {
  const w = coverCanvas.width, h = coverCanvas.height;
  outCanvas.width = w;
  outCanvas.height = h;
  const coverData = coverCanvas.getContext('2d').getImageData(0, 0, w, h);
  const stegoData = stegoCanvas.getContext('2d').getImageData(0, 0, w, h);
  const outCtx = outCanvas.getContext('2d');
  const residual = outCtx.createImageData(w, h);
  for (let i = 0; i < coverData.data.length; i += 4) {
    residual.data[i]   = Math.min(255, Math.abs(stegoData.data[i]   - coverData.data[i])   * 150);
    residual.data[i+1] = Math.min(255, Math.abs(stegoData.data[i+1] - coverData.data[i+1]) * 150);
    residual.data[i+2] = Math.min(255, Math.abs(stegoData.data[i+2] - coverData.data[i+2]) * 150);
    residual.data[i+3] = 255;
  }
  outCtx.putImageData(residual, 0, 0);
}

// Friendlier error if the server simply isn't running / unreachable.
function describeFetchError(err) {
  if (err instanceof TypeError) {
    return `Can't reach the inference server at ${API_BASE}. Is server.py running (uvicorn server:app --port 8000)?`;
  }
  return err.message;
}

// ----------------------------------------------------------- real pipeline steps

// Reads + validates an uploaded PNG/JPEG. Returns the ORIGINAL bytes as a data URL (not re-encoded via
// canvas, which would silently alter pixels - important when testing Telegram/JPEG round-trips).
function readImageFile(file) {
  return new Promise((resolve, reject) => {
    if (!file) return reject(new Error("No file selected."));
    if (!["image/png", "image/jpeg"].includes(file.type)) return reject(new Error("Only PNG or JPEG images are accepted."));
    if (file.size > 4 * 1024 * 1024) return reject(new Error("File too large (max 4 MB)."));
    const reader = new FileReader();
    reader.onerror = () => reject(new Error("Could not read the file."));
    reader.onload = () => {
      const dataUrl = reader.result;
      const img = new Image();
      img.onerror = () => reject(new Error("Not a valid image."));
      img.onload = () => {
        const { naturalWidth: w, naturalHeight: h } = img;
        if (w < 32 || h < 32 || w > 1280 || h > 1280) return reject(new Error(`Image is ${w}x${h}px; allowed range is 32-1280 px per side.`));
        resolve({ dataUrl, w, h });
      };
      img.src = dataUrl;
    };
    reader.readAsDataURL(file);
  });
}

function drawMini(canvas, source) {
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, 64, 64);
  ctx.drawImage(source, 0, 0, source.width, source.height, 0, 0, 64, 64);
}

// Page 1 -> 2: calls POST /api/embed on the real trained CNN (or real LSB).
async function runStegoPipeline() {
  if (state.message.trim() === '') {
    alert("Please enter a secret message payload before embedding.");
    return;
  }

  writeLog("[SYSTEM] Initiating embedding request to inference server...", "system");
  elements.btnEmbedPayload.disabled = true;
  elements.btnEmbedPayload.innerText = "Processing Embedding...";

  writeLog("[INFO] Preprocessing cover pixel matrix (512x512x3)...", "info");
  writeLog(`[INFO] Message byte representation converted to bitstream (${state.message.length * 8} bits, padded to 1024).`, "info");

  try {
    const coverDataUrl = elements.coverCanvas.toDataURL('image/png');

    writeLog(
      state.method === 'cnn'
        ? "[INFO] Running CNN Encoder (trained model, checkpoints/best.pt)..."
        : "[INFO] Running spatial LSB embed (8th bit plane)...",
      "info"
    );

    const res = await apiFetch("/api/embed", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        cover_image: coverDataUrl,
        message: state.message,
        method: state.method,
      }),
    });
    if (!res.ok) throw new Error(`Server returned HTTP ${res.status}`);
    const data = await res.json(); // { stego_image, psnr, ssim }

    state.embedded = true;
    state.stegoUploaded = false;
    state.stegoImgData = data.stego_image;
    state.psnr = data.psnr;
    state.ssim = data.ssim;

    await drawDataUrlToCanvas(elements.resultCoverCanvas, coverDataUrl);
    await drawDataUrlToCanvas(elements.resultStegoCanvas, data.stego_image);
    drawResidualMap(elements.resultCoverCanvas, elements.resultStegoCanvas, elements.resultResidualCanvas);

    if (state.method === 'lsb') {
      elements.methodTitleBadge.innerText = "Spatial LSB Steganography";
      elements.methodDetailsText.innerText = "Least Significant Bit embedding injects message bitstream values directly into the lowest order bits of RGB color channels. While offering excellent image quality with close to zero visual distortions (high PSNR), it stores data in fragile pixel states that are easily destroyed by image scaling.";
      elements.cnnNetworkPreview.style.display = 'none';
    } else {
      elements.methodTitleBadge.innerText = "CNN Encoder-Decoder Model";
      elements.methodDetailsText.innerText = "Using a deep convolutional neural network, the encoder has been optimized jointly with a differentiable downscale simulator. It hides secret payload bits inside complex textures and edge contours rather than plain pixels, providing high resistance to downscaling attacks.";
      elements.cnnNetworkPreview.style.display = 'flex';
    }

    elements.metricPsnr.innerText = `${state.psnr.toFixed(2)} dB`;
    elements.metricSsim.innerText = state.ssim.toFixed(4);
    elements.psnrBar.style.width = `${Math.min(100, (state.psnr / 60) * 100)}%`;
    elements.ssimBar.style.width = `${state.ssim * 100}%`;

    elements.stegoSourceStatus.innerText = `Stego loaded. Method: ${state.method === 'cnn' ? 'Robust CNN' : 'Spatial LSB'}.`;
    elements.btnSimulateAttack.disabled = false;

    writeLog(`[SUCCESS] Steganographic embedding complete. PSNR: ${state.psnr.toFixed(2)}dB, SSIM: ${state.ssim.toFixed(4)}`, "success");

    setTimeout(() => handleNavigation('page-result'), 400);
  } catch (err) {
    const msg = describeFetchError(err);
    writeLog(`[ERROR] Embedding failed: ${msg}`, "error");
    alert(`Embedding failed: ${msg}`);
  } finally {
    elements.btnEmbedPayload.disabled = false;
    elements.btnEmbedPayload.innerText = "Embed Message";
  }
}

// Page 3: calls POST /api/downscale, which runs the exact OpenCV algorithm
// (nearest/bilinear/bicubic/lanczos/area) used in eval.py.
async function runDownscalingAttack() {
  if (!state.stegoImgData) {
    alert("Embed a message in Module 1, or upload a stego image here, first.");
    return;
  }

  writeLog("[ATTACK] Sending stego image to server for real downscale attack...", "system");
  elements.btnSimulateAttack.disabled = true;
  elements.btnSimulateAttack.innerText = "Resizing Stego Image...";
  elements.arrowScaleLabel.innerText = `${state.attackScale * 100}%`;

  await drawDataUrlToCanvas(elements.attackInputCanvas, state.stegoImgData);

  elements.progressStatusText.innerText = `Running ${state.attackAlgo.toUpperCase()} interpolation kernel on server...`;
  elements.attackProgressFill.style.width = `40%`;
  elements.progressPercentage.innerText = `40%`;

  try {
    const res = await apiFetch("/api/downscale", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        image: state.stegoImgData,
        algorithm: state.attackAlgo,
        scale: state.attackScale,
      }),
    });
    if (!res.ok) throw new Error(`Server returned HTTP ${res.status}`);
    const data = await res.json(); // { downscaled_image, size }

    state.downscaledImgData = data.downscaled_image;
    state.downscaledUploaded = false;
    state.attackSimulated = true;

    await drawDataUrlToCanvas(elements.attackOutputCanvas, data.downscaled_image);

    elements.attackProgressFill.style.width = `100%`;
    elements.progressPercentage.innerText = `100%`;
    elements.progressStatusText.innerText = "Attack completed successfully.";

    elements.btnContinueToExtraction.disabled = false;
    elements.btnContinueToExtraction.classList.remove('disabled');

    elements.downscaledImportStatus.innerText = `Attacked image loaded (${data.size}x${data.size}px). Ready.`;

    const miniCtx = elements.miniDownscaledCanvas.getContext('2d');
    elements.miniDownscaledCanvas.width = 64;
    elements.miniDownscaledCanvas.height = 64;
    miniCtx.drawImage(elements.attackOutputCanvas, 0, 0, data.size, data.size, 0, 0, 64, 64);

    writeLog(`[SUCCESS] Real ${state.attackAlgo} resize complete via OpenCV. Output: ${data.size}x${data.size}px.`, "success");
  } catch (err) {
    const msg = describeFetchError(err);
    writeLog(`[ERROR] Downscale attack failed: ${msg}`, "error");
    alert(`Downscale attack failed: ${msg}`);
  } finally {
    elements.btnSimulateAttack.disabled = false;
    elements.btnSimulateAttack.innerText = "Simulate Downscaling Attack";
  }
}

// Page 4: calls POST /api/extract, running the real decoder (or real LSB
// extraction) on the actually-downscaled image, and gets a real BER back
// by comparing against the original message bits.
async function runExtractionAndEvaluation() {
  if (!state.downscaledImgData) {
    alert("Run the attack in Module 3, or upload an image on this page, first.");
    return;
  }

  state.method = document.getElementById('extract-method').value;
  const expected = document.getElementById('expected-msg-input').value;

  elements.btnExtractMessage.disabled = true;
  elements.btnExtractMessage.innerText = "Decrypting Payload...";
  elements.displayOriginalMsg.innerText = expected || "(unknown - extracting blind)";

  try {
    // Send the original cover too, so the server can report a real
    // PSNR/SSIM "after a resize round-trip" (downscaled image upscaled
    // back to 512x512 purely for this comparison — the decoder above never
    // sees this upscaled copy, only the real downscaled one).
    // Only send the cover when the image really came from this session's pipeline; an uploaded
    // image has no matching cover, and comparing against a blank canvas would give fake PSNR/SSIM.
    const useCover = state.embedded && !state.downscaledUploaded && !state.stegoUploaded;
    const coverDataUrl = useCover ? elements.resultCoverCanvas.toDataURL('image/png') : "";

    const res = await apiFetch("/api/extract", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        image: state.downscaledImgData,
        method: state.method,
        original_message: expected,
        cover_image: coverDataUrl,
      }),
    });
    if (!res.ok) throw new Error(`Server returned HTTP ${res.status}`);
    const data = await res.json(); // { recovered_message, ber, extracted_psnr, extracted_ssim }

    state.extracted = true;
    const berKnown = data.ber !== null && data.ber !== undefined;
    state.ber = berKnown ? data.ber * 100 : null;

    elements.displayExtractedMsg.innerText = data.recovered_message;
    elements.extractBer.innerText = berKnown ? `${state.ber.toFixed(2)} %` : 'N/A (no expected message)';

    if (state.method === 'lsb') {
      elements.berStatusIndicator.className = "m-indicator red";
      elements.extractedMsgLabel.innerText = "Recovered Text (M') - [LSB]";
    } else {
      elements.berStatusIndicator.className = !berKnown ? "m-indicator" : (state.ber < 5 ? "m-indicator green" : "m-indicator red");
      elements.extractedMsgLabel.innerText = "Recovered Text (M') - [CNN]";
    }

    // Real PSNR/SSIM: cover vs. (the actual downscaled image, upscaled back
    // to 512x512 for this comparison only). Computed server-side with the
    // same skimage functions eval.py uses, so it's a genuine measurement,
    // not an invented number — and it will legitimately read lower than
    // Page 2's embedding-time PSNR/SSIM, because it also reflects the
    // quality lost to the downscale+upscale round trip, not just embedding.
    state.extractedPsnr = data.extracted_psnr;
    state.extractedSsim = data.extracted_ssim;
    elements.extractPsnr.innerText = data.extracted_psnr != null ? `${data.extracted_psnr.toFixed(2)} dB` : 'N/A';
    elements.extractSsim.innerText = data.extracted_ssim != null ? data.extracted_ssim.toFixed(4) : 'N/A';

    if (berKnown) {
      updateBenchmarkTable();
      updateAcademicSummaryParagraph();
    }

    writeLog(`[SUCCESS] Extraction complete via server. BER: ${berKnown ? state.ber.toFixed(2) + '%' : 'n/a'}`, "success");
  } catch (err) {
    const msg = describeFetchError(err);
    writeLog(`[ERROR] Extraction failed: ${msg}`, "error");
    alert(`Extraction failed: ${msg}`);
  } finally {
    elements.btnExtractMessage.disabled = false;
    elements.btnExtractMessage.innerText = "Extract Hidden Message";
  }
}

// Dynamically updates comparison benchmark table: row 1 is the live run that
// just happened (real numbers from the server), row 2 is the other method's
// real aggregate average from the full eval.py benchmark (REFERENCE above).
function updateBenchmarkTable() {
  const scalePct = `${state.attackScale * 100}%`;
  const algoName = state.attackAlgo.charAt(0).toUpperCase() + state.attackAlgo.slice(1);

  const liveLabel = state.method === 'cnn' ? 'CNN (Deep/Robust)' : 'LSB (Standard)';
  const liveClass = state.method === 'cnn' ? 'cnn-cell' : 'lsb-cell';
  const liveValClass = state.ber < 10 ? 'better-val' : 'worse-val';

  const otherMethod = state.method === 'cnn' ? 'lsb' : 'cnn';
  const otherLabel = otherMethod === 'cnn' ? 'CNN (Deep/Robust)' : 'LSB (Standard)';
  const otherClass = otherMethod === 'cnn' ? 'cnn-cell' : 'lsb-cell';
  const otherValClass = otherMethod === 'cnn' ? 'better-val' : 'worse-val';
  const ref = REFERENCE[otherMethod];

  elements.evaluationTableRows.innerHTML = `
    <tr>
      <td class="method-cell ${liveClass}">${liveLabel} — this run</td>
      <td>${scalePct} (${algoName})</td>
      <td class="${liveValClass}">${state.ber.toFixed(2)}%</td>
      <td>${state.psnr.toFixed(2)} dB</td>
      <td>${state.ssim.toFixed(4)}</td>
    </tr>
    <tr>
      <td class="method-cell ${otherClass}">${otherLabel} — benchmark avg</td>
      <td>all algorithms, all sizes</td>
      <td class="${otherValClass}">${ref.ber.toFixed(2)}%</td>
      <td>${ref.psnr.toFixed(2)} dB</td>
      <td>${ref.ssim.toFixed(4)}</td>
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
    summary = `The deep CNN Encoder-Decoder model was tested against the <strong>${scalePct} ${algoName} Resizing</strong> attack, yielding a <strong>${state.ber.toFixed(2)}% Bit Error Rate (BER)</strong>. The decoder reconstructs the secret message with high fidelity because information was distributed globally within structural coefficients optimized by the differentiable downscale layer during training.`;
  }
  elements.evaluationSummaryText.innerHTML = summary;
}

// Initialize Application Events
function initApp() {

  elements.navButtons.forEach(btn => {
    btn.addEventListener('click', () => {
      const target = btn.getAttribute('data-target');
      handleNavigation(target);
    });
  });

  elements.toggleFigures.addEventListener('change', (e) => {
    if (e.target.checked) {
      document.body.classList.remove('hide-academic-figures');
    } else {
      document.body.classList.add('hide-academic-figures');
    }
  });

  elements.toggleTheme.addEventListener('change', (e) => {
    if (e.target.checked) {
      document.body.className = 'dark-theme';
    } else {
      document.body.className = 'light-theme';
    }
  });

  elements.sampleThumbButtons.forEach(btn => {
    btn.addEventListener('click', () => {
      elements.sampleThumbButtons.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');

      const sampleType = btn.getAttribute('data-sample');
      state.coverScene = sampleType;
      state.coverLoaded = true;

      drawProceduralImage(elements.coverCanvas, sampleType);
      writeLog(`[INFO] Loaded sample cover image scene: '${sampleType.toUpperCase()}'`, "info");
    });
  });

  elements.secretMessageInput.addEventListener('input', updatePayloadMetrics);

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

  elements.btnTriggerUpload.addEventListener('click', () => {
    elements.coverFileInput.click();
  });

  elements.coverFileInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (!file) return;

    elements.sampleThumbButtons.forEach(b => b.classList.remove('active'));
    elements.btnUseSample.classList.remove('active');
    elements.btnTriggerUpload.classList.add('active');

    const reader = new FileReader();
    reader.onload = (event) => {
      const img = new Image();
      img.onload = () => {
        const ctx = elements.coverCanvas.getContext('2d');
        ctx.clearRect(0, 0, 512, 512);
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
    const activeThumb = document.querySelector('.sample-thumb-btn.active');
    if (activeThumb) activeThumb.click();
  });

  elements.btnResetEmbedding.addEventListener('click', () => {
    elements.secretMessageInput.value = "Deep Steganography hides messages within image features, maintaining robustness against Downscaling Attacks.";
    updatePayloadMetrics();
    const thumbLena = document.querySelector('.sample-thumb-btn[data-sample="lena"]');
    if (thumbLena) thumbLena.click();
    writeLog("[SYSTEM] Configuration settings reset to academic defaults.", "system");
  });

  elements.btnEmbedPayload.addEventListener('click', runStegoPipeline);

  document.getElementById('login-form').addEventListener('submit', handleLogin);
  document.getElementById('btn-logout').addEventListener('click', handleLogout);
  silentRefresh();   // restore the session after a page reload (uses the httpOnly cookie)

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

  elements.btnDownloadStego.addEventListener('click', () => {
    const dataUrl = elements.resultStegoCanvas.toDataURL('image/png');
    const link = document.createElement('a');
    link.download = `stego_${state.method}_${state.coverScene}.png`;
    link.href = dataUrl;
    link.click();
  });

  elements.btnGoToDownscaling.addEventListener('click', () => {
    elements.btnImportLatest.click();
    handleNavigation('page-downscaling');
  });

  elements.btnImportLatest.addEventListener('click', () => {
    if (!state.embedded) {
      alert("No stego image has been embedded yet. Please complete Module 1 and 2.");
      return;
    }
    const miniCtx = elements.miniStegoCanvas.getContext('2d');
    miniCtx.clearRect(0, 0, 64, 64);
    miniCtx.drawImage(elements.resultStegoCanvas, 0, 0, elements.resultStegoCanvas.width, elements.resultStegoCanvas.height, 0, 0, 64, 64);

    const attInCtx = elements.attackInputCanvas.getContext('2d');
    elements.attackInputCanvas.width = elements.resultStegoCanvas.width;
    elements.attackInputCanvas.height = elements.resultStegoCanvas.height;
    attInCtx.drawImage(elements.resultStegoCanvas, 0, 0);

    elements.stegoSourceStatus.innerText = `Stego loaded. Method: ${state.method === 'cnn' ? 'Robust CNN' : 'Spatial LSB'}.`;

    writeLog("[INFO] Imported latest stego output image as attack simulator source.", "info");
  });

  elements.downscaleAlgorithm.addEventListener('change', (e) => {
    state.attackAlgo = e.target.value;
    updateThreatVectorInfo();
  });

  document.querySelectorAll('input[name="scale-factor"]').forEach(radio => {
    radio.addEventListener('change', (e) => {
      state.attackScale = parseFloat(e.target.value);
    });
  });

  elements.btnSimulateAttack.addEventListener('click', runDownscalingAttack);

  elements.btnContinueToExtraction.addEventListener('click', () => {
    elements.btnImportAttacked.click();
    handleNavigation('page-extraction');
  });

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

  elements.btnExtractMessage.addEventListener('click', runExtractionAndEvaluation);

  // keep the Page-4 decoder dropdown and expected-message box in sync with earlier steps
  document.getElementById('extract-method').addEventListener('change', (e) => { state.method = e.target.value; });
  elements.methodCards.forEach(card => card.addEventListener('click', () => {
    document.getElementById('extract-method').value = state.method;
  }));
  elements.btnEmbedPayload.addEventListener('click', () => {
    document.getElementById('expected-msg-input').value = state.message;
  });

  // ---- upload your own image (e.g. one received over Telegram) ----
  document.getElementById('btn-upload-stego').addEventListener('click', () => document.getElementById('stego-file-input').click());
  document.getElementById('stego-file-input').addEventListener('change', async (e) => {
    const file = e.target.files[0];
    e.target.value = '';
    try {
      const { dataUrl, w, h } = await readImageFile(file);
      state.stegoImgData = dataUrl;
      state.stegoUploaded = true;
      await drawDataUrlToCanvas(elements.attackInputCanvas, dataUrl);
      drawMini(elements.miniStegoCanvas, elements.attackInputCanvas);
      elements.stegoSourceStatus.innerText = `Uploaded image loaded (${w}x${h}px).`;
      elements.btnSimulateAttack.disabled = false;
      writeLog(`[INFO] Stego image uploaded from file (${w}x${h}px).`, "info");
    } catch (err) { alert(err.message); }
  });

  document.getElementById('btn-upload-attacked').addEventListener('click', () => document.getElementById('attacked-file-input').click());
  document.getElementById('attacked-file-input').addEventListener('change', async (e) => {
    const file = e.target.files[0];
    e.target.value = '';
    try {
      const { dataUrl, w, h } = await readImageFile(file);
      state.downscaledImgData = dataUrl;
      state.downscaledUploaded = true;
      state.attackSimulated = true;
      await drawDataUrlToCanvas(elements.attackOutputCanvas, dataUrl);
      drawMini(elements.miniDownscaledCanvas, elements.attackOutputCanvas);
      elements.downscaledImportStatus.innerText = `Uploaded image loaded (${w}x${h}px). Ready.`;
      writeLog(`[INFO] Image for extraction uploaded from file (${w}x${h}px).`, "info");
    } catch (err) { alert(err.message); }
  });

  elements.btnExportReport.addEventListener('click', () => {
    elements.reportDate.innerText = new Date().toLocaleDateString('en-US', { year: 'numeric', month: 'long', day: 'numeric' });
    elements.reportCoverName.innerText = `${state.coverScene.toUpperCase()} (512x512 px)`;
    elements.reportMsgLen.innerText = `${state.message.length} characters (${state.message.length * 8} bits)`;

    const scalePct = `${state.attackScale * 100}%`;
    const algoName = elements.downscaleAlgorithm.options[elements.downscaleAlgorithm.selectedIndex].text;

    elements.reportTableAlgo.innerText = algoName;
    elements.reportTableScale.innerText = `${scalePct} (${elements.attackOutputCanvas.width}x${elements.attackOutputCanvas.height} px)`;

    elements.reportLsbBer.innerText = `${REFERENCE.lsb.ber.toFixed(2)}%`;
    elements.reportCnnBer.innerText = `${REFERENCE.cnn.ber.toFixed(2)}%`;

    elements.reportModal.classList.add('active');
  });

  elements.btnCloseModal.addEventListener('click', () => {
    elements.reportModal.classList.remove('active');
  });

  elements.btnDismissModal.addEventListener('click', () => {
    elements.reportModal.classList.remove('active');
  });

  elements.btnPrintDocument.addEventListener('click', () => {
    window.print();
  });

  drawProceduralImage(elements.coverCanvas, 'lena');
  updatePayloadMetrics();
  updateThreatVectorInfo();
}

window.addEventListener('DOMContentLoaded', initApp);
