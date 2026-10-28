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

// Per-size averages from eval_summary.csv (mean over 100 images x 5 algorithms). These are the
// exact values the old static chart showed; the chart and size table are now drawn from here.
const BENCHMARK_BY_SIZE = {
  sizes:  [384, 299, 256, 224],
  scales: [0.75, 0.58398438, 0.5, 0.4375],   // must match the Page 3 radio values
  cnn:    [1.04, 1.18, 1.19, 1.27],
  lsb:    [44.3, 49.2, 44.7, 49.4],
};

// Display names for the built-in synthetic sample drawings (keys = data-sample values).
const SAMPLE_NAMES = { lena: 'Portrait', mandrill: 'Mandrill', satellite: 'Aerial', medical: 'Brain scan' };

// ONE set of success thresholds for the whole UI. Must match configs/config.yaml (eval section).
// The CNN's embedding PSNR (~32 dB) is BELOW the 35 dB target: that is a real, stated limitation.
const THRESHOLDS = { ber: 5, psnr: 35, ssim: 0.95 };

// Payload size is counted in UTF-8 BYTES (what the server embeds), not JS string length.
const utf8Bytes = (t) => new TextEncoder().encode(t).length;


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
  recoveredExact: null,      // true/false once extracted with an expected message
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
  const text = elements.secretMessageInput.value.replace(/\r\n/g, '\n');
  state.message = text;

  const chars = [...text].length;
  const bits = utf8Bytes(text) * 8;

  elements.charCounter.innerText = `${chars} / 120 chars`;
  elements.bitCounter.innerText = `${bits} / 1024 bits`;
}

function applyTheme(theme) {
  document.body.classList.toggle('dark-theme', theme === 'dark');
  document.body.classList.toggle('light-theme', theme === 'light');
  elements.toggleTheme.checked = theme === 'dark';
  try { localStorage.setItem('theme', theme); } catch (_) { /* theme preference only */ }
}

// SPA Routing
const PAGE_ORDER = ['page-embedding', 'page-result', 'page-downscaling', 'page-extraction'];

function handleNavigation(targetId) {
  elements.navButtons.forEach(btn => {
    if (btn.getAttribute('data-target') === targetId) {
      btn.classList.add('active');
      btn.setAttribute('aria-current', 'step');
      // Top bar mirrors the active step (label text comes from the nav itself).
      const label = btn.querySelector('.step-label');
      document.getElementById('topbar-heading').textContent = label ? label.textContent : '';
      document.getElementById('topbar-step').textContent =
        `Step ${PAGE_ORDER.indexOf(targetId) + 1} of ${PAGE_ORDER.length}`;
    } else {
      btn.classList.remove('active');
      btn.removeAttribute('aria-current');
    }
  });
  updateStepProgress();

  elements.viewports.forEach(vp => {
    if (vp.id === targetId) {
      vp.classList.add('active');
    } else {
      vp.classList.remove('active');
    }
  });
}

// Tick completed workflow steps in the sidebar. Purely visual: derived from existing state,
// never blocks navigation (uploads let you start at any step).
function updateStepProgress() {
  const done = {
    'page-embedding': state.embedded || state.stegoUploaded,
    'page-result': state.embedded,
    'page-downscaling': state.attackSimulated,
    'page-extraction': state.extracted,
  };
  elements.navButtons.forEach(btn => {
    btn.classList.toggle('done', Boolean(done[btn.getAttribute('data-target')]));
  });
  document.body.classList.toggle('has-embed', Boolean(state.embedded));
  const chip = document.getElementById('topbar-method');
  if (chip) chip.textContent = state.method === 'lsb' ? 'LSB' : 'CNN';
}

// Button labels live in a <span> so changing the text never deletes the icon.
function setBtnLabel(btn, text) {
  const span = btn.querySelector('span');
  if (span) span.textContent = text; else btn.textContent = text;
}

// Plain-language explanation of the selected resize algorithm
function updateThreatVectorInfo() {
  const algo = state.attackAlgo;
  let explanation = '';

  switch(algo) {
    case 'nearest':
      explanation = `<strong>Nearest Neighbor:</strong> resizes by rounding coordinates to the nearest pixel index: <code>f(x, y) = g(round(x/r), round(y/r))</code>. At the tested sizes (44%&ndash;75%) it simply skips 25%&ndash;56% of rows and columns. No averaging happens, so it causes aliasing, and the skipped pixels' bits are gone.`;
      break;
    case 'bilinear':
      explanation = `<strong>Bilinear Interpolation:</strong> performs 2D linear filtering using a 2x2 local pixel neighborhood: <code>W(x) = max(0, 1 - |x|)</code>. The resulting pixels are averages of adjacent locations, blurring away high-frequency spatial bits.`;
      break;
    case 'bicubic':
      explanation = `<strong>Bicubic Interpolation:</strong> computes pixel values using a 4x4 coordinate neighborhood and cubic spline convolution: <code>W(x) = (a+2)|x|³ - (a+3)|x|² + 1</code>. Each output pixel is a weighted blend of 16 inputs, which smooths away pixel-level detail such as LSB bits.`;
      break;
    case 'lanczos':
      explanation = `<strong>Lanczos Resampling:</strong> applies a windowed sinc kernel convolution filter: <code>L(x) = sinc(x)sinc(x/a)</code>. Held out of CNN training entirely, so it tests whether the decoder generalises to an unseen resize.`;
      break;
    case 'area':
      explanation = `<strong>Area-Based Resampling:</strong> computes average value of pixels under a pixel mapping box. This averages neighbouring pixels like a box blur, which removes pixel-level variations.`;
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
  setBtnLabel(elements.btnEmbedPayload, "Embedding…");

  writeLog("[INFO] Preprocessing cover pixel matrix (512x512x3)...", "info");
  writeLog(`[INFO] Message byte representation converted to bitstream (${utf8Bytes(state.message) * 8} bits, padded to 1024 and whitened).`, "info");

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
    updateStepProgress();
    state.stegoUploaded = false;
    state.stegoImgData = data.stego_image;
    state.psnr = data.psnr;
    state.ssim = data.ssim;

    await drawDataUrlToCanvas(elements.resultCoverCanvas, coverDataUrl);
    await drawDataUrlToCanvas(elements.resultStegoCanvas, data.stego_image);
    drawResidualMap(elements.resultCoverCanvas, elements.resultStegoCanvas, elements.resultResidualCanvas);

    if (state.method === 'lsb') {
      elements.methodTitleBadge.innerText = "LSB baseline";
      elements.methodDetailsText.innerText = "Least Significant Bit embedding injects message bitstream values directly into the lowest order bits of RGB color channels. While offering excellent image quality with close to zero visual distortions (high PSNR), it stores data in fragile pixel states that are easily destroyed by image scaling.";
      elements.cnnNetworkPreview.style.display = 'none';
    } else {
      elements.methodTitleBadge.innerText = "CNN encoder-decoder";
      elements.methodDetailsText.innerText = "The encoder and decoder were trained together with a differentiable downscaling layer between them (nearest, bilinear, bicubic and area, to 224-384 px), so the hidden bits are spread in a way that survives resizing. Lanczos was held out of training as a generalisation test. The cost is image quality: about 32 dB PSNR, versus about 80 dB for LSB.";
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
    setBtnLabel(elements.btnEmbedPayload, "Embed message");
  }
}

// Page 3: calls POST /api/downscale, which runs the exact OpenCV algorithm
// (nearest/bilinear/bicubic/lanczos/area) used in eval.py.
async function runDownscalingAttack() {
  if (!state.stegoImgData) {
    alert("Embed a message in step 1, or upload a stego image here, first.");
    return;
  }

  writeLog("[ATTACK] Sending stego image to server for real downscale attack...", "system");
  elements.btnSimulateAttack.disabled = true;
  setBtnLabel(elements.btnSimulateAttack, "Resizing…");
  elements.arrowScaleLabel.innerText = `${Math.round(state.attackScale * 100)}%`;

  await drawDataUrlToCanvas(elements.attackInputCanvas, state.stegoImgData);
  elements.attackInputCanvas.dataset.loaded = '1';

  const track = document.getElementById('attack-progress-track');
  elements.progressStatusText.innerText = `Resizing with ${state.attackAlgo} on the server…`;
  elements.progressPercentage.innerText = '';
  elements.attackProgressFill.style.width = '0%';
  track.classList.add('busy');
  track.removeAttribute('aria-valuenow');

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
    updateStepProgress();

    await drawDataUrlToCanvas(elements.attackOutputCanvas, data.downscaled_image);

    track.classList.remove('busy');
    track.setAttribute('aria-valuenow', '100');
    elements.attackProgressFill.style.width = '100%';
    elements.progressPercentage.innerText = `${data.size} × ${data.size} px`;
    elements.progressStatusText.innerText = "Done.";
    elements.attackOutputCanvas.dataset.loaded = '1';

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
    track.classList.remove('busy');
    track.setAttribute('aria-valuenow', '0');
    elements.progressStatusText.innerText = "Attack failed. See the activity log.";
    alert(`Downscale attack failed: ${msg}`);
  } finally {
    elements.btnSimulateAttack.disabled = false;
    setBtnLabel(elements.btnSimulateAttack, "Run downscaling attack");
  }
}

// Page 4: calls POST /api/extract, running the real decoder (or real LSB
// extraction) on the actually-downscaled image, and gets a real BER back
// by comparing against the original message bits.
async function runExtractionAndEvaluation() {
  if (!state.downscaledImgData) {
    alert("Run the attack in step 3, or upload an image on this page, first.");
    return;
  }

  state.method = document.getElementById('extract-method').value;
  // Normalise Windows line endings so a pasted message compares equal to the embedded one.
  const expected = document.getElementById('expected-msg-input').value.replace(/\r\n/g, '\n');

  elements.btnExtractMessage.disabled = true;
  setBtnLabel(elements.btnExtractMessage, "Decoding…");
  elements.displayOriginalMsg.textContent = expected || "(not given - extracting blind)";

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
    updateStepProgress();
    const berKnown = data.ber !== null && data.ber !== undefined;
    state.ber = berKnown ? data.ber * 100 : null;

    state.recoveredExact = expected ? data.recovered_message === expected : null;
    renderMessageDiff(expected, data.recovered_message);
    updateBerHero(berKnown);
    elements.extractedMsgLabel.innerText =
      state.method === 'lsb' ? "Recovered message (LSB decoder)" : "Recovered message (CNN decoder)";

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

    updateBenchmarkTable();
    renderBenchmarkChart();
    if (berKnown) {
      updateAcademicSummaryParagraph();
    }

    writeLog(`[SUCCESS] Extraction complete via server. BER: ${berKnown ? state.ber.toFixed(2) + '%' : 'n/a'}`, "success");
  } catch (err) {
    const msg = describeFetchError(err);
    writeLog(`[ERROR] Extraction failed: ${msg}`, "error");
    alert(`Extraction failed: ${msg}`);
  } finally {
    elements.btnExtractMessage.disabled = false;
    setBtnLabel(elements.btnExtractMessage, "Extract hidden message");
  }
}

// Comparison table. Before an extraction it shows the two benchmark rows; afterwards the first row
// is this run (server numbers) and the second is the OTHER method's benchmark average (REFERENCE).
function updateBenchmarkTable() {
  const fmtBer = (v) => `${v.toFixed(2)}%`;
  const verdict = (v) => (v <= THRESHOLDS.ber ? 'better-val' : 'worse-val');
  const name = (m) => (m === 'cnn' ? 'CNN' : 'LSB');
  const benchRow = (m) => {
    const ref = REFERENCE[m];
    return `<tr>
      <td><span class="method-cell ${m}-cell">${name(m)}</span><span class="cell-sub">benchmark average</span></td>
      <td class="${verdict(ref.ber)}">${fmtBer(ref.ber)}</td>
      <td>${ref.psnr.toFixed(2)} dB</td>
      <td>${ref.ssim.toFixed(4)}</td>
    </tr>`;
  };

  if (!state.extracted || state.ber === null) {
    elements.evaluationTableRows.innerHTML = benchRow('cnn') + benchRow('lsb');
    return;
  }

  // Embedding PSNR/SSIM exist only if the embed step ran in this session on this image.
  const embeddedHere = state.embedded && !state.stegoUploaded && !state.downscaledUploaded;
  const algoName = state.attackAlgo.charAt(0).toUpperCase() + state.attackAlgo.slice(1);
  const attack = state.downscaledUploaded
    ? 'uploaded image'
    : `${Math.round(state.attackScale * 100)}% ${algoName}`;
  const other = state.method === 'cnn' ? 'lsb' : 'cnn';

  elements.evaluationTableRows.innerHTML = `<tr class="run-row">
      <td><span class="method-cell ${state.method}-cell">${name(state.method)}</span><span class="cell-sub">this run &middot; ${attack}</span></td>
      <td class="${verdict(state.ber)}">${fmtBer(state.ber)}</td>
      <td>${embeddedHere ? `${state.psnr.toFixed(2)} dB` : '--'}</td>
      <td>${embeddedHere ? state.ssim.toFixed(4) : '--'}</td>
    </tr>` + benchRow(state.method) + benchRow(other);
}

// Headline BER card. States: idle (nothing yet), na (no expected message), pass, fail.
function updateBerHero(berKnown) {
  const card = document.getElementById('card-ber');
  const verdictEl = document.getElementById('ber-verdict');
  const bitsEl = document.getElementById('ber-bits');
  if (!berKnown) {
    card.dataset.state = 'na';
    elements.extractBer.textContent = 'N/A';
    bitsEl.textContent = '';
    verdictEl.textContent = 'Enter the expected message to measure BER.';
    elements.berStatusIndicator.className = 'm-indicator';
    return;
  }
  const pass = state.ber <= THRESHOLDS.ber;
  card.dataset.state = pass ? 'pass' : 'fail';
  elements.extractBer.textContent = `${state.ber.toFixed(2)}%`;
  // BER = wrong bits / 1024, so this count is exact, not an estimate.
  bitsEl.textContent = `${Math.round(state.ber * 1024 / 100)} of 1024 bits wrong`;
  verdictEl.textContent = pass ? `Within the ${THRESHOLDS.ber}% target` : `Above the ${THRESHOLDS.ber}% target`;
  elements.berStatusIndicator.className = pass ? 'm-indicator green' : 'm-indicator red';
}

// Character-by-character comparison. Recovered text comes from the server/image and is untrusted,
// so it is rendered ONLY through textContent (never innerHTML).
function renderMessageDiff(expected, recovered) {
  const out = elements.displayExtractedMsg;
  const summary = document.getElementById('diff-summary');
  out.textContent = '';
  if (!expected) {
    out.textContent = recovered || '(empty)';
    summary.hidden = true;
    return;
  }
  const exp = [...expected];
  const rec = [...recovered];
  let wrong = 0;
  const n = Math.max(exp.length, rec.length);
  for (let i = 0; i < n; i++) {
    const span = document.createElement('span');
    if (i >= rec.length) {             // decoder stopped early (a corrupted byte became the terminator)
      span.textContent = '\u00b7';
      span.className = 'diff-missing';
      span.title = 'missing';
      wrong++;
    } else {
      span.textContent = rec[i];
      if (i >= exp.length || rec[i] !== exp[i]) { span.className = 'diff-bad'; wrong++; }
    }
    out.appendChild(span);
  }
  const correct = exp.length - Math.min(wrong, exp.length);
  summary.hidden = false;
  summary.dataset.state = wrong === 0 ? 'pass' : 'fail';
  const padNote = state.ber > 0
    ? ` The ${Math.round(state.ber * 1024 / 100)} wrong bits fell in the unused padding after the message.` : '';
  summary.textContent = wrong === 0
    ? `Exact match: all ${exp.length} characters recovered.${padNote}`
    : `${correct} of ${exp.length} characters match. Wrong characters are highlighted; dots mark characters that were not recovered.`;
}

// Benchmark chart (SVG built with DOM APIs; colours come from CSS classes so both themes work).
function renderBenchmarkChart() {
  const host = document.getElementById('svg-chart-wrapper');
  const NS = 'http://www.w3.org/2000/svg';
  const W = 460, H = 230, L = 44, R = 16, T = 14, B = 40;
  const yMax = 55;
  const xs = BENCHMARK_BY_SIZE.sizes.map((_, i) => L + 40 + i * ((W - L - R - 80) / 3));
  const y = (v) => T + (H - T - B) * (1 - Math.min(v, yMax) / yMax);
  const el = (tag, attrs, text) => {
    const e = document.createElementNS(NS, tag);
    for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
    if (text !== undefined) e.textContent = text;
    return e;
  };

  const svg = el('svg', { viewBox: `0 0 ${W} ${H}`, width: '100%', role: 'img',
    'aria-label': 'Benchmark BER per output size. CNN about 1.0 to 1.3 percent, LSB about 44 to 49 percent.' });

  [0, 10, 20, 30, 40, 50].forEach(v => {
    svg.appendChild(el('line', { x1: L, x2: W - R, y1: y(v), y2: y(v), class: v === 0 ? 'chart-axis' : 'chart-grid' }));
    svg.appendChild(el('text', { x: L - 8, y: y(v) + 4, 'text-anchor': 'end', class: 'chart-tick' }, `${v}%`));
  });
  // target line
  svg.appendChild(el('line', { x1: L, x2: W - R, y1: y(THRESHOLDS.ber), y2: y(THRESHOLDS.ber), class: 'chart-target' }));

  BENCHMARK_BY_SIZE.sizes.forEach((px, i) => {
    svg.appendChild(el('text', { x: xs[i], y: H - B + 18, 'text-anchor': 'middle', class: 'chart-tick' }, `${px} px`));
    svg.appendChild(el('text', { x: xs[i], y: H - B + 32, 'text-anchor': 'middle', class: 'chart-tick chart-tick-sub' },
      `${Math.round(BENCHMARK_BY_SIZE.scales[i] * 100)}%`));
  });

  for (const m of ['lsb', 'cnn']) {
    const pts = BENCHMARK_BY_SIZE[m].map((v, i) => `${xs[i]},${y(v)}`).join(' ');
    svg.appendChild(el('polyline', { points: pts, class: `series-line series-${m}` }));
    BENCHMARK_BY_SIZE[m].forEach((v, i) => {
      const dot = el('circle', { cx: xs[i], cy: y(v), r: 4, class: `series-dot series-${m}` });
      dot.appendChild(el('title', {}, `${m.toUpperCase()} at ${BENCHMARK_BY_SIZE.sizes[i]} px: ${v}%`));
      svg.appendChild(dot);
    });
  }

  // this run (only when the attack size is known, i.e. the attack ran in this session)
  const idx = BENCHMARK_BY_SIZE.scales.indexOf(state.attackScale);
  const showRun = state.extracted && state.ber !== null && !state.downscaledUploaded && idx >= 0;
  document.getElementById('run-legend').hidden = !showRun;
  if (showRun) {
    const pass = state.ber <= THRESHOLDS.ber;
    const cx = xs[idx], cy = y(state.ber);
    svg.appendChild(el('circle', { cx, cy, r: 8, class: `run-marker ${pass ? 'pass' : 'fail'}` }));
    const above = cy > T + 30;
    svg.appendChild(el('text', { x: cx, y: above ? cy - 14 : cy + 22, 'text-anchor': 'middle', class: `run-label ${pass ? 'pass' : 'fail'}` },
      `This run ${state.ber.toFixed(2)}%`));
  }
  host.replaceChildren(svg);
}

// UI-only setup: nothing here touches the server, auth or measured numbers.
function initPhase3Ui() {
  // Sample thumbnails show the actual drawings instead of unrelated gradients.
  elements.sampleThumbButtons.forEach(btn => {
    const thumb = btn.querySelector('.thumb-img');
    if (!thumb) return;
    const c = document.createElement('canvas');
    c.width = 160; c.height = 160;
    drawProceduralImage(c, btn.getAttribute('data-sample'));
    thumb.style.backgroundImage = `url(${c.toDataURL('image/png')})`;
    thumb.classList.add('thumb-drawn');
  });

  // Toggle-style buttons expose their state to screen readers (aria-pressed follows .active).
  const pressables = document.querySelectorAll('.sample-thumb-btn, .source-btn, .mode-btn');
  const sync = (el) => el.setAttribute('aria-pressed', el.classList.contains('active') ? 'true' : 'false');
  pressables.forEach(sync);
  const mo = new MutationObserver(records => records.forEach(r => sync(r.target)));
  pressables.forEach(el => mo.observe(el, { attributes: true, attributeFilter: ['class'] }));

  // Empty-state shortcuts.
  document.querySelectorAll('[data-goto]').forEach(btn =>
    btn.addEventListener('click', () => handleNavigation(btn.getAttribute('data-goto'))));

  // Target ticks on the Page 2 gauges, from the same thresholds as everywhere else
  // (the PSNR gauge spans 0-60 dB, the SSIM gauge 0-1, matching how the bars are filled).
  const psnrTick = document.querySelector('[data-target-for="psnr"]');
  const ssimTick = document.querySelector('[data-target-for="ssim"]');
  if (psnrTick) psnrTick.style.left = `${(THRESHOLDS.psnr / 60) * 100}%`;
  if (ssimTick) ssimTick.style.left = `${THRESHOLDS.ssim * 100}%`;
}

// Small exact-number table under the chart (the chart shows shape, this shows values).
function renderSizeTable() {
  const row = (m) => `<tr><td><span class="method-cell ${m}-cell">${m.toUpperCase()}</span></td>` +
    BENCHMARK_BY_SIZE[m].map(v => `<td class="${v <= THRESHOLDS.ber ? 'better-val' : 'worse-val'}">${v}%</td>`).join('') + '</tr>';
  document.getElementById('size-table-rows').innerHTML = row('cnn') + row('lsb');
}

// Research summary for the run that just happened. The wording follows the MEASURED result
// (against THRESHOLDS.ber), so a failed run is never described as a success.
function updateAcademicSummaryParagraph() {
  const algoName = state.attackAlgo.charAt(0).toUpperCase() + state.attackAlgo.slice(1);
  const scalePct = `${Math.round(state.attackScale * 100)}%`;
  const ber = state.ber.toFixed(2);
  const ok = state.ber <= THRESHOLDS.ber;
  const exact = state.recoveredExact === true
    ? ' The recovered text matches the expected message exactly.'
    : (state.recoveredExact === false ? ' The recovered text still contains errors (no error-correcting code is used yet).' : '');
  const attack = `<strong>${scalePct} ${algoName}</strong> downscaling`;

  let summary;
  if (state.method === 'lsb') {
    summary = ok
      ? `LSB extraction after ${attack} gave <strong>${ber}% BER</strong>, within the ${THRESHOLDS.ber}% threshold.${exact} This is unusual for LSB and is not what the benchmark shows (48.00% average); check that the image was actually downscaled.`
      : `LSB extraction after ${attack} gave <strong>${ber}% BER</strong>, above the ${THRESHOLDS.ber}% threshold.${exact} Resizing resamples pixel values, so the least significant bits do not survive. The benchmark average for LSB is 48.00%, no better than guessing.`;
  } else {
    summary = ok
      ? `The CNN decoder recovered the payload after ${attack} with <strong>${ber}% BER</strong>, within the ${THRESHOLDS.ber}% threshold.${exact} The benchmark average across 4,000 attacks is 1.15%.`
      : `The CNN decoder gave <strong>${ber}% BER</strong> after ${attack}, above the ${THRESHOLDS.ber}% threshold.${exact} This is worse than the 1.15% benchmark average. Likely causes: a cover image unlike the natural photos used in training (e.g. flat colours or graphics), or an image altered by something other than the selected resize (e.g. JPEG re-compression).`;
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

  elements.toggleTheme.addEventListener('change', (e) => applyTheme(e.target.checked ? 'dark' : 'light'));
  let savedTheme = null;
  try { savedTheme = localStorage.getItem('theme'); } catch (_) { /* storage blocked: use default */ }
  if (savedTheme === 'light' || savedTheme === 'dark') applyTheme(savedTheme);

  elements.sampleThumbButtons.forEach(btn => {
    btn.addEventListener('click', () => {
      elements.sampleThumbButtons.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');

      const sampleType = btn.getAttribute('data-sample');
      state.coverScene = sampleType;
      state.coverLoaded = true;

      drawProceduralImage(elements.coverCanvas, sampleType);
      writeLog(`[INFO] Sample '${SAMPLE_NAMES[sampleType] || sampleType}' loaded (synthetic drawing).`, "info");
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
      updateStepProgress();
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
      alert("No stego image yet. Embed a message in step 1 first.");
      return;
    }
    const miniCtx = elements.miniStegoCanvas.getContext('2d');
    miniCtx.clearRect(0, 0, 64, 64);
    miniCtx.drawImage(elements.resultStegoCanvas, 0, 0, elements.resultStegoCanvas.width, elements.resultStegoCanvas.height, 0, 0, 64, 64);

    const attInCtx = elements.attackInputCanvas.getContext('2d');
    elements.attackInputCanvas.width = elements.resultStegoCanvas.width;
    elements.attackInputCanvas.height = elements.resultStegoCanvas.height;
    attInCtx.drawImage(elements.resultStegoCanvas, 0, 0);
    elements.attackInputCanvas.dataset.loaded = '1';

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
      alert("No attacked image yet. Run the attack in step 3 first.");
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
      updateStepProgress();
      await drawDataUrlToCanvas(elements.attackInputCanvas, dataUrl);
      elements.attackInputCanvas.dataset.loaded = '1';
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
      updateStepProgress();
      await drawDataUrlToCanvas(elements.attackOutputCanvas, dataUrl);
      drawMini(elements.miniDownscaledCanvas, elements.attackOutputCanvas);
      elements.downscaledImportStatus.innerText = `Uploaded image loaded (${w}x${h}px). Ready.`;
      writeLog(`[INFO] Image for extraction uploaded from file (${w}x${h}px).`, "info");
    } catch (err) { alert(err.message); }
  });

  elements.btnExportReport.addEventListener('click', () => {
    elements.reportDate.innerText = new Date().toLocaleDateString('en-US', { year: 'numeric', month: 'long', day: 'numeric' });
    const coverName = SAMPLE_NAMES[state.coverScene] ? `${SAMPLE_NAMES[state.coverScene]} (synthetic sample)` : state.coverScene;
    elements.reportCoverName.innerText = `${coverName}, 512x512 px`;
    elements.reportMsgLen.innerText = `${[...state.message].length} characters (${utf8Bytes(state.message) * 8} of 1024 bits)`;

    const scalePct = `${Math.round(state.attackScale * 100)}%`;
    const algoName = elements.downscaleAlgorithm.options[elements.downscaleAlgorithm.selectedIndex].text;

    const attacked = state.attackSimulated && !state.downscaledUploaded;
    elements.reportTableAlgo.innerText = attacked ? algoName : (state.downscaledUploaded ? 'Uploaded image (attack not run in this app)' : '--');
    elements.reportTableScale.innerText = attacked
      ? `${scalePct} (${elements.attackOutputCanvas.width}x${elements.attackOutputCanvas.height} px)` : '--';

    // Benchmark rows: always the real eval.py averages (REFERENCE), never per-run numbers.
    elements.reportLsbBer.innerText = `${REFERENCE.lsb.ber.toFixed(2)}%`;
    elements.reportCnnBer.innerText = `${REFERENCE.cnn.ber.toFixed(2)}%`;
    document.getElementById('report-lsb-psnr').innerText = `${REFERENCE.lsb.psnr.toFixed(2)} dB`;
    document.getElementById('report-cnn-psnr').innerText = `${REFERENCE.cnn.psnr.toFixed(2)} dB`;
    document.getElementById('report-lsb-ssim').innerText = REFERENCE.lsb.ssim.toFixed(4);
    document.getElementById('report-cnn-ssim').innerText = REFERENCE.cnn.ssim.toFixed(4);

    // This session's run: only what was actually measured; "--" when a step wasn't done.
    document.getElementById('report-run-method').innerText =
      state.extracted ? (state.method === 'cnn' ? 'CNN encoder-decoder' : 'LSB') : '--';
    document.getElementById('report-run-ber').innerText =
      state.extracted && state.ber !== null ? `${state.ber.toFixed(2)}%` : '-- (no extraction with an expected message)';
    document.getElementById('report-run-exact').innerText =
      state.recoveredExact === true ? 'Yes' : state.recoveredExact === false ? 'No' : '--';

    elements.reportModal.classList.add('active');
    elements.btnCloseModal.focus();
  });

  const closeReport = () => {
    elements.reportModal.classList.remove('active');
    elements.btnExportReport.focus();
  };
  elements.btnCloseModal.addEventListener('click', closeReport);
  elements.btnDismissModal.addEventListener('click', closeReport);
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && elements.reportModal.classList.contains('active')) closeReport();
  });

  elements.btnPrintDocument.addEventListener('click', () => {
    window.print();
  });

  drawProceduralImage(elements.coverCanvas, 'lena');
  updatePayloadMetrics();
  updateThreatVectorInfo();
  updateStepProgress();
  document.getElementById('ber-target-value').textContent = THRESHOLDS.ber;
  renderBenchmarkChart();
  renderSizeTable();
  updateBenchmarkTable();
  initPhase3Ui();
}

window.addEventListener('DOMContentLoaded', initApp);
