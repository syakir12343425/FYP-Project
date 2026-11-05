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
// Local development: UI on :8080, API on :8000. Deployed: Caddy serves the UI and the API on ONE https origin,
// so requests go to the same site ('' = relative URLs) and the cookie stays first-party.
const API_BASE = (['localhost', '127.0.0.1'].includes(location.hostname) && location.port === '8080')
  ? `http://${location.hostname}:8000` : '';

// Real aggregate numbers, all four methods from ONE tool and the same attacks (3 Nov 2026):
// tools/benchmark_methods.py --images data/DIV2K_valid_HR --limit 100
// = 100 DIV2K validation images x 5 algorithms x 4 sizes = 2,000 attacks per method.
// CNN = model D: colour penalty + messaging-app training (checkpoints_channel_l300/best.pt, results/div2k_cnn_D.csv);
// LSB/DCT/DWT = results/div2k_methods_default.csv (DCT T=50, DWT S=128, the server defaults).
// exact = share of attacks where the whole message was recovered after Reed-Solomon.
// Used as reference values in the comparison table / report so the UI never shows invented numbers.
const BENCH_ATTACKS = 2000;
const REFERENCE = {
  cnn: { ber: 1.76, psnr: 35.54, ssim: 0.9767, exact: 88.9, worst: 16.89 },
  lsb: { ber: 45.43, psnr: 79.98, ssim: 1.0000, exact: 0.0, worst: 63.18 },
  dct: { ber: 4.19, psnr: 40.12, ssim: 0.9739, exact: 64.0, worst: 30.96 },
  dwt: { ber: 7.14, psnr: 40.76, ssim: 0.9885, exact: 70.3, worst: 50.49 },
};

// WhatsApp stickers (Results log 9d, 9c-2): tools/sticker_test.py on the 24 Kodak photos at WebP quality 90,
// the setting measured from real WhatsApp Web stickers. CNN = model D (the app's model).
const STICKER_RESULTS = {
  photos: 24,
  cnn: { survived: 22, ber: 2.18 },
  dct: { survived: 24, ber: 0.01 },
  dwt: { survived: 24, ber: 0.56 },
  lsb: { survived: 0, ber: 49.83 },
};
// One real send per method through WhatsApp Web (3 Nov 2026, cartoon sticker covers; CNN made with the
// earlier model A). DWT was not sent.
const REAL_SEND = {
  cnn: 'Exact (2.73% BER, model A)',
  dct: 'Exact (0.68% BER)',
  dwt: 'Not tested',
  lsb: 'Failed (52.15% BER)',
};
const METHOD_ORDER = ['cnn', 'dct', 'dwt', 'lsb'];

// Per-size mean BER from the same runs (mean over 100 images x 5 algorithms).
const BENCHMARK_BY_SIZE = {
  sizes:  [384, 299, 256, 224],
  scales: [0.75, 0.58398438, 0.5, 0.4375],   // must match the Page 3 radio values
  cnn:    [1.30, 1.73, 1.85, 2.15],
  lsb:    [39.56, 45.56, 47.12, 49.48],
  dct:    [0.90, 4.06, 3.15, 8.65],
  dwt:    [4.95, 7.41, 4.72, 11.47],
};

// The four embedding methods. Every label in the UI reads from here.
// bench: true where REFERENCE holds DIV2K numbers (now all four methods).
const METHODS = {
  cnn: {
    short: 'CNN', long: 'CNN encoder-decoder', bench: true,
    log: 'Running CNN encoder (model D, checkpoints_channel_l300/best.pt)...',
    details: 'The encoder and decoder were trained together with a differentiable downscaling layer between them (nearest, bilinear, bicubic and area, to 224-384 px), so the hidden bits are spread in a way that survives resizing. Lanczos was held out of training as a generalisation test. The first model hid the message mostly in colour and left a visible tint (about 32 dB PSNR). This model was fine-tuned with a penalty on colour change and with WhatsApp-style attacks (JPEG, WebP, no resize) added to training: about 35.5 dB PSNR, and a faint colour pattern can still be seen in large white areas when zoomed in.',
    failHint: 'Likely causes: a cover image unlike the natural photos used in training (e.g. flat colours or graphics), or an image altered by something other than the selected resize (e.g. JPEG re-compression).',
  },
  lsb: {
    short: 'LSB', long: 'LSB baseline', bench: true,
    log: 'Running spatial LSB embed (lowest bit plane)...',
    details: 'Least Significant Bit embedding writes each message bit into the lowest bit of a pixel value. The change is invisible (about 80 dB PSNR), but resizing recomputes every pixel value, so those exact bits do not survive.',
    failHint: 'Resizing resamples pixel values, so the least significant bits do not survive. The benchmark average for LSB is 45.43%, no better than guessing.',
  },
  dct: {
    short: 'DCT', long: 'DCT (coefficient pair)', bench: true,
    log: 'Running DCT embed (8x8 blocks, coefficient pair (2,3)/(3,2))...',
    details: 'Classical frequency-domain method. The luminance is split into 8x8 blocks; each bit sets the order of two mid-frequency DCT coefficients in one block. Those frequencies mostly survive downscaling, with more errors at smaller sizes. About 40 dB PSNR at the default strength.',
    failHint: 'The DCT pair sits close to the highest frequency a small output keeps, so errors grow at 224-256 px and on fine textures.',
  },
  dwt: {
    short: 'DWT', long: 'DWT (LL-band QIM)', bench: true,
    log: 'Running DWT embed (4-level Haar, LL-band QIM)...',
    details: 'Classical wavelet method. A 4-level Haar transform gives one low-frequency coefficient per 16x16 block; each bit is set by quantising that coefficient (QIM), which shifts the block brightness slightly. Block averages survive averaging resizes very well. About 41 dB PSNR at the default step.',
    failHint: 'The DWT relies on block averages; nearest-neighbour resizing keeps single pixels instead of averaging them, which is its known weak case.',
  },
};
const methodInfo = (m) => METHODS[m] || METHODS.cnn;

// Display names for the built-in synthetic sample drawings (keys = data-sample values).
const SAMPLES = {
  astronaut: { name: 'Astronaut', file: 'assets/samples/astronaut.jpg' },
  coffee:    { name: 'Coffee',    file: 'assets/samples/coffee.jpg' },
  cat:       { name: 'Cat',       file: 'assets/samples/cat.jpg' },
  rocket:    { name: 'Rocket',    file: 'assets/samples/rocket.jpg' },
};
const SAMPLE_NAMES = Object.fromEntries(Object.entries(SAMPLES).map(([k, v]) => [k, v.name]));

// Payload layout (must match payload.py): 63 message bytes + 1 length byte + 64 Reed-Solomon parity bytes.
const MAX_MESSAGE_BYTES = 63;

// Attack size as the supervisor asked: "25% smaller" rather than "75%" (width and height).
const smallerLabel = (scale) => `${Math.round((1 - scale) * 100)}% smaller`;

// ONE set of success thresholds for the whole UI. Must match configs/config.yaml (eval section).
// The first CNN (~32 dB) was below the 35 dB target; model D (35.54 dB on DIV2K) meets it.
const THRESHOLDS = { ber: 5, psnr: 35, ssim: 0.95 };

// Payload size is counted in UTF-8 BYTES (what the server embeds), not JS string length.
const utf8Bytes = (t) => new TextEncoder().encode(t).length;


// ------------------------------------------------------------------ auth
// Access token lives ONLY in this JS variable (never localStorage -> not stealable via XSS storage reads).
// The refresh token is an httpOnly+SameSite=Strict cookie the JS can't even see.
let accessToken = null;
let refreshTimer = null;

// Role and must-change flag only shape the UI; the server re-checks both on every request.
let session = { username: '', role: 'user', mustChange: false, mfaEnabled: false };

function setSession(data) {
  accessToken = data.access_token;
  session = { username: data.username, role: data.role || 'user', mustChange: Boolean(data.must_change_password),
              mfaEnabled: Boolean(data.mfa_enabled), mfaRequired: Boolean(data.mfa_mandatory) };
  showLoginNotice('');
  document.getElementById('login-resend').hidden = true;
  // Two-step verification is compulsory: until it is on, only the Account page is available.
  if (session.mfaRequired && !session.mfaEnabled) document.body.dataset.mfaSetup = '1';
  else delete document.body.dataset.mfaSetup;
  document.getElementById('current-user').innerText = data.username;
  document.body.dataset.role = session.role;
  document.getElementById('login-overlay').classList.remove('active');
  // The Start page entrance plays once, when the app first opens for a signed-in person.
  if (!document.body.dataset.painted && document.body.dataset.mode === 'home') {
    document.body.dataset.painted = '1';
    document.body.classList.add('first-paint');
    setTimeout(() => document.body.classList.remove('first-paint'), 1200);
  }
  if (session.mustChange) openPasswordDialog(true);
  else if (document.body.dataset.mfaSetup) handleNavigation('page-account');
  else if (session.role !== 'admin' && document.body.dataset.mode === 'users') handleNavigation('page-home');
  if (document.body.dataset.mode === 'account') loadMfaStatus();
  clearTimeout(refreshTimer);
  refreshTimer = setTimeout(silentRefresh, Math.max(10, data.expires_in - 60) * 1000);
}

function clearSession() {
  accessToken = null;
  session = { username: '', role: 'user', mustChange: false, mfaEnabled: false };
  delete document.body.dataset.role;
  showPasswordLoginStep();
  resetMfaPanels();
  closePasswordDialog();
  hideTempPassword();
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
  if (res.status === 403 && session.mustChange) { openPasswordDialog(true); throw new Error("Choose a new password first."); }
  if (res.status === 403 && document.body.dataset.mfaSetup) { handleNavigation('page-account'); throw new Error("Set up two-step verification first."); }
  return res;
}

// Reads FastAPI's {"detail": "..."} (a string, or a list of validation errors) into one sentence.
async function errorDetail(res, fallback) {
  try {
    const body = await res.json();
    if (typeof body.detail === 'string') return body.detail;
    if (Array.isArray(body.detail) && body.detail[0] && body.detail[0].msg) return body.detail[0].msg;
  } catch (_) { /* not JSON */ }
  return fallback;
}

// POST to a session endpoint that needs both the bearer token and the CSRF header.
async function sessionPost(path, body) {
  const run = () => fetch(`${API_BASE}${path}`, {
    method: 'POST', credentials: 'include',
    headers: { 'Content-Type': 'application/json', 'X-Requested-With': 'fetch', 'Authorization': `Bearer ${accessToken}` },
    body: JSON.stringify(body || {}),
  });
  let res = await run();
  if (res.status === 401 && await silentRefresh()) res = await run();
  if (res.status === 429) throw new Error('Too many attempts. Wait a minute and try again.');
  if (!res.ok) throw new Error(await errorDetail(res, `Request failed (${res.status}).`));
  return res.json();
}

// ------------------------------------------------------------ change password
function openPasswordDialog(forced) {
  const ov = document.getElementById('password-overlay');
  ov.dataset.forced = String(forced);
  document.getElementById('password-title').textContent = forced ? 'Choose your own password' : 'Change password';
  document.getElementById('password-intro').textContent = forced
    ? 'You signed in with a one-time password from your admin. Choose your own password to continue.'
    : 'Other devices where you are signed in will be signed out.';
  document.getElementById('password-current-label').textContent = forced ? 'One-time password' : 'Current password';
  document.getElementById('password-username').value = session.username;
  refreshPasswordRules();
  document.getElementById('password-error').textContent = '';
  ov.classList.add('active');
  document.getElementById('password-current').focus();
}

function closePasswordDialog() {
  const ov = document.getElementById('password-overlay');
  ov.classList.remove('active');
  ['password-current', 'password-new', 'password-repeat'].forEach(id => { document.getElementById(id).value = ''; });
  refreshPasswordRules();
}

async function handlePasswordChange(e) {
  e.preventDefault();
  const err = document.getElementById('password-error');
  const btn = document.getElementById('password-submit');
  const current = document.getElementById('password-current').value;
  const next = document.getElementById('password-new').value;
  err.textContent = '';
  if (next !== document.getElementById('password-repeat').value) { err.textContent = 'The two new passwords are different.'; return; }
  btn.disabled = true;
  try {
    const run = () => fetch(`${API_BASE}/api/auth/change-password`, {
      method: 'POST', credentials: 'include',
      headers: { 'Content-Type': 'application/json', 'X-Requested-With': 'fetch', 'Authorization': `Bearer ${accessToken}` },
      body: JSON.stringify({ current_password: current, new_password: next }),
    });
    let res = await run();
    if (res.status === 401 && await silentRefresh()) res = await run();
    if (res.status === 429) throw new Error('Too many attempts. Wait a minute and try again.');
    if (!res.ok) throw new Error(await errorDetail(res, 'Could not change the password.'));
    closePasswordDialog();
    setSession(await res.json());
  } catch (ex) {
    err.textContent = ex instanceof TypeError ? describeFetchError(ex) : ex.message;
  } finally {
    btn.disabled = false;
  }
}

// ------------------------------------------------------------ users page (admins)
const fmtDate = (iso) => iso
  ? new Date(iso).toLocaleString('en-GB', { day: 'numeric', month: 'short', year: 'numeric', hour: '2-digit', minute: '2-digit' })
  : 'Never';

function userStatus(u) {
  if (u.pending_approval) return ['Waiting for approval', 'warn'];
  if (u.email && !u.email_verified && !u.disabled) return ['Email not confirmed', 'warn'];
  if (u.disabled) return ['Disabled', 'off'];
  if (u.locked) return ['Locked (too many wrong passwords)', 'warn'];
  if (u.must_change_password) return ['Waiting for first sign-in', 'warn'];
  return ['Active', 'ok'];
}

function showTempPassword(title, pw) {
  document.getElementById('temp-password-title').textContent = title;
  document.getElementById('temp-password-value').textContent = pw;
  document.getElementById('temp-password-box').hidden = false;
}
function hideTempPassword() {
  document.getElementById('temp-password-value').textContent = '';
  document.getElementById('temp-password-box').hidden = true;
}

async function adminCall(path, method = 'POST', body) {
  const res = await apiFetch(path, {
    method, headers: body ? { 'Content-Type': 'application/json' } : {},
    body: body ? JSON.stringify(body) : undefined,
  });
  if (!res.ok) throw new Error(await errorDetail(res, `Request failed (${res.status}).`));
  return res.json();
}

function renderUsers(users) {
  const tbody = document.getElementById('users-tbody');
  tbody.replaceChildren();
  const waiting = users.filter(u => u.pending_approval).length;
  document.getElementById('users-count').textContent = `${users.length} account${users.length === 1 ? '' : 's'}`
    + (waiting ? `, ${waiting} waiting for approval` : '');
  for (const u of users) {
    const tr = document.createElement('tr');
    const cell = (text, cls) => { const td = document.createElement('td'); td.textContent = text; if (cls) td.className = cls; tr.appendChild(td); return td; };
    const nameCell = cell(u.username, 'user-name');
    if (u.email) {
      const em = document.createElement('span');
      em.className = 'user-email';
      em.textContent = u.email + (u.email_verified ? '' : ' (not confirmed yet)');
      nameCell.appendChild(em);
    }
    cell(u.role === 'admin' ? 'Admin' : 'User');
    const [label, tone] = userStatus(u);
    const st = cell('');
    const pill = document.createElement('span'); pill.className = `status-pill ${tone}`; pill.textContent = label; st.appendChild(pill);
    cell(u.mfa_enabled ? 'On' : 'Off', u.mfa_enabled ? 'mfa-on' : 'cell-note');
    cell(fmtDate(u.last_login_at));
    const actions = document.createElement('div'); actions.className = 'user-actions';
    cell('', 'user-actions-cell').appendChild(actions);
    const add = (text, fn, cls = 'btn-secondary') => {
      const b = document.createElement('button');
      b.type = 'button'; b.className = `btn ${cls} btn-sm`; b.textContent = text;
      b.setAttribute('aria-label', `${text}: ${u.username}`);
      b.addEventListener('click', async () => {
        b.disabled = true;
        document.getElementById('users-error').textContent = '';
        try { await fn(); } catch (ex) { document.getElementById('users-error').textContent = ex.message; }
        finally { b.disabled = false; }
      });
      actions.appendChild(b);
    };
    if (u.locked) add('Unlock', async () => { await adminCall(`/api/admin/users/${u.id}/unlock`); await loadUsers(); });
    if (u.pending_approval) {
      add('Approve', async () => { await adminCall(`/api/admin/users/${u.id}/approve`); await loadUsers(); }, 'btn-primary');
      add('Decline', async () => { await adminCall(`/api/admin/users/${u.id}/decline`); await loadUsers(); }, 'btn-danger-outline');
    } else if (u.role !== 'admin') {
      add('Reset password', async () => {
        const r = await adminCall(`/api/admin/users/${u.id}/reset-password`);
        showTempPassword(`New one-time password for ${r.user.username}`, r.temporary_password);
        await loadUsers();
      });
      if (u.mfa_enabled) add('Reset two-step', async () => { await adminCall(`/api/admin/users/${u.id}/reset-mfa`); await loadUsers(); });
      if (u.disabled) add('Enable', async () => { await adminCall(`/api/admin/users/${u.id}/enable`); await loadUsers(); });
      else add('Disable', async () => { await adminCall(`/api/admin/users/${u.id}/disable`); await loadUsers(); }, 'btn-danger-outline');
    } else if (!u.locked) {
      const note = document.createElement('span'); note.className = 'cell-note'; note.textContent = 'Managed on the server';
      actions.appendChild(note);
    }
    tbody.appendChild(tr);
  }
}

async function loadUsers() {
  const err = document.getElementById('users-error');
  try {
    const data = await adminCall('/api/admin/users', 'GET');
    err.textContent = '';
    document.getElementById('users-mfa-needed').hidden = true;
    renderUsers(data.users);
  } catch (ex) {
    const needsMfa = /two-step/i.test(ex.message);
    document.getElementById('users-mfa-needed').hidden = !needsMfa;
    err.textContent = needsMfa ? '' : (ex instanceof TypeError ? describeFetchError(ex) : ex.message);
    if (needsMfa) document.getElementById('users-tbody').replaceChildren();
  }
}

async function handleCreateUser(e) {
  e.preventDefault();
  const input = document.getElementById('user-create-name');
  const err = document.getElementById('user-create-error');
  const btn = document.getElementById('user-create-submit');
  err.textContent = '';
  btn.disabled = true;
  try {
    const r = await adminCall('/api/admin/users', 'POST', { username: input.value.trim() });
    input.value = '';
    showTempPassword(`Account created for ${r.user.username}. One-time password:`, r.temporary_password);
    await loadUsers();
  } catch (ex) {
    err.textContent = ex instanceof TypeError ? describeFetchError(ex) : ex.message;
  } finally {
    btn.disabled = false;
  }
}

// ------------------------------------------------------------ account page: two-step verification
let recoveryCodes = [];

function showOnly(id) {
  ['mfa-off', 'mfa-step-password', 'mfa-step-scan', 'mfa-codes', 'mfa-on'].forEach(x => {
    document.getElementById(x).hidden = x !== id;
  });
}

function resetMfaPanels() {
  recoveryCodes = [];
  document.getElementById('mfa-codes-list').replaceChildren();
  ['mfa-setup-password', 'mfa-enable-code', 'mfa-manage-password', 'mfa-manage-code'].forEach(id => { document.getElementById(id).value = ''; });
  ['mfa-setup-error', 'mfa-enable-error', 'mfa-manage-error'].forEach(id => { document.getElementById(id).textContent = ''; });
  document.getElementById('mfa-secret').textContent = '';
  const qr = document.getElementById('mfa-qr'); qr.hidden = true; qr.removeAttribute('src');
}

function renderMfaStatus(st) {
  const pill = document.getElementById('mfa-pill');
  pill.textContent = st.mfa_enabled ? 'On' : 'Off';
  pill.className = `status-pill ${st.mfa_enabled ? 'ok' : 'off'}`;
  document.getElementById('mfa-required-note').hidden = !(st.required && !st.mfa_enabled);
  document.getElementById('mfa-required-note').textContent = session.role === 'admin'
    ? 'Two-step verification is required on this site. Turn it on to start using RobustStego and to manage accounts.'
    : 'Two-step verification is required on this site. Turn it on to start using RobustStego.';
  document.getElementById('mfa-disable').hidden = Boolean(st.required);
  if (st.mfa_enabled) {
    const n = st.recovery_codes_left;
    document.getElementById('mfa-codes-left').textContent =
      `${n} recovery code${n === 1 ? '' : 's'} left.` + (n <= 3 ? ' Get new ones soon.' : '');
    showOnly('mfa-on');
  } else {
    showOnly('mfa-off');
  }
}

async function loadMfaStatus() {
  if (!accessToken) return;
  resetMfaPanels();
  try {
    const res = await apiFetch('/api/auth/mfa/status');
    if (!res.ok) throw new Error(await errorDetail(res, 'Could not load two-step status.'));
    renderMfaStatus(await res.json());
  } catch (ex) {
    showOnly('mfa-off');
    document.getElementById('mfa-setup-error').textContent = ex.message;
  }
}

function showRecoveryCodes(codes) {
  recoveryCodes = codes;
  const list = document.getElementById('mfa-codes-list');
  list.replaceChildren(...codes.map(c => { const li = document.createElement('li'); li.textContent = c; return li; }));
  showOnly('mfa-codes');
}

function initMfaPanels() {
  document.getElementById('mfa-start').addEventListener('click', () => {
    showOnly('mfa-step-password');
    document.getElementById('mfa-setup-password').focus();
  });
  document.querySelectorAll('.mfa-cancel').forEach(b => b.addEventListener('click', loadMfaStatus));

  document.getElementById('mfa-step-password').addEventListener('submit', async (e) => {
    e.preventDefault();
    const err = document.getElementById('mfa-setup-error');
    err.textContent = '';
    try {
      const r = await sessionPost('/api/auth/mfa/setup', { password: document.getElementById('mfa-setup-password').value });
      document.getElementById('mfa-setup-password').value = '';
      document.getElementById('mfa-secret').textContent = r.secret.match(/.{1,4}/g).join(' ');
      const qr = document.getElementById('mfa-qr');
      // Only an SVG data URI is accepted, and only into an <img>, where an SVG cannot run scripts.
      if (typeof r.qr_data_uri === 'string' && r.qr_data_uri.startsWith('data:image/svg+xml')) { qr.src = r.qr_data_uri; qr.hidden = false; }
      else qr.hidden = true;
      showOnly('mfa-step-scan');
      document.getElementById('mfa-enable-code').focus();
    } catch (ex) { err.textContent = ex.message; }
  });

  document.getElementById('mfa-step-scan').addEventListener('submit', async (e) => {
    e.preventDefault();
    const err = document.getElementById('mfa-enable-error');
    err.textContent = '';
    try {
      const r = await sessionPost('/api/auth/mfa/enable', { code: document.getElementById('mfa-enable-code').value });
      session.mfaEnabled = true;
      delete document.body.dataset.mfaSetup;
      document.getElementById('mfa-secret').textContent = '';
      document.getElementById('mfa-pill').textContent = 'On';
      document.getElementById('mfa-pill').className = 'status-pill ok';
      document.getElementById('mfa-required-note').hidden = true;
      showRecoveryCodes(r.recovery_codes);
    } catch (ex) { err.textContent = ex.message; }
  });

  document.getElementById('mfa-manage').addEventListener('submit', async (e) => {
    e.preventDefault();
    const action = e.submitter ? e.submitter.dataset.mfaAction : 'codes';
    const err = document.getElementById('mfa-manage-error');
    err.textContent = '';
    const body = { password: document.getElementById('mfa-manage-password').value, code: document.getElementById('mfa-manage-code').value };
    try {
      if (action === 'disable') {
        await sessionPost('/api/auth/mfa/disable', body);
        session.mfaEnabled = false;
        await loadMfaStatus();
      } else {
        const r = await sessionPost('/api/auth/mfa/recovery-codes', body);
        document.getElementById('mfa-manage-password').value = '';
        document.getElementById('mfa-manage-code').value = '';
        showRecoveryCodes(r.recovery_codes);
      }
    } catch (ex) { err.textContent = ex.message; }
  });

  const codesText = () => `RobustStego recovery codes for ${session.username}\nEach code works once.\n\n${recoveryCodes.join('\n')}\n`;
  document.getElementById('mfa-codes-copy').addEventListener('click', async () => {
    const b = document.getElementById('mfa-codes-copy');
    try { await navigator.clipboard.writeText(codesText()); b.textContent = 'Copied'; } catch (_) { b.textContent = 'Select and copy by hand'; }
    setTimeout(() => { b.textContent = 'Copy'; }, 2000);
  });
  document.getElementById('mfa-codes-download').addEventListener('click', () => {
    const url = URL.createObjectURL(new Blob([codesText()], { type: 'text/plain' }));
    const a = document.createElement('a');
    a.href = url; a.download = 'robuststego-recovery-codes.txt';
    document.body.appendChild(a); a.click(); a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  });
  document.getElementById('mfa-codes-done').addEventListener('click', loadMfaStatus);
}

function initAccounts() {
  addRevealButtons();
  initPasswordRules();
  initMfaPanels();
  initSignup();
  document.getElementById('login-mfa-form').addEventListener('submit', handleMfaLogin);
  document.getElementById('login-mfa-back').addEventListener('click', () => showPasswordLoginStep());
  document.getElementById('password-form').addEventListener('submit', handlePasswordChange);
  document.getElementById('password-cancel').addEventListener('click', closePasswordDialog);
  document.getElementById('password-logout').addEventListener('click', handleLogout);
  document.getElementById('btn-change-password').addEventListener('click', () => openPasswordDialog(false));
  document.getElementById('password-overlay').addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && document.getElementById('password-overlay').dataset.forced !== 'true') closePasswordDialog();
  });
  document.getElementById('user-create-form').addEventListener('submit', handleCreateUser);
  document.getElementById('btn-hide-temp').addEventListener('click', hideTempPassword);
  document.getElementById('btn-copy-temp').addEventListener('click', async () => {
    const b = document.getElementById('btn-copy-temp');
    try { await navigator.clipboard.writeText(document.getElementById('temp-password-value').textContent); b.textContent = 'Copied'; }
    catch (_) { b.textContent = 'Select and copy it by hand'; }
    setTimeout(() => { b.textContent = 'Copy'; }, 2000);
  });
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
    if (res.status === 403) {
      const detail = await errorDetail(res, 'This account cannot sign in yet.');
      document.getElementById('login-resend').hidden = !/confirm your email/i.test(detail);
      throw new Error(detail);
    }
    if (!res.ok) throw new Error("Invalid credentials, or account temporarily locked.");
    document.getElementById('login-password').value = '';
    const data = await res.json();
    // Code step only when the server actually issued a code-step token (the account HAS two-step on).
    if (data.mfa_required && data.mfa_token) showMfaLoginStep(data.mfa_token);
    else setSession(data);
  } catch (err) {
    errEl.innerText = err instanceof TypeError ? describeFetchError(err) : err.message;
  } finally {
    btn.disabled = false;
  }
}

// ------------------------------------------------------------ sign-in, second step
// The mfa token only proves the password was right; it lives in this variable for at most 5 minutes.
let mfaLoginToken = null;

function showMfaLoginStep(token) {
  mfaLoginToken = token;
  showLoginCard('login-mfa-form');
  document.getElementById('login-mfa-error').textContent = '';
  document.getElementById('login-mfa-code').value = '';
  document.getElementById('login-mfa-code').focus();
}

function showLoginCard(id) {
  document.querySelectorAll('.login-card input').forEach(i => i.hidePassword && i.hidePassword());
  ['login-form', 'login-mfa-form', 'signup-form', 'signup-done'].forEach(x => { document.getElementById(x).hidden = x !== id; });
}

function showPasswordLoginStep(message = '') {
  mfaLoginToken = null;
  showLoginCard('login-form');
  document.getElementById('login-error').textContent = message;
}

async function handleMfaLogin(e) {
  e.preventDefault();
  const err = document.getElementById('login-mfa-error');
  const btn = document.getElementById('login-mfa-submit');
  err.textContent = '';
  btn.disabled = true;
  try {
    const res = await authPost('/api/auth/mfa', { mfa_token: mfaLoginToken, code: document.getElementById('login-mfa-code').value });
    if (res.status === 429) throw new Error('Too many attempts. Wait a minute and try again.');
    if (!res.ok) {
      const detail = await errorDetail(res, 'Wrong code.');
      if (/too long/i.test(detail)) { showPasswordLoginStep('Sign-in took too long. Enter your password again.'); return; }
      throw new Error(detail === 'Wrong code' ? 'Wrong code. Use the newest code from the app, or a recovery code.' : detail);
    }
    const data = await res.json();
    showPasswordLoginStep();
    setSession(data);
  } catch (ex) {
    err.textContent = ex instanceof TypeError ? describeFetchError(ex) : ex.message;
  } finally {
    btn.disabled = false;
  }
}

// ------------------------------------------------------------ self sign-up
async function handleSignup(e) {
  e.preventDefault();
  const err = document.getElementById('signup-error');
  const btn = document.getElementById('signup-submit');
  const pw = document.getElementById('signup-password').value;
  err.textContent = '';
  if (pw !== document.getElementById('signup-repeat').value) { err.textContent = 'The two passwords are different.'; return; }
  btn.disabled = true;
  try {
    const res = await authPost('/api/auth/register', {
      email: document.getElementById('signup-email').value.trim(),
      username: document.getElementById('signup-username').value.trim(), password: pw });
    if (res.status === 429) throw new Error('Too many sign-ups from this network. Try again in an hour.');
    if (!res.ok) throw new Error(await errorDetail(res, 'Could not send the request.'));
    ['signup-email', 'signup-username', 'signup-password', 'signup-repeat'].forEach(id => { document.getElementById(id).value = ''; });
  refreshPasswordRules();
    showLoginCard('signup-done');
    document.getElementById('signup-done-back').focus();
  } catch (ex) {
    err.textContent = ex instanceof TypeError ? describeFetchError(ex) : ex.message;
  } finally {
    btn.disabled = false;
  }
}

function showLoginNotice(text) {
  const n = document.getElementById('login-notice');
  n.textContent = text;
  n.hidden = !text;
}

// The confirmation email links to  <site>/#verify=<token>. The token is in the #fragment so it never reaches
// a server log; it is removed from the address bar straight away.
async function handleVerifyLink() {
  const m = /^#verify=([A-Za-z0-9_-]{20,200})$/.exec(location.hash);
  if (!m) return;
  history.replaceState(null, '', location.pathname + location.search);
  showPasswordLoginStep();
  try {
    const res = await authPost('/api/auth/verify-email', { token: m[1] });
    if (!res.ok) throw new Error(await errorDetail(res, 'This link is invalid or has expired.'));
    showLoginNotice((await res.json()).message);
  } catch (ex) {
    document.getElementById('login-error').textContent = ex instanceof TypeError ? describeFetchError(ex) : ex.message;
  }
}

// ------------------------------------------------------------ password fields: show/hide + live rules
const EYE = '<svg aria-hidden="true" focusable="false" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M2 12s3.5-7 10-7 10 7 10 7-3.5 7-10 7S2 12 2 12z"/><circle cx="12" cy="12" r="3"/></svg>';
const EYE_OFF = '<svg aria-hidden="true" focusable="false" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 3l18 18"/><path d="M10.6 5.1A10.8 10.8 0 0 1 12 5c6.5 0 10 7 10 7a17.6 17.6 0 0 1-3.2 4.2"/><path d="M6.6 6.6C3.8 8.4 2 12 2 12s3.5 7 10 7c1.8 0 3.4-.5 4.8-1.3"/><path d="M9.9 9.9a3 3 0 0 0 4.2 4.2"/></svg>';

// Wraps every password input with an eye button. The button only switches the input between type
// "password" and "text"; the value never leaves the field. Hidden again when the form is sent or reset.
function addRevealButtons() {
  document.querySelectorAll('input[type="password"]').forEach(input => {
    if (input.closest('.pw-field')) return;
    const wrap = document.createElement('div');
    wrap.className = 'pw-field';
    input.parentNode.insertBefore(wrap, input);
    wrap.appendChild(input);
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'pw-toggle';
    btn.setAttribute('aria-controls', input.id);
    wrap.appendChild(btn);
    const set = (show) => {
      input.type = show ? 'text' : 'password';
      btn.innerHTML = show ? EYE_OFF : EYE;           // fixed icon markup only, never user text
      btn.setAttribute('aria-pressed', String(show));
      btn.setAttribute('aria-label', show ? 'Hide password' : 'Show password');
      btn.title = show ? 'Hide password' : 'Show password';
    };
    set(false);
    btn.addEventListener('click', () => {
      const pos = input.selectionStart;
      set(input.type === 'password');
      input.focus();
      try { input.setSelectionRange(pos, pos); } catch (_) { /* keep the cursor where it was */ }
    });
    const form = input.form;
    if (form) {
      form.addEventListener('submit', () => set(false));
      form.addEventListener('reset', () => set(false));
    }
    input.hidePassword = () => set(false);
  });
}

// The SAME rules the server checks (security.password_problem), so a green list means the server will accept it.
const PW_RULES = [
  { key: 'len',   text: 'At least 12 characters',             test: (pw) => pw.length >= 12 },
  { key: 'case',  text: 'Upper and lower case letters',       test: (pw) => /[a-z]/.test(pw) && /[A-Z]/.test(pw) },
  { key: 'digit', text: 'A number',                           test: (pw) => /\d/.test(pw) },
  { key: 'sym',   text: 'A symbol, like ! ? # or -',         test: (pw) => /[^\p{L}\p{N}_\s]/u.test(pw) },
  { key: 'user',  text: 'Does not contain your username',     test: (pw, user) => !user || !pw.toLowerCase().includes(user.toLowerCase()) },
];
const MAX_PW_BYTES = 72;   // bcrypt ignores anything longer

function passwordProblems(pw, user) {
  const failed = PW_RULES.filter(r => !r.test(pw, user)).map(r => r.key);
  if (new TextEncoder().encode(pw).length > MAX_PW_BYTES) failed.push('long');
  return failed;
}

function initPasswordRules() {
  document.querySelectorAll('.pw-rules[data-rules-for]').forEach(box => {
    const input = document.getElementById(box.dataset.rulesFor);
    const userEl = document.getElementById(box.dataset.usernameFrom);
    const repeat = document.getElementById(box.dataset.repeat);
    const matchEl = repeat ? document.getElementById(repeat.getAttribute('aria-describedby')) : null;
    const list = document.createElement('ul');
    list.className = 'pw-rule-list';
    const items = {};
    PW_RULES.forEach(r => {
      const li = document.createElement('li');
      li.dataset.state = 'todo';
      const mark = document.createElement('span'); mark.className = 'pw-mark'; mark.setAttribute('aria-hidden', 'true');
      const label = document.createElement('span'); label.textContent = r.text;
      const sr = document.createElement('span'); sr.className = 'sr-only';
      li.append(mark, label, sr);
      items[r.key] = { li, sr };
      list.appendChild(li);
    });
    const tooLong = document.createElement('p');
    tooLong.className = 'pw-too-long'; tooLong.hidden = true;
    tooLong.textContent = 'Too long: at most 72 bytes (letters with accents or emoji count as more than one).';
    const meter = document.createElement('div');
    meter.className = 'pw-meter'; meter.setAttribute('aria-hidden', 'true');
    meter.appendChild(document.createElement('span'));
    box.replaceChildren(meter, list, tooLong);

    const update = () => {
      const pw = input.value;
      const user = userEl ? userEl.value.trim() : '';
      const failed = passwordProblems(pw, user);
      PW_RULES.forEach(r => {
        const ok = !failed.includes(r.key);
        const state = pw.length === 0 ? 'todo' : (ok ? 'ok' : 'bad');
        items[r.key].li.dataset.state = state;
        items[r.key].sr.textContent = state === 'ok' ? ' (done)' : ' (not yet)';
      });
      tooLong.hidden = !failed.includes('long');
      const done = PW_RULES.length - failed.filter(k => k !== 'long').length;
      meter.dataset.level = pw.length === 0 ? '0' : String(done);
      meter.firstChild.style.width = `${pw.length === 0 ? 0 : (done / PW_RULES.length) * 100}%`;
      input.setCustomValidity(failed.length ? 'This password does not meet the rules listed below it.' : '');
      if (repeat && matchEl) {
        const r = repeat.value;
        const same = r.length > 0 && r === pw;
        matchEl.dataset.state = r.length === 0 ? '' : (same ? 'ok' : 'bad');
        matchEl.textContent = r.length === 0 ? '' : (same ? 'Passwords match' : 'Passwords do not match yet');
        repeat.setCustomValidity(r.length && !same ? 'The two passwords are different.' : '');
      }
    };
    input.addEventListener('input', update);
    if (userEl) userEl.addEventListener('input', update);
    if (repeat) repeat.addEventListener('input', update);
    const form = input.form;
    if (form) form.addEventListener('reset', () => setTimeout(update));
    box.refresh = update;
    update();
  });
}

function refreshPasswordRules() {
  document.querySelectorAll('.pw-rules').forEach(b => b.refresh && b.refresh());
}

function initSignup() {
  document.getElementById('signup-form').addEventListener('submit', handleSignup);
  document.getElementById('show-signup').addEventListener('click', () => {
    showLoginCard('signup-form');
    document.getElementById('signup-error').textContent = '';
    document.getElementById('signup-email').focus();
  });
  document.getElementById('show-login').addEventListener('click', () => { showPasswordLoginStep(); document.getElementById('login-username').focus(); });
  document.getElementById('signup-done-back').addEventListener('click', () => { showPasswordLoginStep(); document.getElementById('login-username').focus(); });
  document.getElementById('login-resend').addEventListener('click', async () => {
    const btn = document.getElementById('login-resend');
    const err = document.getElementById('login-error');
    btn.disabled = true;
    try {
      const res = await authPost('/api/auth/resend-verification', {
        username: document.getElementById('login-username').value, password: document.getElementById('login-password').value });
      if (res.status === 429) throw new Error('Too many requests. Try again in an hour.');
      if (!res.ok) throw new Error(await errorDetail(res, 'Could not send a new link.'));
      showLoginNotice((await res.json()).message);
      err.textContent = ''; btn.hidden = true;
    } catch (ex) { err.textContent = ex instanceof TypeError ? describeFetchError(ex) : ex.message; }
    finally { btn.disabled = false; }
  });
  handleVerifyLink();
  window.addEventListener('hashchange', handleVerifyLink);   // link opened in a tab that already has the app
  // Hide the link when the server has sign-up turned off.
  fetch(`${API_BASE}/api/auth/signup-open`).then(r => r.ok ? r.json() : { open: true })
    .then(d => { document.getElementById('signup-link-row').hidden = !d.open; }).catch(() => {});
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
  coverScene: 'astronaut', // a SAMPLES key, or 'Uploaded File'
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
  ecc: null,                 // {ok, corrected} from the server's Reed-Solomon decode
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

// Built-in sample covers (3 Nov 2026): real photos instead of synthetic drawings. The drawings were flat,
// so every one triggered the flat-area tip and made the CNN's grid visible in demos. Photos from the
// scikit-image sample set: astronaut (public domain, NASA), coffee (CC0, Rachel Michetti), cat "Chelsea"
// (CC0, Stefan van der Walt), rocket (public domain, SpaceX). Centre-cropped and resized to 512x512.
function loadSamplePhoto(canvas, key) {
  return new Promise((resolve, reject) => {
    const sample = SAMPLES[key];
    if (!sample) return reject(new Error(`Unknown sample: ${key}`));
    const img = new Image();
    img.onload = () => {
      const ctx = canvas.getContext('2d');
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
      resolve();
    };
    img.onerror = () => reject(new Error(`Could not load ${sample.file}`));
    img.src = sample.file;
  });
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

  const bytes = utf8Bytes(text);
  elements.charCounter.innerText = `${bytes} / ${MAX_MESSAGE_BYTES} bytes`;
  elements.charCounter.classList.toggle('over-limit', bytes > MAX_MESSAGE_BYTES);
}

function applyTheme(theme) {
  document.body.classList.toggle('dark-theme', theme === 'dark');
  document.body.classList.toggle('light-theme', theme === 'light');
  elements.toggleTheme.checked = theme === 'dark';
  try { localStorage.setItem('theme', theme); } catch (_) { /* theme preference only */ }
}

// SPA Routing
const PAGE_ORDER = ['page-embedding', 'page-result', 'page-downscaling', 'page-extraction'];

// Which flow a page belongs to: the Start page, the 4-step Hide flow, or the 1-step Read flow.
const pageMode = (id) => (id === 'page-home' ? 'home' : id === 'page-read' ? 'read'
  : id === 'page-results' ? 'results' : id === 'page-users' ? 'users' : id === 'page-account' ? 'account' : 'hide');

function handleNavigation(targetId) {
  if (document.body.dataset.mfaSetup && targetId !== 'page-account') targetId = 'page-account';
  if (targetId === 'page-home') requestAnimationFrame(moveTabIndicator);
  if (targetId === 'page-users') {
    if (session.role !== 'admin') targetId = 'page-home';
    else loadUsers();
  }
  if (targetId === 'page-account') loadMfaStatus();
  document.body.dataset.mode = pageMode(targetId);
  const mode = pageMode(targetId);
  // Page heading: hide steps take the step label from the stepper; the other pages have fixed names.
  const stepBtn = document.querySelector(`.nav-btn[data-target="${targetId}"] .step-label`);
  document.getElementById('topbar-heading').textContent =
    mode === 'home' ? 'Hide or read a message'
      : mode === 'read' ? 'Read a hidden message'
      : mode === 'results' ? 'Measured results'
      : mode === 'users' ? 'Accounts'
      : mode === 'account' ? 'Your account'
      : (stepBtn ? stepBtn.textContent : '');
  document.getElementById('topbar-step').textContent =
    mode === 'home' ? 'Start'
      : mode === 'read' ? 'Read a message'
      : mode === 'results' ? 'Results'
      : mode === 'users' ? 'Users'
      : mode === 'account' ? 'Account'
      : `Hide a message \u00b7 Step ${PAGE_ORDER.indexOf(targetId) + 1} of ${PAGE_ORDER.length}`;
  document.querySelectorAll('.site-link').forEach(l => {
    if (l.dataset.modeLink === mode) l.setAttribute('aria-current', 'page'); else l.removeAttribute('aria-current');
  });
  elements.navButtons.forEach(btn => {
    if (btn.getAttribute('data-target') === targetId) {
      btn.classList.add('active');
      btn.setAttribute('aria-current', 'step');
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
  // Move focus to the new page heading so keyboard and screen-reader users land on the new content.
  const heading = document.getElementById('topbar-heading');
  if (heading) { heading.setAttribute('tabindex', '-1'); heading.focus({ preventScroll: true }); }
  window.scrollTo(0, 0);
}

// Tick completed workflow steps in the sidebar. Purely visual: derived from existing state,
// never blocks navigation (uploads let you start at any step).
function updateStepProgress() {
  const done = {
    'page-embedding': state.embedded || state.stegoUploaded,
    'page-result': state.embedded,
    'page-downscaling': state.attackSimulated,
    'page-extraction': state.extracted,
    'page-read': readState.result !== null,
  };
  elements.navButtons.forEach(btn => {
    btn.classList.toggle('done', Boolean(done[btn.getAttribute('data-target')]));
  });
  document.body.classList.toggle('has-embed', Boolean(state.embedded));
  const chip = document.getElementById('topbar-method');
  if (chip) chip.textContent = methodInfo(state.method).short;
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

// Turns a non-OK response into an Error with the server's own message when it sent one.
async function responseError(res) {
  let detail = '';
  try { const j = await res.json(); detail = typeof j.detail === 'string' ? j.detail : ''; } catch (_) { /* not JSON */ }
  return new Error(detail || `Server returned HTTP ${res.status}`);
}

// Friendlier error if the server simply isn't running / unreachable.
function describeFetchError(err) {
  if (err instanceof TypeError) {
    return `Can't reach the inference server at ${API_BASE}. Is server.py running (uvicorn server:app --port 8000)?`;
  }
  return err.message;
}

// ----------------------------------------------------------- real pipeline steps

// Reads + validates an uploaded PNG/JPEG/WebP (WebP = WhatsApp sticker). Returns the ORIGINAL bytes as a data URL (not re-encoded via
// canvas, which would silently alter pixels - important when testing Telegram/JPEG round-trips).
function readImageFile(file) {
  return new Promise((resolve, reject) => {
    if (!file) return reject(new Error("No file selected."));
    if (!["image/png", "image/jpeg", "image/webp"].includes(file.type)) return reject(new Error("Only PNG, JPEG or WebP images are accepted."));
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

// Inline error messages (UX audit, 3 Nov 2026): errors appear next to the field or button they concern,
// are announced to screen readers (role="alert"), and never block the page like alert() did.
// textContent only: messages can include server text.
function showInlineError(anchor, msg, where = 'after') {
  if (!anchor) return;
  const sib = where === 'after' ? anchor.nextElementSibling : anchor.previousElementSibling;
  let el = sib && sib.classList.contains('inline-error') ? sib : null;
  if (!el) {
    el = document.createElement('p');
    el.className = 'inline-error';
    el.setAttribute('role', 'alert');
    if (where === 'after') anchor.after(el); else anchor.before(el);
  }
  el.textContent = msg;
  el.hidden = false;
}
function clearInlineError(anchor) {
  if (!anchor) return;
  for (const sib of [anchor.nextElementSibling, anchor.previousElementSibling]) {
    if (sib && sib.classList.contains('inline-error')) sib.hidden = true;
  }
}
const embedFooter = () => document.querySelector('#page-embedding .card-footer-actions');
const stegoSourceBox = () => document.querySelector('#page-downscaling .stego-input-status');
const attackedSourceBox = () => document.querySelector('#page-extraction .stego-input-status');
const messageBox = () => document.querySelector('.message-input-wrapper');

// Page 1 -> 2: calls POST /api/embed on the real trained CNN (or real LSB).
async function runStegoPipeline() {
  clearInlineError(messageBox());
  clearInlineError(embedFooter());
  if (state.message.trim() === '') {
    showInlineError(messageBox(), "Type a secret message before embedding.");
    elements.secretMessageInput.focus();
    return;
  }
  if (utf8Bytes(state.message) > MAX_MESSAGE_BYTES) {
    showInlineError(messageBox(), `The message is ${utf8Bytes(state.message)} bytes; the limit is ${MAX_MESSAGE_BYTES} bytes (emoji and accented letters use more than 1 byte). Shorten it and try again.`);
    elements.secretMessageInput.focus();
    return;
  }

  writeLog("[SYSTEM] Initiating embedding request to inference server...", "system");
  elements.btnEmbedPayload.disabled = true;
  setBtnLabel(elements.btnEmbedPayload, "Embedding…");

  writeLog("[INFO] Preprocessing cover pixel matrix (512x512x3)...", "info");
  writeLog(`[INFO] Message byte representation converted to bitstream (${utf8Bytes(state.message)} bytes + Reed-Solomon parity = 1024 bits, whitened).`, "info");

  try {
    const coverDataUrl = elements.coverCanvas.toDataURL('image/png');

    writeLog(`[INFO] ${methodInfo(state.method).log}`, "info");

    const res = await apiFetch("/api/embed", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        cover_image: coverDataUrl,
        message: state.message,
        method: state.method,
      }),
    });
    if (!res.ok) throw await responseError(res);
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

    elements.methodTitleBadge.innerText = methodInfo(state.method).long;
    elements.methodDetailsText.innerText = methodInfo(state.method).details;
    elements.cnnNetworkPreview.style.display = state.method === 'cnn' ? 'flex' : 'none';

    elements.metricPsnr.innerText = `${state.psnr.toFixed(2)} dB`;
    elements.metricSsim.innerText = state.ssim.toFixed(4);
    elements.psnrBar.style.width = `${Math.min(100, (state.psnr / 60) * 100)}%`;
    elements.ssimBar.style.width = `${state.ssim * 100}%`;

    elements.stegoSourceStatus.innerText = `Stego loaded. Method: ${methodInfo(state.method).long}.`;
    elements.btnSimulateAttack.disabled = false;

    writeLog(`[SUCCESS] Steganographic embedding complete. PSNR: ${state.psnr.toFixed(2)}dB, SSIM: ${state.ssim.toFixed(4)}`, "success");

    setTimeout(() => handleNavigation('page-result'), 400);
  } catch (err) {
    const msg = describeFetchError(err);
    writeLog(`[ERROR] Embedding failed: ${msg}`, "error");
    showInlineError(embedFooter(), `Embedding failed: ${msg}`, 'before');
  } finally {
    elements.btnEmbedPayload.disabled = false;
    setBtnLabel(elements.btnEmbedPayload, "Embed message");
  }
}

// Page 3: calls POST /api/downscale, which runs the exact OpenCV algorithm
// (nearest/bilinear/bicubic/lanczos/area) used in eval.py.
async function runDownscalingAttack() {
  clearInlineError(elements.btnSimulateAttack);
  if (!state.stegoImgData) {
    showInlineError(elements.btnSimulateAttack, "No stego image yet. Embed a message in step 1, or upload a stego image above.");
    return;
  }

  writeLog("[ATTACK] Sending stego image to server for real downscale attack...", "system");
  elements.btnSimulateAttack.disabled = true;
  setBtnLabel(elements.btnSimulateAttack, "Resizing…");
  elements.arrowScaleLabel.innerText = smallerLabel(state.attackScale);

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
    if (!res.ok) throw await responseError(res);
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
    state.lastAttack = { algo: state.attackAlgo, size: data.size };
    document.getElementById('btn-download-resized').disabled = false;

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
    showInlineError(elements.btnSimulateAttack, `Downscale attack failed: ${msg}`);
  } finally {
    elements.btnSimulateAttack.disabled = false;
    setBtnLabel(elements.btnSimulateAttack, "Run downscaling attack");
  }
}

// Page 4: calls POST /api/extract, running the real decoder (or real LSB
// extraction) on the actually-downscaled image, and gets a real BER back
// by comparing against the original message bits.
async function runExtractionAndEvaluation() {
  const extractBar = elements.btnExtractMessage.closest('.form-actions-bar') || elements.btnExtractMessage;
  clearInlineError(extractBar);
  if (!state.downscaledImgData) {
    showInlineError(extractBar, "No image to decode yet. Run the attack in step 3, or upload an image above.");
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
    if (!res.ok) throw await responseError(res);
    const data = await res.json(); // { recovered_message, ber, extracted_psnr, extracted_ssim }

    state.extracted = true;
    updateStepProgress();
    const berKnown = data.ber !== null && data.ber !== undefined;
    state.ber = berKnown ? data.ber * 100 : null;

    state.recoveredExact = expected ? data.recovered_message === expected : null;
    // Older servers don't report error correction: show nothing rather than a false "failed".
    state.ecc = typeof data.ecc_ok === 'boolean'
      ? { ok: data.ecc_ok, corrected: Math.max(0, Math.trunc(Number(data.ecc_corrected_bytes)) || 0) }
      : null;
    updateEccStatus();
    renderMessageDiff(expected, data.recovered_message);
    updateBerHero(berKnown);
    elements.extractedMsgLabel.innerText =
      `Recovered message (${methodInfo(state.method).short} decoder)`;

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
    document.getElementById('extract-quality-grid').hidden = data.extracted_psnr == null && data.extracted_ssim == null;

    updateBenchmarkTable();
    renderBenchmarkChart();
    setBenchTab('run');
    if (berKnown) {
      updateAcademicSummaryParagraph();
    } else {
      // Blind extraction: no BER without the original message, but say what the decoder found.
      const dec = `the ${methodInfo(state.method).long} decoder`;
      elements.evaluationSummaryText.textContent = !state.ecc
        ? `Extracted with ${dec}. Enter the expected message to measure the bit error rate.`
        : state.ecc.ok
          ? `A hidden message was recovered with ${dec}` +
            (state.ecc.corrected > 0 ? `; Reed-Solomon error correction repaired ${state.ecc.corrected} of 128 bytes. ` : '. ') +
            'Enter the expected message to measure the bit error rate.'
          : `No valid message was recovered with ${dec}: more than 32 of 128 bytes were damaged, or the image ` +
            'was made with a different method. Try another decoder, or use Read a hidden message, which tries all four.';
    }

    writeLog(`[SUCCESS] Extraction complete via server. BER: ${berKnown ? state.ber.toFixed(2) + '%' : 'n/a'}`, "success");
  } catch (err) {
    const msg = describeFetchError(err);
    writeLog(`[ERROR] Extraction failed: ${msg}`, "error");
    showInlineError(extractBar, `Extraction failed: ${msg}`);
  } finally {
    elements.btnExtractMessage.disabled = false;
    setBtnLabel(elements.btnExtractMessage, "Extract hidden message");
  }
}

// Benchmark card tabs (design round 1): "This run" = comparison table, "By size" = chart + size table.
function setBenchTab(name) {
  document.querySelectorAll('.bench-tab').forEach(b => {
    const on = b.dataset.benchTab === name;
    b.classList.toggle('active', on);
    b.setAttribute('aria-pressed', String(on));
  });
  document.querySelectorAll('.bench-panel').forEach(p => { p.hidden = p.dataset.benchPanel !== name; });
}

// ------------------------------------------------------------------ flat-area tip (3 Nov 2026)
// The CNN changes every 16x16 block (32x32 message grid), so on large flat areas its colour pattern can be
// seen; DCT's fine 8x8 brightness texture is much less visible there (Results log 9c-2). Heuristic, calibrated
// on 16 images: share of 16x16 blocks whose brightness std is below 2.5 (0-255). Cartoon stickers scored
// 33-52%, ordinary photos 1-18% (sky/plain-background photos such as rocket 71%, moon 52% also trigger).
const FLAT_BLOCK_STD = 2.5;
const FLAT_SHARE_TIP = 0.30;

function flatBlockShare(canvas) {
  const { width: w, height: h } = canvas;
  const data = canvas.getContext('2d').getImageData(0, 0, w, h).data;
  const B = 16;
  let flat = 0, total = 0;
  for (let by = 0; by + B <= h; by += B) {
    for (let bx = 0; bx + B <= w; bx += B) {
      let s = 0, s2 = 0;
      for (let y = by; y < by + B; y++) {
        for (let x = bx; x < bx + B; x++) {
          const i = (y * w + x) * 4;
          const lum = 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
          s += lum; s2 += lum * lum;
        }
      }
      const n = B * B, mean = s / n;
      if (Math.sqrt(Math.max(0, s2 / n - mean * mean)) < FLAT_BLOCK_STD) flat++;
      total++;
    }
  }
  return total ? flat / total : 0;
}

function updateFlatTip() {
  const tip = document.getElementById('flat-tip');
  if (!tip) return;
  let share = 0;
  try { share = flatBlockShare(elements.coverCanvas); } catch (_) { tip.hidden = true; return; }
  if (share >= FLAT_SHARE_TIP && state.method === 'cnn') {
    tip.textContent = `Tip: about ${Math.round(share * 100)}% of this image is flat (plain background or large single-colour areas). ` +
      'The CNN changes every part of the image, so its faint colour pattern may be visible here. ' +
      'DCT usually hides better on images like this (for example cartoon stickers).';
    tip.hidden = false;
  } else {
    tip.hidden = true;
  }
}

// ------------------------------------------------------------------ Read flow (3 Nov 2026)
// A receiver does not know the original message or which method the sender used, so this flow tries
// the decoders and shows the MESSAGE first. Reed-Solomon makes this safe: a decoder either repairs the
// payload completely (ecc_ok) or reports failure, and in testing (tools/cross_matrix.py) no wrong decoder
// ever produced a valid message. BER is optional, under Technical details, for when the original is known.
const DECODER_ORDER = ['cnn', 'dct', 'dwt', 'lsb'];
const readState = { imgData: null, result: null, method: null };

async function extractOnce(image, method, original = '') {
  const res = await apiFetch("/api/extract", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ image, method, original_message: original, cover_image: "" }),
  });
  if (!res.ok) throw await responseError(res);
  return res.json();
}

function setReadHero(stateName, label, message, note) {
  const hero = document.getElementById('read-hero');
  hero.dataset.state = stateName;
  document.getElementById('read-hero-label').textContent = label;
  document.getElementById('read-message').textContent = message;
  document.getElementById('read-hero-note').textContent = note;
  document.getElementById('btn-read-copy').hidden = stateName !== 'pass';
}

function renderReadAttempts(results) {
  const body = document.getElementById('read-attempts');
  body.replaceChildren(...results.map(r => {
    const tr = document.createElement('tr');
    const cells = [
      methodInfo(r.method).long,
      r.ecc_ok ? 'Message found' : 'No valid message',
      r.ecc_ok ? `repaired ${r.ecc_corrected_bytes} of 128 bytes` : 'failed (more than 32 of 128 bytes damaged)',
    ];
    cells.forEach((text, i) => {
      const td = document.createElement('td');
      td.textContent = text;
      if (i === 1) td.className = r.ecc_ok ? 'better-val' : 'worse-val';
      tr.appendChild(td);
    });
    return tr;
  }));
}

async function runReadExtraction() {
  if (!readState.imgData) return;
  const choice = document.getElementById('read-method').value;
  const methods = choice === 'auto' ? DECODER_ORDER : [choice];
  const btn = document.getElementById('btn-read-extract');
  btn.disabled = true;
  setBtnLabel(btn, choice === 'auto' ? 'Trying all decoders…' : 'Decoding…');
  document.getElementById('read-ber-result').textContent = '';
  try {
    const results = await Promise.all(methods.map(async m => ({ method: m, ...(await extractOnce(readState.imgData, m)) })));
    renderReadAttempts(results);
    const ok = results.filter(r => r.ecc_ok).sort((a, b) => a.ecc_corrected_bytes - b.ecc_corrected_bytes);
    const texts = [...new Set(ok.map(r => r.recovered_message))];
    if (ok.length === 0) {
      readState.result = { found: false };
      readState.method = choice === 'auto' ? null : choice;
      setReadHero('fail', 'Result', 'No readable hidden message',
        choice === 'auto'
          ? 'None of the four decoders could read a message. Either this image has no hidden message, ' +
            'or it was changed too much (more than 32 of 128 bytes damaged) to repair.'
          : `The ${methodInfo(choice).short} decoder could not read a message. ` +
            'Try Automatic under Technical details: the sender may have used a different method.');
      writeLog(`[INFO] Read: no valid message (${methods.join(', ')} tried).`, 'info');
    } else {
      const best = ok[0];
      readState.result = { found: true, text: best.recovered_message };
      readState.method = best.method;
      const names = ok.map(r => methodInfo(r.method).short).join(' and ');
      const repaired = best.ecc_corrected_bytes === 0
        ? 'no errors to repair'
        : `error correction repaired ${best.ecc_corrected_bytes} of 128 bytes`;
      const warn = texts.length > 1
        ? ' Warning: decoders returned different messages; showing the one that needed the least repair.' : '';
      setReadHero('pass', 'Hidden message found', best.recovered_message,
        `Read with the ${names} decoder${ok.length > 1 ? 's' : ''} \u00b7 ${repaired}.${warn}`);
      writeLog(`[SUCCESS] Read: message found with ${names} (${best.ecc_corrected_bytes} bytes repaired).`, 'success');
    }
    document.getElementById('btn-read-check').disabled = !readState.method;
    updateStepProgress();
  } catch (err) {
    const msg = describeFetchError(err);
    setReadHero('fail', 'Could not decode', msg, '');
    writeLog(`[ERROR] Read failed: ${msg}`, 'error');
  } finally {
    btn.disabled = false;
    setBtnLabel(btn, 'Read hidden message');
  }
}

// Optional: the user knows the original message and wants the raw bit error rate.
async function runReadBerCheck() {
  const out = document.getElementById('read-ber-result');
  const original = document.getElementById('read-expected').value.replace(/\r\n/g, '\n');
  if (!readState.method) { out.textContent = 'Read the image first, or choose a decoder above.'; return; }
  if (!original) { out.textContent = 'Type the original message first.'; return; }
  if (utf8Bytes(original) > MAX_MESSAGE_BYTES) { out.textContent = `The original message is longer than ${MAX_MESSAGE_BYTES} bytes.`; return; }
  try {
    const r = await extractOnce(readState.imgData, readState.method, original);
    const wrong = Math.round(r.ber * 1024);
    out.textContent = `Bit error rate ${(r.ber * 100).toFixed(2)}% (${wrong} of 1024 bits wrong) with the ` +
      `${methodInfo(readState.method).short} decoder. ` +
      (r.ecc_ok && r.recovered_message === original
        ? 'The recovered message matches the original exactly.'
        : 'The recovered message does not match the original.');
  } catch (err) {
    out.textContent = describeFetchError(err);
  }
}

function initReadFlow() {
  const input = document.getElementById('read-file-input');
  document.getElementById('btn-read-upload').addEventListener('click', () => input.click());
  input.addEventListener('change', async (e) => {
    const file = e.target.files[0];
    e.target.value = '';
    try {
      const { dataUrl, w, h } = await readImageFile(file);
      readState.imgData = dataUrl;
      readState.result = null;
      readState.method = null;
      const img = new Image();
      img.onload = () => drawMini(document.getElementById('read-mini-canvas'), img);
      img.src = dataUrl;
      const preview = document.getElementById('read-preview');
      preview.src = dataUrl; preview.hidden = false;
      document.getElementById('read-drop').dataset.hasImage = 'true';
      document.getElementById('read-import-status').textContent = `Image loaded, ${w} \u00d7 ${h} px.`;
      document.getElementById('btn-read-extract').disabled = false;
      document.getElementById('btn-read-check').disabled = true;
      document.getElementById('read-ber-result').textContent = '';
      setReadHero('idle', 'Hidden message', 'Press Read hidden message.', '');
      updateStepProgress();
      writeLog(`[INFO] Read: image uploaded (${w}x${h}px).`, 'info');
      clearInlineError(document.querySelector('#page-read .stego-input-status'));
    } catch (err) { showInlineError(document.querySelector('#page-read .stego-input-status'), `${err.message} Choose another file.`); }
  });
  // Drag and drop onto the large image area: handed to the same file input, so the same checks apply.
  const drop = document.getElementById('read-drop');
  ['dragenter', 'dragover'].forEach(ev => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add('dragging'); }));
  ['dragleave', 'drop'].forEach(ev => drop.addEventListener(ev, () => drop.classList.remove('dragging')));
  drop.addEventListener('drop', (e) => {
    e.preventDefault();
    const f = e.dataTransfer.files[0];
    if (!f) return;
    const dt = new DataTransfer(); dt.items.add(f);
    input.files = dt.files;
    input.dispatchEvent(new Event('change'));
  });
  document.getElementById('btn-read-extract').addEventListener('click', runReadExtraction);
  document.getElementById('btn-read-check').addEventListener('click', runReadBerCheck);
  document.getElementById('btn-read-copy').addEventListener('click', async () => {
    const btn = document.getElementById('btn-read-copy');
    try {
      await navigator.clipboard.writeText(readState.result && readState.result.text ? readState.result.text : '');
      btn.textContent = 'Copied';
    } catch (_) {
      // Clipboard blocked: select the text so the user can copy it by hand.
      const range = document.createRange();
      range.selectNodeContents(document.getElementById('read-message'));
      const sel = window.getSelection(); sel.removeAllRanges(); sel.addRange(range);
      btn.textContent = 'Selected - press Ctrl+C';
    }
    setTimeout(() => { btn.textContent = 'Copy message'; }, 2000);
  });
}

// Comparison table. Before an extraction it shows the two benchmark rows; afterwards the first row
// is this run (server numbers) and the second is the OTHER method's benchmark average (REFERENCE).
function updateBenchmarkTable() {
  const fmtBer = (v) => `${v.toFixed(2)}%`;
  const verdict = (v) => (v <= THRESHOLDS.ber ? 'better-val' : 'worse-val');
  const name = (m) => methodInfo(m).short;
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
    : `${algoName}, ${smallerLabel(state.attackScale)}`;
  const benchRows = METHODS[state.method].bench
    ? benchRow(state.method) + benchRow(state.method === 'cnn' ? 'lsb' : 'cnn')
    : benchRow('cnn') + benchRow('lsb') +
      `<tr><td colspan="4" class="cell-note">${name(state.method)} is not yet benchmarked on DIV2K; run tools/benchmark_methods.py.</td></tr>`;

  elements.evaluationTableRows.innerHTML = `<tr class="run-row">
      <td><span class="method-cell ${state.method}-cell">${name(state.method)}</span><span class="cell-sub">this run &middot; ${attack}</span></td>
      <td class="${verdict(state.ber)}">${fmtBer(state.ber)}</td>
      <td>${embeddedHere ? `${state.psnr.toFixed(2)} dB` : '--'}</td>
      <td>${embeddedHere ? state.ssim.toFixed(4) : '--'}</td>
    </tr>` + benchRows;
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

// Reed-Solomon outcome, shown under the BER verdict. Independent of the expected message.
function updateEccStatus() {
  const el = document.getElementById('ecc-status');
  if (!state.ecc) { el.hidden = true; return; }
  el.hidden = false;
  el.dataset.state = state.ecc.ok ? 'ok' : 'failed';
  el.textContent = !state.ecc.ok
    ? 'Error correction failed: more than 32 of 128 bytes damaged'
    : state.ecc.corrected === 0
      ? 'Error correction: no damaged bytes'
      : `Error correction repaired ${state.ecc.corrected} of 128 bytes`;
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
  const fixedNote = state.ecc && state.ecc.ok && state.ecc.corrected > 0
    ? ` Error correction repaired ${state.ecc.corrected} damaged bytes.` : '';
  const failNote = state.ecc && !state.ecc.ok
    ? ' Too many bytes were damaged for error correction, so this is an uncorrected best-effort read.' : '';
  summary.textContent = wrong === 0
    ? `Exact match: all ${exp.length} characters recovered.${fixedNote}`
    : `${correct} of ${exp.length} characters match.${failNote} Wrong characters are highlighted; dots mark characters that were not recovered.`;
}

// Benchmark chart (SVG built with DOM APIs; colours come from CSS classes so both themes work).
// Draws the four methods' benchmark BER per size. Used on Page 4 (with this run's marker) and on the
// Results page (withRun = false).
function renderBenchmarkChart(hostId = 'svg-chart-wrapper', withRun = true) {
  const host = document.getElementById(hostId);
  if (!host) return;
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
    'aria-label': 'Benchmark BER per output size. ' + METHOD_ORDER.map(m =>
      `${m.toUpperCase()} ${Math.min(...BENCHMARK_BY_SIZE[m])} to ${Math.max(...BENCHMARK_BY_SIZE[m])} percent`).join(', ') + '.' });

  [0, 10, 20, 30, 40, 50].forEach(v => {
    svg.appendChild(el('line', { x1: L, x2: W - R, y1: y(v), y2: y(v), class: v === 0 ? 'chart-axis' : 'chart-grid' }));
    svg.appendChild(el('text', { x: L - 8, y: y(v) + 4, 'text-anchor': 'end', class: 'chart-tick' }, `${v}%`));
  });
  // target line
  svg.appendChild(el('line', { x1: L, x2: W - R, y1: y(THRESHOLDS.ber), y2: y(THRESHOLDS.ber), class: 'chart-target' }));

  BENCHMARK_BY_SIZE.sizes.forEach((px, i) => {
    svg.appendChild(el('text', { x: xs[i], y: H - B + 18, 'text-anchor': 'middle', class: 'chart-tick' }, `${px} px`));
    svg.appendChild(el('text', { x: xs[i], y: H - B + 32, 'text-anchor': 'middle', class: 'chart-tick chart-tick-sub' },
      smallerLabel(BENCHMARK_BY_SIZE.scales[i])));
  });

  for (const m of ['lsb', 'dwt', 'dct', 'cnn']) {   // CNN drawn last, on top
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
  const showRun = withRun && state.extracted && state.ber !== null && !state.downscaledUploaded && idx >= 0;
  if (withRun) document.getElementById('run-legend').hidden = !showRun;
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

// Results page (design round 2): every number comes from REFERENCE / BENCHMARK_BY_SIZE / STICKER_RESULTS /
// REAL_SEND above, which cite the Results log. Built with DOM APIs and textContent only.
function renderResultsPage() {
  const cell = (text, cls) => { const td = document.createElement('td'); td.textContent = text; if (cls) td.className = cls; return td; };
  const methodCell = (m) => {
    const td = document.createElement('td');
    const span = document.createElement('span');
    span.className = `method-cell ${m}-cell`;
    span.textContent = methodInfo(m).short;
    td.appendChild(span);
    return td;
  };
  const ok = (v) => (v ? 'better-val' : 'worse-val');

  document.getElementById('results-bench-rows').replaceChildren(...METHOD_ORDER.map(m => {
    const r = REFERENCE[m];
    const tr = document.createElement('tr');
    tr.append(methodCell(m), cell(`${r.psnr.toFixed(2)} dB`), cell(r.ssim.toFixed(4)),
      cell(`${r.ber.toFixed(2)}%`, ok(r.ber <= THRESHOLDS.ber)), cell(`${r.worst.toFixed(2)}%`),
      cell(`${r.exact.toFixed(1)}%`));
    return tr;
  }));
  document.getElementById('results-bench-note').textContent =
    `${BENCH_ATTACKS.toLocaleString('en')} attacks per method: 100 DIV2K photos \u00d7 5 resize algorithms \u00d7 4 sizes ` +
    '(384, 299, 256, 224 px). Default strengths (DCT T=50, DWT S=128). At equal image quality (about 35 dB), a stronger ' +
    'DCT reaches about the same exact-message rate as the CNN (90.9% vs 88.9%); the CNN keeps the lower worst case and higher SSIM.';

  document.getElementById('results-sticker-rows').replaceChildren(...METHOD_ORDER.map(m => {
    const r = STICKER_RESULTS[m];
    const tr = document.createElement('tr');
    tr.append(methodCell(m), cell(`${r.survived} of ${STICKER_RESULTS.photos}`, ok(r.survived >= STICKER_RESULTS.photos * 0.9)),
      cell(`${r.ber.toFixed(2)}%`), cell(REAL_SEND[m]));
    return tr;
  }));
  document.getElementById('results-sticker-note').textContent =
    `Prediction: ${STICKER_RESULTS.photos} Kodak test photos saved as WebP at quality 90, the setting measured from real ` +
    'WhatsApp Web stickers (all three real stickers matched it exactly). Real send: one sticker per method through ' +
    'WhatsApp Web; it shows the route works, not how often.';

  const advice = [
    ['Natural photos sent through AI pipelines (resizing)', 'CNN',
      `Same behaviour for every resize algorithm; ${REFERENCE.cnn.exact}% exact vs ${REFERENCE.dct.exact}% (DCT) and ${REFERENCE.dwt.exact}% (DWT) at default settings.`],
    ['Cartoon stickers or flat images on WhatsApp', 'DCT',
      `${STICKER_RESULTS.dct.survived} of ${STICKER_RESULTS.photos} survived; its fine 8\u00d78 pattern stays invisible on flat backgrounds, where the CNN's colour blocks can show.`],
    ['Photos sent as WhatsApp stickers', 'CNN or DCT',
      `CNN ${STICKER_RESULTS.cnn.survived} of ${STICKER_RESULTS.photos}, DCT ${STICKER_RESULTS.dct.survived} of ${STICKER_RESULTS.photos}; DCT has more margin.`],
    ['Image will never be changed (sent as a file)', 'LSB',
      'Completely invisible, but any resize or compression destroys the message.'],
  ];
  document.getElementById('results-advice-rows').replaceChildren(...advice.map(([sit, best, why]) => {
    const tr = document.createElement('tr');
    tr.append(cell(sit), cell(best), cell(why));
    return tr;
  }));
  renderBenchmarkChart('results-chart', false);
}

// UI-only setup: nothing here touches the server, auth or measured numbers.
function initPhase3Ui() {
  // Sample thumbnails show the actual photos.
  elements.sampleThumbButtons.forEach(btn => {
    const thumb = btn.querySelector('.thumb-img');
    const sample = SAMPLES[btn.getAttribute('data-sample')];
    if (!thumb || !sample) return;
    thumb.style.backgroundImage = `url(${sample.file})`;
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
    BENCHMARK_BY_SIZE[m].map(v => `<td class="${v <= THRESHOLDS.ber ? 'better-val' : 'worse-val'}">${v.toFixed(2)}%</td>`).join('') + '</tr>';
  document.getElementById('size-table-rows').innerHTML = METHOD_ORDER.map(row).join('');
}

// Research summary for the run that just happened. The wording follows the MEASURED result
// (against THRESHOLDS.ber), so a failed run is never described as a success.
function updateAcademicSummaryParagraph() {
  const algoName = state.attackAlgo.charAt(0).toUpperCase() + state.attackAlgo.slice(1);
  const sizePx = Math.round(512 * state.attackScale);
  const ber = state.ber.toFixed(2);
  const ok = state.ber <= THRESHOLDS.ber;
  const exact = state.recoveredExact === true
    ? ' The recovered text matches the expected message exactly.'
    : (state.recoveredExact === false ? ' The recovered text still contains errors.' : '');
  const eccNote = !state.ecc ? ''
    : state.ecc.ok
      ? (state.ecc.corrected > 0 ? ` Reed-Solomon error correction repaired ${state.ecc.corrected} of 128 bytes.` : '')
      : ' Reed-Solomon error correction failed because more than 32 of 128 bytes were damaged.';
  const attack = `a <strong>${algoName}</strong> resize to <strong>${sizePx} px (${smallerLabel(state.attackScale)})</strong>`;

  const info = methodInfo(state.method);
  let summary;
  if (state.method === 'lsb') {
    summary = ok
      ? `LSB extraction after ${attack} gave <strong>${ber}% BER</strong>, within the ${THRESHOLDS.ber}% threshold.${exact}${eccNote} This is unusual for LSB and is not what the benchmark shows (${REFERENCE.lsb.ber.toFixed(2)}% average); check that the image was actually downscaled.`
      : `LSB extraction after ${attack} gave <strong>${ber}% BER</strong>, above the ${THRESHOLDS.ber}% threshold.${exact}${eccNote} ${info.failHint}`;
  } else if (state.method === 'cnn') {
    summary = ok
      ? `The CNN decoder recovered the payload after ${attack} with <strong>${ber}% BER</strong>, within the ${THRESHOLDS.ber}% threshold.${exact}${eccNote} The benchmark average across ${BENCH_ATTACKS.toLocaleString('en')} attacks is ${REFERENCE.cnn.ber.toFixed(2)}% (exact message in ${REFERENCE.cnn.exact}% of attacks).`
      : `The CNN decoder gave <strong>${ber}% BER</strong> after ${attack}, above the ${THRESHOLDS.ber}% threshold.${exact}${eccNote} This is worse than the ${REFERENCE.cnn.ber.toFixed(2)}% benchmark average. ${info.failHint}`;
  } else {
    summary = ok
      ? `The ${info.long} decoder recovered the payload after ${attack} with <strong>${ber}% BER</strong>, within the ${THRESHOLDS.ber}% threshold.${exact}${eccNote} The benchmark average across ${BENCH_ATTACKS.toLocaleString('en')} attacks is ${REFERENCE[state.method].ber.toFixed(2)}%.`
      : `The ${info.long} decoder gave <strong>${ber}% BER</strong> after ${attack}, above the ${THRESHOLDS.ber}% threshold.${exact}${eccNote} The benchmark average is ${REFERENCE[state.method].ber.toFixed(2)}%. ${info.failHint}`;
  }
  elements.evaluationSummaryText.innerHTML = summary;
}

// Initialize Application Events
// Soft UI motion (4 Nov 2026) -------------------------------------------------------------
// The sliding pill behind the selected Start tab.
function moveTabIndicator() {
  const ind = document.querySelector('.home-tab-indicator');
  const tab = document.querySelector('.home-tab[aria-selected="true"]');
  if (!ind || !tab || !tab.offsetWidth) return;
  ind.style.width = `${tab.offsetWidth}px`;
  ind.style.transform = `translateX(${tab.offsetLeft}px)`;
}

// A result card pops in once each time its state changes (pass / fail / ok / failed), so a new answer is
// noticed. Watches data-state instead of touching every place that sets it.
function initRevealOnStateChange() {
  const watch = document.querySelectorAll('.ber-hero, #ecc-status, [data-state]');
  const obs = new MutationObserver(muts => muts.forEach(m => {
    const el = m.target;
    if (['idle', 'na', 'empty'].includes(el.dataset.state)) return;
    el.classList.remove('reveal');
    void el.offsetWidth;                 // restart the animation
    el.classList.add('reveal');
  }));
  watch.forEach(el => obs.observe(el, { attributes: true, attributeFilter: ['data-state'] }));
}

// Start page, online-tool style (3 Nov 2026): choose Hide or Read, then drop a photo.
// The file is handed to the existing upload inputs, so validation and processing stay in one place.
function initHomeTool() {
  const panel = document.getElementById('home-panel');
  const tabs = [...document.querySelectorAll('.home-tab')];
  const input = document.getElementById('home-file-input');
  const zone = document.getElementById('home-dropzone');
  const err = document.getElementById('home-error');
  const TEXT = {
    hide: ['Drop a photo to hide a message in', 'PNG, JPEG or WebP, up to 4 MB. It is resized to 512 \u00d7 512 px.', 'Choose a photo'],
    read: ['Drop the image you received', 'PNG, JPEG or WebP, up to 4 MB, for example a saved WhatsApp sticker.', 'Choose an image'],
  };
  const setTab = (name, focus = false) => {
    tabs.forEach(t => {
      const on = t.dataset.homeTab === name;
      t.setAttribute('aria-selected', String(on));
      t.tabIndex = on ? 0 : -1;
      if (on) { panel.setAttribute('aria-labelledby', t.id); if (focus) t.focus(); }
    });
    panel.dataset.tab = name;
    moveTabIndicator();
    const [title, sub, btn] = TEXT[name];
    document.getElementById('home-drop-title').textContent = title;
    document.getElementById('home-drop-sub').textContent = sub;
    document.getElementById('home-browse').textContent = btn;
    err.textContent = '';
  };
  tabs.forEach((t, i) => {
    t.addEventListener('click', () => setTab(t.dataset.homeTab));
    t.addEventListener('keydown', (e) => {
      if (e.key !== 'ArrowRight' && e.key !== 'ArrowLeft') return;
      e.preventDefault();
      setTab(tabs[(i + 1) % tabs.length].dataset.homeTab, true);
    });
  });

  const handOff = async (file) => {
    err.textContent = '';
    try { await readImageFile(file); } catch (e) { err.textContent = e.message; return; }
    const dt = new DataTransfer();
    dt.items.add(file);
    if (panel.dataset.tab === 'hide') {
      handleNavigation('page-embedding');
      elements.coverFileInput.files = dt.files;
      elements.coverFileInput.dispatchEvent(new Event('change'));
    } else {
      handleNavigation('page-read');
      const readInput = document.getElementById('read-file-input');
      readInput.files = dt.files;
      readInput.dispatchEvent(new Event('change'));
      // Read straight away once the image has loaded: drop and go.
      const btn = document.getElementById('btn-read-extract');
      for (let i = 0; i < 40 && btn.disabled; i++) await new Promise(r => setTimeout(r, 50));
      if (!btn.disabled) btn.click();
    }
  };
  document.getElementById('home-browse').addEventListener('click', () => input.click());
  input.addEventListener('change', (e) => { const f = e.target.files[0]; e.target.value = ''; if (f) handOff(f); });
  ['dragenter', 'dragover'].forEach(ev => zone.addEventListener(ev, (e) => { e.preventDefault(); zone.classList.add('dragging'); }));
  ['dragleave', 'drop'].forEach(ev => zone.addEventListener(ev, () => zone.classList.remove('dragging')));
  zone.addEventListener('drop', (e) => { e.preventDefault(); const f = e.dataTransfer.files[0]; if (f) handOff(f); });

  document.querySelectorAll('.home-sample').forEach(b => b.addEventListener('click', () => {
    handleNavigation('page-embedding');
    const target = document.querySelector(`.sample-thumb-btn[data-sample="${b.dataset.sample}"]`);
    if (target) target.click();
  }));
}

function initApp() {
  initReadFlow();
  initHomeTool();
  initRevealOnStateChange();
  requestAnimationFrame(moveTabIndicator);
  window.addEventListener('resize', moveTabIndicator);
  if (document.fonts && document.fonts.ready) document.fonts.ready.then(moveTabIndicator);

  initAccounts();
  renderResultsPage();
  document.querySelectorAll('.bench-tab').forEach(b => b.addEventListener('click', () => setBenchTab(b.dataset.benchTab)));

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

      loadSamplePhoto(elements.coverCanvas, sampleType).then(() => {
        updateFlatTip();
        writeLog(`[INFO] Sample photo '${SAMPLE_NAMES[sampleType] || sampleType}' loaded.`, "info");
      }).catch(err => writeLog(`[ERROR] ${err.message}`, "error"));
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
      updateFlatTip();
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
        updateFlatTip();
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
    elements.secretMessageInput.value = "Hidden message that survives downscaling.";
    updatePayloadMetrics();
    const firstSample = document.querySelector('.sample-thumb-btn[data-sample="astronaut"]');
    if (firstSample) firstSample.click();
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
    clearInlineError(stegoSourceBox());
    if (!state.embedded) {
      showInlineError(stegoSourceBox(), "No stego image from step 2 yet. Embed a message in step 1 first, or upload a stego image.");
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

    elements.stegoSourceStatus.innerText = `Stego loaded. Method: ${methodInfo(state.method).long}.`;

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

  // Download the resized image exactly as the server returned it (PNG, lossless; no canvas re-encode).
  document.getElementById('btn-download-resized').addEventListener('click', () => {
    if (!state.downscaledImgData || !state.lastAttack) return;
    const link = document.createElement('a');
    link.download = `resized_${state.method}_${state.lastAttack.algo}_${state.lastAttack.size}px.png`;
    link.href = state.downscaledImgData;
    link.click();
    writeLog(`[INFO] Resized image downloaded (${state.lastAttack.size}x${state.lastAttack.size}px, ${state.lastAttack.algo}).`, 'info');
  });

  elements.btnContinueToExtraction.addEventListener('click', () => {
    elements.btnImportAttacked.click();
    handleNavigation('page-extraction');
  });

  elements.btnImportAttacked.addEventListener('click', () => {
    clearInlineError(attackedSourceBox());
    if (!state.attackSimulated) {
      showInlineError(attackedSourceBox(), "No attacked image from step 3 yet. Run the attack first, or upload an image.");
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
      clearInlineError(stegoSourceBox());
    } catch (err) { showInlineError(stegoSourceBox(), `${err.message} Choose another file.`); }
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
      clearInlineError(attackedSourceBox());
    } catch (err) { showInlineError(attackedSourceBox(), `${err.message} Choose another file.`); }
  });

  elements.btnExportReport.addEventListener('click', () => {
    elements.reportDate.innerText = new Date().toLocaleDateString('en-US', { year: 'numeric', month: 'long', day: 'numeric' });
    const coverName = SAMPLE_NAMES[state.coverScene] ? `${SAMPLE_NAMES[state.coverScene]} (scikit-image sample photo)` : state.coverScene;
    elements.reportCoverName.innerText = `${coverName}, 512x512 px`;
    elements.reportMsgLen.innerText = `${[...state.message].length} characters (${utf8Bytes(state.message)} of ${MAX_MESSAGE_BYTES} bytes; 1024 bits with error correction)`;

    const algoName = elements.downscaleAlgorithm.options[elements.downscaleAlgorithm.selectedIndex].text;

    const attacked = state.attackSimulated && !state.downscaledUploaded;
    elements.reportTableAlgo.innerText = attacked ? algoName : (state.downscaledUploaded ? 'Uploaded image (attack not run in this app)' : '--');
    elements.reportTableScale.innerText = attacked
      ? `${smallerLabel(state.attackScale)} (${elements.attackOutputCanvas.width}x${elements.attackOutputCanvas.height} px)` : '--';

    // Benchmark rows: always the real benchmark averages (REFERENCE), never per-run numbers.
    elements.reportLsbBer.innerText = `${REFERENCE.lsb.ber.toFixed(2)}%`;
    elements.reportCnnBer.innerText = `${REFERENCE.cnn.ber.toFixed(2)}%`;
    document.getElementById('report-lsb-psnr').innerText = `${REFERENCE.lsb.psnr.toFixed(2)} dB`;
    document.getElementById('report-cnn-psnr').innerText = `${REFERENCE.cnn.psnr.toFixed(2)} dB`;
    document.getElementById('report-lsb-ssim').innerText = REFERENCE.lsb.ssim.toFixed(4);
    document.getElementById('report-cnn-ssim').innerText = REFERENCE.cnn.ssim.toFixed(4);

    // This session's run: only what was actually measured; "--" when a step wasn't done.
    document.getElementById('report-run-method').innerText =
      state.extracted ? methodInfo(state.method).long : '--';
    document.getElementById('report-run-ber').innerText =
      state.extracted && state.ber !== null ? `${state.ber.toFixed(2)}%` : '-- (no extraction with an expected message)';
    const eccText = !state.ecc ? '' : state.ecc.ok
      ? ` (error correction repaired ${state.ecc.corrected} of 128 bytes)` : ' (error correction failed)';
    document.getElementById('report-run-exact').innerText =
      state.recoveredExact === true ? `Yes${eccText}` : state.recoveredExact === false ? `No${eccText}` : '--';

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

  loadSamplePhoto(elements.coverCanvas, 'astronaut').then(updateFlatTip).catch(() => {});
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
