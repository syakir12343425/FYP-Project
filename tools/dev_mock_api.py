r"""DEVELOPMENT ONLY: a stand-in for server.py so the UI can be worked on without PyTorch, the checkpoint or a database.

- embed / downscale / extract use the REAL project code for LSB, DCT and DWT; "cnn" requests fall back to LSB.
- Sign-in is fake: everyone is signed in automatically (or use password Right-Pass-123!, code 123456).
- Accounts and two-step endpoints return made-up data so those pages can be designed and clicked through.
Never show numbers from this server as results, and never run it anywhere but 127.0.0.1.

    python tools\dev_mock_api.py                          (admin, two-step on)
    $env:MOCK_ROLE="user"; python tools\dev_mock_api.py
    $env:MOCK_LOGGEDOUT="1"; $env:MOCK_MFA="1"; python tools\dev_mock_api.py   (login + code step)
    $env:MOCK_MFA="0"; python tools\dev_mock_api.py     (admin who still has to turn two-step on)
"""
import base64, io, json, sys
from http.server import BaseHTTPRequestHandler, HTTPServer
import numpy as np, cv2
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
sys.path.insert(0, sys.argv[1] if len(sys.argv) > 1 else str(__import__("pathlib").Path(__file__).resolve().parent.parent))
from payload import encode_message, decode_payload
from classical_stego import dct_embed, dct_extract, dwt_embed, dwt_extract
from lsb_baseline import lsb_embed, lsb_extract

ALG = {"nearest": cv2.INTER_NEAREST, "bilinear": cv2.INTER_LINEAR, "bicubic": cv2.INTER_CUBIC,
       "lanczos": cv2.INTER_LANCZOS4, "area": cv2.INTER_AREA}

def dec(u):
    return np.array(Image.open(io.BytesIO(base64.b64decode(u.split(",", 1)[1]))).convert("RGB"))

def enc(a):
    b = io.BytesIO(); Image.fromarray(a).save(b, "PNG")
    return "data:image/png;base64," + base64.b64encode(b.getvalue()).decode()

import os, secrets, re as _re
ROLE = os.environ.get("MOCK_ROLE", "admin"); MUST = os.environ.get("MOCK_MUST", "0") == "1"
LOGGED_OUT = os.environ.get("MOCK_LOGGEDOUT", "0") == "1"; MFA = os.environ.get("MOCK_MFA", "1") == "1"
CODES_LEFT = 10
QR = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 21 21"><rect width="21" height="21" fill="#fff"/><rect x="1" y="1" width="7" height="7"/><rect x="13" y="1" width="7" height="7"/><rect x="1" y="13" width="7" height="7"/><rect x="10" y="10" width="2" height="2"/></svg>'
USERS = [{"id": 1, "username": "ADMIN", "role": "admin", "disabled": False, "locked": False, "must_change_password": False,
          "created_at": "2026-10-01T09:00:00+00:00", "last_login_at": "2026-11-03T12:30:00+00:00"},
         {"id": 2, "username": "supervisor", "role": "user", "disabled": False, "locked": False, "must_change_password": False,
          "created_at": "2026-11-01T09:00:00+00:00", "last_login_at": "2026-11-02T10:05:00+00:00", "mfa_enabled": True,
          "email": "syaril@example.edu.my", "email_verified": True},
         {"id": 3, "username": "tester01", "role": "user", "disabled": False, "locked": True, "must_change_password": False,
          "created_at": "2026-11-01T09:00:00+00:00", "last_login_at": None},
         {"id": 4, "username": "classmate_amir", "role": "user", "disabled": False, "locked": False, "must_change_password": False,
          "pending_approval": True, "created_at": "2026-11-04T01:00:00+00:00", "last_login_at": None}]
def sess():
    return {"access_token": "t", "expires_in": 900, "username": "ADMIN" if ROLE == "admin" else "student1", "role": ROLE,
            "must_change_password": MUST, "mfa_enabled": MFA, "mfa_required": True}
RC = ["k7m2q-9xwpr", "a3bcd-efgh2", "pq4rs-tuvw5", "xy6za-bcde7", "fg8hj-kmnp9", "qr2st-uvwx3", "yz4ab-cdef5", "gh6jk-mnpq7", "rs8tu-vwxy9", "za2bc-defg3"]

class H(BaseHTTPRequestHandler):
    def _h(self, code=200):
        self.send_response(code)
        self.send_header("Access-Control-Allow-Origin", "http://localhost:8080")
        self.send_header("Access-Control-Allow-Credentials", "true")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.send_header("Access-Control-Allow-Methods", "*")
        self.send_header("Content-Type", "application/json")
        self.end_headers()
    def log_message(self, *a): pass
    def do_OPTIONS(self): self._h()
    def do_GET(self):
        if self.path.endswith("/auth/mfa/status"):
            self._h(); self.wfile.write(json.dumps({"mfa_enabled": MFA, "recovery_codes_left": CODES_LEFT, "required": ROLE == "admin"}).encode()); return
        if self.path.endswith("/admin/users"):
            if ROLE != "admin": self._h(403); self.wfile.write(b'{"detail":"Admins only"}'); return
            if not MFA: self._h(403); self.wfile.write(b'{"detail":"Turn on two-step verification (Account page) before managing accounts"}'); return
            self._h(); self.wfile.write(json.dumps({"users": sorted(USERS, key=lambda u: not u.get("pending_approval"))}).encode()); return
        if self.path.endswith("/auth/signup-open"):
            self._h(); self.wfile.write(b'{"open": true}'); return
        self._h(404); self.wfile.write(b"{}")
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        p = self.path
        global MUST, MFA, LOGGED_OUT, CODES_LEFT
        def err(code, msg): self._h(code); self.wfile.write(json.dumps({"detail": msg}).encode())
        m = _re.search(r"/admin/users/(\d+)/(\w[\w-]*)$", p)
        if p.endswith("/auth/refresh"):
            if LOGGED_OUT: return err(401, "No session")
            out = sess()
        elif p.endswith("/auth/verify-email"):
            if body.get("token") != "good-token-abcdefghijklmnop": return err(400, "This link is invalid or has expired. Sign in to get a new one.")
            out = {"ok": True, "message": "Email confirmed. You can sign in now."}
        elif p.endswith("/auth/resend-verification"):
            out = {"ok": True, "message": "A new link was sent to u***@example.com."}
        elif p.endswith("/auth/register"):
            n = body.get("username", "")
            if "@" not in body.get("email", ""): return err(400, "Enter a valid email address")
            if any(u["username"].lower() == n.lower() for u in USERS): return err(409, "That username is taken")
            if len(body.get("password", "")) < 12: return err(400, "Use at least 12 characters.")
            USERS.append({"id": len(USERS) + 1, "username": n, "role": "user", "disabled": False, "locked": False,
                          "must_change_password": False, "pending_approval": True, "created_at": "2026-11-04T02:00:00+00:00", "last_login_at": None})
            USERS[-1].update({"email": body.get("email"), "email_verified": False, "pending_approval": False})
            out = {"ok": True, "message": "Check your email for a link to confirm your account."}
        elif p.endswith("/auth/login"):
            if body.get("username") == "waiting_user": return err(403, "Your account is waiting for an admin to approve it")
            if body.get("username") == "unconfirmed": return err(403, "Confirm your email first: open the link we sent to u***@example.com")
            if body.get("password") != "Right-Pass-123!": return err(401, "Invalid credentials or account temporarily locked")
            out = {"mfa_required": True, "mfa_token": "mfa-tok"} if MFA else sess()
        elif p.endswith("/auth/mfa"):
            if body.get("code") not in ("123456", "k7m2q-9xwpr"): return err(401, "Wrong code")
            LOGGED_OUT = False; out = sess()
        elif p.endswith("/auth/mfa/setup"):
            if body.get("password") != "Right-Pass-123!": return err(400, "Password is wrong")
            out = {"secret": "HNDSVSCNSQDDKZJ6EGIFKT737Z5CWGVN", "otpauth_uri": "otpauth://totp/x", "qr_data_uri": "data:image/svg+xml;charset=utf-8," + __import__("urllib.parse").parse.quote(QR.replace("<svg ", "<svg ", 1))}
        elif p.endswith("/auth/mfa/enable"):
            if body.get("code") != "123456": return err(400, "Wrong code. Check that your phone's clock is set automatically, then try the newest code.")
            MFA = True; CODES_LEFT = 10; out = {"mfa_enabled": True, "recovery_codes": RC}
        elif p.endswith("/auth/mfa/disable"):
            if ROLE == "admin": return err(403, "Admins must keep two-step verification on")
            MFA = False; out = {"mfa_enabled": False}
        elif p.endswith("/auth/mfa/recovery-codes"):
            if body.get("code") != "123456": return err(400, "Password or code is wrong")
            CODES_LEFT = 10; out = {"recovery_codes": RC}
        elif p.endswith("/auth/change-password"):
            if body["current_password"] != "Temp-Pass-123!":
                self._h(400); self.wfile.write(b'{"detail":"Current password is wrong"}'); return
            if len(body["new_password"]) < 12:
                self._h(400); self.wfile.write(b'{"detail":"Use at least 12 characters."}'); return
            MUST = False; out = sess()
        elif p.endswith("/admin/users"):
            name = body["username"]
            if any(u["username"].lower() == name.lower() for u in USERS):
                self._h(409); self.wfile.write(b'{"detail":"That username is taken"}'); return
            u = {"id": len(USERS) + 1, "username": name, "role": "user", "disabled": False, "locked": False,
                 "must_change_password": True, "created_at": "2026-11-03T13:00:00+00:00", "last_login_at": None}
            USERS.append(u); out = {"user": u, "temporary_password": "k7Qm-xR4pZ9w#tBv"}
        elif m:
            u = next(x for x in USERS if x["id"] == int(m.group(1))); act = m.group(2)
            if act == "disable": u["disabled"] = True
            elif act == "enable": u["disabled"] = False
            elif act == "unlock": u["locked"] = False
            elif act == "reset-password": u["must_change_password"] = True
            elif act == "reset-mfa": u["mfa_enabled"] = False
            elif act == "approve": u["pending_approval"] = False
            elif act == "decline": u["pending_approval"] = False; u["disabled"] = True
            out = {"user": u, "temporary_password": "Wn3s?Hc8Lq2r-Yd5"} if act == "reset-password" else {"user": u}
        elif p.endswith("/embed"):
            cover = dec(body["cover_image"])
            if cover.shape[:2] != (512, 512): cover = np.array(Image.fromarray(cover).resize((512, 512), Image.LANCZOS))
            bits = encode_message(body["message"])
            st = {"dct": dct_embed, "dwt": dwt_embed, "lsb": lsb_embed}[body["method"]](cover, bits)
            out = {"stego_image": enc(st), "psnr": float(psnr(cover, st, data_range=255)),
                   "ssim": float(ssim(cover, st, channel_axis=2, data_range=255))}
        elif p.endswith("/downscale"):
            img = dec(body["image"]); n = round(img.shape[0] * body["scale"])
            out = {"downscaled_image": enc(cv2.resize(img, (n, n), interpolation=ALG[body["algorithm"]])), "size": n}
        elif p.endswith("/extract"):
            img = dec(body["image"]); m = body["method"]
            bits = dct_extract(img) if m == "dct" else dwt_extract(img) if m == "dwt" else lsb_extract(img, 1024)
            d = decode_payload(bits)
            ber = float(np.mean(encode_message(body["original_message"]) != bits)) if body.get("original_message") else None
            out = {"recovered_message": d["text"], "ber": ber, "extracted_psnr": None, "extracted_ssim": None,
                   "ecc_ok": d["ok"], "ecc_corrected_bytes": d["corrected_bytes"]}
        else:
            out = {}
        self._h(); self.wfile.write(json.dumps(out).encode())

print("DEV MOCK API on http://127.0.0.1:8000 (not the real server; Ctrl+C to stop)")
HTTPServer(("127.0.0.1", 8000), H).serve_forever()
