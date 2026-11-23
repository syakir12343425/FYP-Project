"""Check demo mode against a RUNNING demo server (start it first with run_demo.bat).

    python tools\\test_demo_mode.py

Checks: health reports demo mode, embed works without signing in, account routes are not there,
a request that looks like it came through a proxy (Tailscale Funnel) is refused, and the CNN
(model D) round trip works: embed, shrink to 384 px (bicubic), read back.
"""
import base64
import io
import json
import sys
import urllib.error
import urllib.request

import numpy as np
from pathlib import Path

from PIL import Image

BASE = "http://127.0.0.1:8000"


def call(path, body=None, headers=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(BASE + path, data=data, method="POST" if body is not None else "GET",
                                 headers={"Content-Type": "application/json", **(headers or {})})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.status, json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        return e.code, {}


def sample_png():
    rng = np.random.default_rng(1)
    img = (rng.random((512, 512, 3)) * 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def sample_photo():
    # A real photo from the app's own samples: the CNN was trained on photos, not noise.
    path = Path(__file__).resolve().parent.parent / "fyp-stego-app" / "assets" / "samples" / "cat.jpg"
    return "data:image/jpeg;base64," + base64.b64encode(path.read_bytes()).decode()


results = []
status, body = call("/api/health")
results.append(("health says demo mode", status == 200 and body.get("demo") is True))
status, _ = call("/api/auth/me")
results.append(("account routes are not mounted (404)", status == 404))
embed = {"cover_image": sample_png(), "message": "demo check", "method": "dct"}
status, body = call("/api/embed", embed)
results.append(("embed works without signing in", status == 200 and "stego_image" in body))
status, _ = call("/api/embed", embed, {"X-Forwarded-For": "203.0.113.5"})
results.append(("proxied request refused (403)", status == 403))
status, _ = call("/api/health", headers={"Tailscale-Funnel-Request": "?1"})
results.append(("Funnel request refused (403)", status == 403))

msg = "demo check"
status, body = call("/api/embed", {"cover_image": sample_photo(), "message": msg, "method": "cnn"})
cnn_ok = status == 200
if cnn_ok:
    status, body = call("/api/downscale", {"image": body["stego_image"], "algorithm": "bicubic", "scale": 0.75})
    cnn_ok = status == 200 and body.get("size") == 384
if cnn_ok:
    status, body = call("/api/extract", {"image": body["downscaled_image"], "method": "cnn", "original_message": msg})
    cnn_ok = status == 200 and body.get("recovered_message") == msg and body.get("ecc_ok") is True
    if status == 200:
        print(f"CNN at 384 px: raw BER {body.get('ber', 0):.2%}, recovered {body.get('recovered_message')!r}")
results.append(("CNN round trip: embed, shrink to 384 px, read back exactly", cnn_ok))

for name, ok in results:
    print(f"{'PASS' if ok else 'FAIL'}  {name}")
passed = sum(ok for _, ok in results)
print(f"\n{passed}/{len(results)} checks passed")
sys.exit(0 if passed == len(results) else 1)
