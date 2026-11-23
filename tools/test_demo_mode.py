"""Check demo mode against a RUNNING demo server (start it first with run_demo.bat).

    python tools\\test_demo_mode.py

Checks: health reports demo mode, embed works without signing in, account routes are not there,
and a request that looks like it came through a proxy (Tailscale Funnel) is refused.
"""
import base64
import io
import json
import sys
import urllib.error
import urllib.request

import numpy as np
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

for name, ok in results:
    print(f"{'PASS' if ok else 'FAIL'}  {name}")
passed = sum(ok for _, ok in results)
print(f"\n{passed}/{len(results)} checks passed")
sys.exit(0 if passed == len(results) else 1)
