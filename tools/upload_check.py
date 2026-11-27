"""Upload security check for RobustStego. Run it against the DEMO server (no sign-in needed).

    .\\run_demo.bat                     (window 1, leave it running)
    python tools\\upload_check.py       (window 2)

Each case sends a hostile "image" to /api/embed and /api/extract and checks that the server:
  - refuses it with a 4xx error (never 500, never a crash),
  - or, for a real image with a hidden payload appended (a polyglot), returns a clean PNG
    that does NOT contain the payload,
  - and is still alive afterwards (/api/health answers).
Nothing here is harmful: the "executable" is only the first bytes of a Windows program header.
"""
import base64
import io
import json
import sys
import time
import urllib.error
import urllib.request

from PIL import Image

BASE = "http://127.0.0.1:8000"
PAYLOAD = b"<?php system($_GET['c']); ?><script>alert(1)</script>"


def call(path, body, timeout=120):
    data = json.dumps(body).encode()
    req = urllib.request.Request(BASE + path, data=data, method="POST", headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read() or b"{}"), time.time() - t0
    except urllib.error.HTTPError as e:
        return e.code, {}, time.time() - t0
    except Exception as e:  # connection reset, timeout: the server may have crashed
        return 0, {"error": str(e)}, time.time() - t0


def alive():
    try:
        with urllib.request.urlopen(BASE + "/api/health", timeout=10) as r:
            return r.status == 200
    except Exception:
        return False


def data_url(raw, mime="image/png"):
    return f"data:{mime};base64," + base64.b64encode(raw).decode()


def png_bytes(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def bomb(side):
    """A tiny PNG file that claims side x side pixels (a decompression bomb)."""
    return png_bytes(Image.new("1", (side, side), 0))


cases = [
    ("Windows program (MZ header) named as PNG", data_url(b"MZ\x90\x00" + b"\x00" * 4000 + b"This program cannot be run in DOS mode")),
    ("PHP / script text named as PNG", data_url(PAYLOAD * 50)),
    ("Random bytes named as JPEG", data_url(bytes(range(256)) * 40, "image/jpeg")),
    ("SVG with a script", data_url(b'<svg xmlns="http://www.w3.org/2000/svg"><script>alert(1)</script></svg>', "image/svg+xml")),
    ("Not base64 at all", "data:image/png;base64,!!!not-base64!!!"),
    ("Empty image", data_url(b"")),
    ("Truncated PNG (header only)", data_url(png_bytes(Image.new("RGB", (64, 64), "red"))[:60])),
    ("Decompression bomb, 30000 x 30000 px", data_url(bomb(30000))),
    ("Large image, 12000 x 12000 px", data_url(bomb(12000))),
]

results = []
print("Is the demo server up?", "yes" if alive() else "NO - start run_demo.bat first")
if not alive():
    sys.exit(1)

for name, url in cases:
    for path, field in (("/api/embed", "cover_image"), ("/api/extract", "image")):
        body = {field: url, "method": "cnn"}
        if path == "/api/embed":
            body["message"] = "test"
        status, _, secs = call(path, body)
        ok = 400 <= status < 500 and alive()
        results.append(ok)
        print(f"{'PASS' if ok else 'FAIL'}  {path:13s} {name:42s} HTTP {status}  {secs:5.1f}s  server alive: {alive()}")

# Polyglot: a real photo with a script appended after the image data.
real = png_bytes(Image.new("RGB", (512, 512), (120, 160, 200))) + PAYLOAD
status, body, secs = call("/api/embed", {"cover_image": data_url(real), "message": "test", "method": "dct"})
clean = False
if status == 200 and "stego_image" in body:
    out = base64.b64decode(body["stego_image"].split(",", 1)[-1])
    clean = PAYLOAD not in out and b"<?php" not in out and b"<script" not in out
ok = status == 200 and clean
results.append(ok)
print(f"{'PASS' if ok else 'FAIL'}  /api/embed     Real PNG + appended script (polyglot)      HTTP {status}  "
      f"returned image is {'clean (re-encoded from pixels)' if clean else 'NOT clean'}")

print(f"\n{sum(results)}/{len(results)} checks passed")
sys.exit(0 if all(results) else 1)
