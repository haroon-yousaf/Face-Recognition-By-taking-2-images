import base64
import io
import time
import threading
from collections import deque

import numpy as np
import cv2
from scipy.signal import butter, filtfilt, welch
from flask import Flask, request, jsonify, Response

# -----------------------------
# Config
# -----------------------------
APP_TITLE = "Face Liveness Detection"
FRAME_DOWNSCALE = 0.75         # resize incoming frames to save CPU
MAX_HISTORY_SEC = 20           # length of rolling history for rPPG and motion
TARGET_FPS = 15                # analysis frame rate target
GREEN_BAND = (0.7, 3.0)        # 42 to 180 bpm
LIVE_THRESHOLD = 0.6           # final decision threshold

# Fusion weights (tune on validation clips)
W_TEXTURE = 0.35
W_BLINK   = 0.20
W_MOTION  = 0.15
W_RPPG    = 0.30

# -----------------------------
# Flask app
# -----------------------------
app = Flask(__name__)

# -----------------------------
# State per single session
# For multi-user, key these by a client_id
# -----------------------------
lock = threading.Lock()
state = {
    "last_ts": 0.0,
    "g_sig": deque(maxlen=MAX_HISTORY_SEC * 30),
    "t_sig": deque(maxlen=MAX_HISTORY_SEC * 30),
    "blink_events": deque(maxlen=200),
    "last_eye_seen": True,
    "eye_miss_count": 0,
    "flow_points": None,
    "flow_prev_gray": None,
    "motion_buf": deque(maxlen=30),
    "fps": TARGET_FPS
}

# -----------------------------
# OpenCV detectors
# -----------------------------
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade  = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye_tree_eyeglasses.xml")

# -----------------------------
# Helpers
# -----------------------------
def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = max(0.0001, lowcut / nyq)
    high = min(0.9999, highcut / nyq)
    b, a = butter(order, [low, high], btype="band")
    return b, a

def bandpass_filter(sig, fs, low, high):
    if len(sig) < fs:  # need ~1 second
        return sig
    b, a = butter_bandpass(low, high, fs)
    return filtfilt(b, a, sig)

def laplacian_focus(gray):
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

def moire_score(gray):
    # crude frequency ring energy as a proxy for screen patterns
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    mag = np.log(np.abs(fshift) + 1e-6)
    h, w = mag.shape
    cy, cx = h // 2, w // 2
    r1, r2 = min(h, w) * 0.15, min(h, w) * 0.40
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((Y - cy) ** 2 + (X - cx) ** 2)
    ring = (dist > r1) & (dist < r2)
    return float(mag[ring].mean())

def snr_band(sig, fs, fl, fh):
    if len(sig) < fs * 2:
        return 0.0
    f, pxx = welch(sig, fs=fs, nperseg=min(len(sig), 256))
    band = (f >= fl) & (f <= fh)
    signal_power = np.max(pxx[band]) if np.any(band) else 0.0
    noise_power = np.mean(pxx[~band]) if np.any(~band) else 1e-6
    return float(signal_power / (noise_power + 1e-6))

def decode_base64_image(data_url):
    header, b64data = data_url.split(",", 1)
    img_bytes = base64.b64decode(b64data)
    arr = np.frombuffer(img_bytes, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return frame

def good_features(gray, roi=None, max_corners=100):
    mask = None
    if roi is not None:
        mask = np.zeros_like(gray)
        x, y, w, h = roi
        mask[y:y+h, x:x+w] = 255
    pts = cv2.goodFeaturesToTrack(gray, maxCorners=max_corners, qualityLevel=0.01, minDistance=7, mask=mask)
    return pts

# -----------------------------
# Routes
# -----------------------------
@app.get("/")
def index():
    # Minimal HTML that captures camera frames and calls /analyze
    html = f"""
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8" />
  <title>{APP_TITLE}</title>
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <style>
    body {{ font-family: system-ui, Arial, sans-serif; margin: 16px; }}
    #wrap {{ display: grid; gap: 12px; max-width: 900px; }}
    #scores {{ display: grid; gap: 4px; }}
    .bar {{ height: 10px; background: #ddd; border-radius: 6px; overflow: hidden; }}
    .bar > div {{ height: 100%; background: #4caf50; width: 0%; transition: width 150ms; }}
    video, canvas {{ width: 100%; max-width: 480px; border-radius: 8px; }}
    .row {{ display: flex; gap: 24px; flex-wrap: wrap; align-items: flex-start; }}
    .pill {{ display: inline-block; padding: 6px 10px; border-radius: 999px; color: white; font-weight: 600; }}
  </style>
</head>
<body>
  <div id="wrap">
    <h2>{APP_TITLE}</h2>
    <div class="row">
      <video id="v" autoplay playsinline muted></video>
      <canvas id="c" width="640" height="480" style="display:none"></canvas>
      <div style="min-width:260px;flex:1">
        <h3>Live result</h3>
        <div id="verdict" class="pill" style="background:#999">Waiting</div>
        <div style="margin-top:8px;font-size:14px;">Live score: <b id="liveprob">0.00</b></div>

        <div id="scores" style="margin-top:12px">
          <div>Texture <span id="t1">0.00</span><div class="bar"><div id="b1"></div></div></div>
          <div>Blink <span id="t2">0.00</span><div class="bar"><div id="b2"></div></div></div>
          <div>Motion <span id="t3">0.00</span><div class="bar"><div id="b3"></div></div></div>
          <div>rPPG <span id="t4">0.00</span><div class="bar"><div id="b4"></div></div></div>
        </div>
        <p style="font-size:13px;opacity:.8;margin-top:12px">
          Tips: Make sure your face is well lit. Hold the device steady. Natural blinking is expected.
        </p>
      </div>
    </div>
  </div>

<script>
const v = document.getElementById('v');
const c = document.getElementById('c');
const ctx = c.getContext('2d');
const verdict = document.getElementById('verdict');
const liveprob = document.getElementById('liveprob');
const t1 = document.getElementById('t1'), t2 = document.getElementById('t2'),
      t3 = document.getElementById('t3'), t4 = document.getElementById('t4');
const b1 = document.getElementById('b1'), b2 = document.getElementById('b2'),
      b3 = document.getElementById('b3'), b4 = document.getElementById('b4');

async function init() {{
  try {{
    const stream = await navigator.mediaDevices.getUserMedia({{ video: true, audio: false }});
    v.srcObject = stream;
  }} catch (e) {{
    alert('Camera access failed: ' + e);
  }}
  loop();
}}

async function loop() {{
  if (v.videoWidth > 0) {{
    c.width = v.videoWidth;
    c.height = v.videoHeight;
    ctx.drawImage(v, 0, 0, c.width, c.height);
    const dataUrl = c.toDataURL('image/jpeg', 0.85);
    try {{
      const res = await fetch('/analyze', {{
        method: 'POST',
        headers: {{ 'Content-Type': 'application/json' }},
        body: JSON.stringify({{ frame: dataUrl }})
      }});
      const j = await res.json();
      liveprob.textContent = j.live_prob.toFixed(2);
      t1.textContent = j.texture_score.toFixed(2);
      t2.textContent = j.blink_score.toFixed(2);
      t3.textContent = j.motion_score.toFixed(2);
      t4.textContent = j.rppg_score.toFixed(2);
      b1.style.width = (j.texture_score*100).toFixed(0)+'%';
      b2.style.width = (j.blink_score*100).toFixed(0)+'%';
      b3.style.width = (j.motion_score*100).toFixed(0)+'%';
      b4.style.width = (j.rppg_score*100).toFixed(0)+'%';
      verdict.textContent = j.verdict;
      verdict.style.background = j.verdict === 'LIVE' ? '#2e7d32' : '#c62828';
    }} catch(e) {{
      // ignore transient errors
    }}
  }}
  setTimeout(loop, {int(1000 / TARGET_FPS)});
}}

init();
</script>
</body>
</html>
"""
    return Response(html, content_type="text/html")

@app.post("/analyze")
def analyze():
    data = request.get_json(silent=True)
    if not data or "frame" not in data:
        return jsonify({"error": "no frame"}), 400

    frame = decode_base64_image(data["frame"])
    if frame is None:
        return jsonify({"error": "bad frame"}), 400

    # downscale for speed
    if FRAME_DOWNSCALE != 1.0:
        frame = cv2.resize(frame, None, fx=FRAME_DOWNSCALE, fy=FRAME_DOWNSCALE, interpolation=cv2.INTER_AREA)

    h, w, _ = frame.shape
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # face detection
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))
    texture_score = 0.0
    blink_score = 0.0
    motion_score = 0.0
    rppg_score = 0.0

    with lock:
        now = time.time()
        if len(faces) > 0:
            # choose largest face
            x, y, fw, fh = max(faces, key=lambda r: r[2]*r[3])
            face_gray = gray[y:y+fh, x:x+fw]
            face_bgr  = frame[y:y+fh, x:x+fw]

            # texture cues
            focus = laplacian_focus(face_gray)
            moire = moire_score(face_gray)
            focus_n = np.tanh(focus / 150.0)
            moire_penalty = np.tanh(max(0.0, (moire - 5.0)) / 5.0)
            texture_score = float(np.clip(focus_n - 0.5 * moire_penalty, 0.0, 1.0))

            # eye detection for blink heuristic
            eyes = eye_cascade.detectMultiScale(face_gray, scaleFactor=1.1, minNeighbors=3, minSize=(20, 20))
            eyes_seen = len(eyes) >= 1
            if not eyes_seen:
                state["eye_miss_count"] += 1
            else:
                # detect a transition closed->open as a blink end
                if state["last_eye_seen"] is False and state["eye_miss_count"] >= int(0.1 * state["fps"]):
                    state["blink_events"].append(now)
                state["eye_miss_count"] = 0
            state["last_eye_seen"] = eyes_seen

            # blink score from recent blink rate
            recent = [t for t in state["blink_events"] if now - t < 20.0]
            blink_rate = len(recent) * 3.0  # per minute estimate
            blink_score = float(np.clip(blink_rate / 10.0, 0.0, 1.0))

            # motion via sparse optical flow inside face ROI
            face_roi = (x, y, fw, fh)
            if state["flow_prev_gray"] is None:
                state["flow_prev_gray"] = gray.copy()
                state["flow_points"] = good_features(gray, roi=face_roi, max_corners=80)
            else:
                if state["flow_points"] is None:
                    state["flow_points"] = good_features(gray, roi=face_roi, max_corners=80)
                prev = state["flow_prev_gray"]
                pts0 = state["flow_points"]
                if pts0 is not None:
                    pts1, st, err = cv2.calcOpticalFlowPyrLK(prev, gray, pts0, None, winSize=(15, 15), maxLevel=2,
                                                             criteria=(cv2.TermCriteria_EPS | cv2.TermCriteria_COUNT, 10, 0.03))
                    if pts1 is not None and st is not None:
                        good0 = pts0[st == 1]
                        good1 = pts1[st == 1]
                        if len(good0) > 0:
                            delta = np.linalg.norm(good1 - good0, axis=1)
                            mean_motion = float(np.mean(delta))
                            state["motion_buf"].append(mean_motion)
                state["flow_prev_gray"] = gray.copy()
                state["flow_points"] = good_features(gray, roi=face_roi, max_corners=80)

            # rPPG green signal from cheek area
            cx = x + int(0.65 * fw)
            cy = y + int(0.55 * fh)
            bw = max(10, int(0.20 * fw))
            bh = max(10, int(0.18 * fh))
            x1 = max(0, min(w - 1, cx - bw // 2))
            y1 = max(0, min(h - 1, cy - bh // 2))
            x2 = max(0, min(w, x1 + bw))
            y2 = max(0, min(h, y1 + bh))
            roi = frame[y1:y2, x1:x2, :]
            g_mean = float(np.mean(roi[:, :, 1])) if roi.size else 0.0

            state["g_sig"].append(g_mean)
            state["t_sig"].append(now)

            # compute rPPG SNR
            rppg_snr = 0.0
            times = np.array(state["t_sig"])
            g = np.array(state["g_sig"])
            if len(times) > state["fps"] * 3:
                t0 = times[0]
                t = times - t0
                fs = state["fps"]
                # resample to constant rate
                t_uniform = np.linspace(t[0], t[-1], int(max(1.0, (t[-1] - t[0]) * fs)))
                if len(t_uniform) > 5:
                    g_uniform = np.interp(t_uniform, t, g)
                    g_d = g_uniform - np.mean(g_uniform)
                    g_f = bandpass_filter(g_d, fs=fs, low=GREEN_BAND[0], high=GREEN_BAND[1])
                    rppg_snr = snr_band(g_f, fs, GREEN_BAND[0], GREEN_BAND[1])
            rppg_score = float(np.clip(rppg_snr / 3.0, 0.0, 1.0))

            # motion score from recent buffer
            motion = np.mean(state["motion_buf"]) if len(state["motion_buf"]) else 0.0
            motion_score = float(np.clip(1.0 - abs(motion - 1.5) / 3.0, 0.0, 1.0))

        else:
            # no face found; decay state a bit
            state["motion_buf"].append(0.0)

        # fuse
        live_prob = (
            W_TEXTURE * texture_score
            + W_BLINK   * blink_score
            + W_MOTION  * motion_score
            + W_RPPG    * rppg_score
        )
        verdict = "LIVE" if live_prob >= LIVE_THRESHOLD else "SUSPECT"

    return jsonify({
        "texture_score": texture_score,
        "blink_score": blink_score,
        "motion_score": motion_score,
        "rppg_score": rppg_score,
        "live_prob": float(live_prob),
        "verdict": verdict
    })

if __name__ == "__main__":
    print(f"Starting {APP_TITLE} on http://127.0.0.1:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)
