# app.py
# CNIC template verification web app.
# - Compares an uploaded image against a standard template.
# - Provides a simple "Pass" or "Fail" result along with debug scores.

import os
import math
import cv2
import numpy as np

from flask import (
    Flask, request, render_template_string, redirect, url_for, flash
)
from werkzeug.utils import secure_filename

# -------------------- Config --------------------
ALLOWED_EXTS = {"jpg", "jpeg", "png"}
MAX_CONTENT_LENGTH = 8 * 1024 * 1024  # 8 MB
BASE_DIR = os.path.dirname(__file__)
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
TEMPLATE_DIR = os.path.join(BASE_DIR, "templates")

# Path to your official CNIC template (front)
TEMPLATE_FRONT_PATH = os.path.join(TEMPLATE_DIR, "cnic_front.jpg")

# Region configuration (x, y, w, h) defined on the TEMPLATE image coordinate space
# !!! IMPORTANT: adjust these rectangles once to fit your real template !!!
REGIONS_CFG = {
    "photo_box":      (60, 120, 240, 300),
    "signature_box": (360, 360, 320, 120),
    "cnic_row":      (160, 80, 520, 80),
    "emblem_box":    (520, 160, 180, 180),
}

# Adjusted, more lenient thresholds for region similarity
REGION_THRESHOLDS = {
    "photo_box": 0.15,
    "signature_box": 0.15,
    "cnic_row": 0.20,
    "emblem_box": 0.20,
}
# Adjusted, more lenient global alignment thresholds
HOMOGRAPHY_MIN_INLIERS = 8
KEYPOINT_MIN = 100
FULLFRAME_EDGE_SIM_THRESH = 0.08

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(TEMPLATE_DIR, exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH
app.secret_key = "change-this-in-prod"

# -------------------- Helpers --------------------
def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTS

def save_upload(file_storage):
    filename = secure_filename(file_storage.filename)
    if not filename or not allowed_file(filename):
        raise ValueError("Unsupported file format. Only JPG/JPEG/PNG allowed.")
    ext = filename.rsplit(".", 1)[1].lower()
    new_name = f"{os.urandom(16).hex()}.{ext}"
    path = os.path.join(UPLOAD_DIR, new_name)
    file_storage.save(path)
    return path

# -------------------- Template & Alignment Verification --------------------
def _normalize_gray(img):
    if len(img.shape) == 3:
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        g = img
    # Apply Gaussian Blur to reduce noise
    g = cv2.GaussianBlur(g, (5, 5), 0)
    g = g.astype(np.float32)
    if g.max() > 0:
        g /= 255.0
    return g

def _edge_map(g):
    g8 = (g * 255).astype(np.uint8)
    # Use adaptive thresholding for better results on varying lighting
    edges = cv2.adaptiveThreshold(g8, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    # Then use Canny on the adaptively thresholded image
    edges = cv2.Canny(edges, 50, 150)
    return edges

def _cosine_sim(a, b):
    a = a.astype(np.float32).ravel()
    b = b.astype(np.float32).ravel()
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))

def _region_similarity(tpl_crop, warped_crop):
    tpl_g = _normalize_gray(tpl_crop)
    wrp_g = _normalize_gray(warped_crop)
    H, W = 120, 120
    tpl_g = cv2.resize(tpl_g, (W, H), interpolation=cv2.INTER_AREA)
    wrp_g = cv2.resize(wrp_g, (W, H), interpolation=cv2.INTER_AREA)
    e1 = _edge_map(tpl_g)
    e2 = _edge_map(wrp_g)
    edge_sim = _cosine_sim(e1, e2)
    h1 = cv2.calcHist([(tpl_g * 255).astype(np.uint8)], [0], None, [32], [0, 256]).astype(np.float32)
    h2 = cv2.calcHist([(wrp_g * 255).astype(np.uint8)], [0], None, [32], [0, 256]).astype(np.float32)
    hist_sim = _cosine_sim(h1.ravel(), h2.ravel())
    sim = 0.6 * edge_sim + 0.4 * hist_sim
    return float(max(0.0, min(1.0, sim)))

def template_alignment_check(user_img_path, template_path=TEMPLATE_FRONT_PATH):
    # Step 1: Read and validate images
    tpl_bgr = cv2.imread(template_path)
    usr_bgr = cv2.imread(user_img_path)
    if tpl_bgr is None or usr_bgr is None:
        return {"ok": False, "reason": "Failed to read images"}

    tpl_gray = cv2.cvtColor(tpl_bgr, cv2.COLOR_BGR2GRAY)
    usr_gray = cv2.cvtColor(usr_bgr, cv2.COLOR_BGR2GRAY)

    # Step 2: Keypoint and Homography Alignment
    orb = cv2.ORB_create(3000)
    kp1, des1 = orb.detectAndCompute(tpl_gray, None)
    kp2, des2 = orb.detectAndCompute(usr_gray, None)

    if des1 is None or des2 is None or len(kp1) < KEYPOINT_MIN or len(kp2) < KEYPOINT_MIN:
        return {"ok": False, "reason": "Not enough keypoints detected for alignment."}

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(des1, des2, k=2)
    good = [m for m, n in matches if m.distance < 0.75 * n.distance]

    if len(good) < 50:
        return {"ok": False, "reason": "Not enough good matches for homography."}

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
    
    inliers = int(mask.sum()) if mask is not None else 0

    if H is None or inliers < HOMOGRAPHY_MIN_INLIERS:
        return {"ok": False, "inliers": inliers, "reason": "Homography failed or too few inliers."}
    
    Ht, Wt = tpl_bgr.shape[:2]
    usr_warp = cv2.warpPerspective(usr_bgr, H, (Wt, Ht), flags=cv2.INTER_LINEAR)
    
    # Step 3: Full-frame Edge Similarity Check
    tpl_g = _normalize_gray(tpl_bgr)
    wrp_g = _normalize_gray(usr_warp)
    e1 = _edge_map(tpl_g)
    e2 = _edge_map(wrp_g)
    edge_sim = _cosine_sim(e1, e2)

    if edge_sim < FULLFRAME_EDGE_SIM_THRESH:
        return {"ok": False, "reason": "Low overall edge similarity.", "inliers": inliers, "edge_similarity": edge_sim}

    # Step 4: Region-specific Checks
    region_scores, region_pass = {}, {}
    for name, (x, y, w, h) in REGIONS_CFG.items():
        tpl_crop, wrp_crop = tpl_bgr[y:y+h, x:x+w], usr_warp[y:y+h, x:x+w]
        
        if tpl_crop.size == 0 or wrp_crop.size == 0:
            region_pass[name] = False
            continue
            
        s = _region_similarity(tpl_crop, wrp_crop)
        region_scores[name] = s
        region_pass[name] = (s >= REGION_THRESHOLDS.get(name, 0.3))
    
    if not all(region_pass.values()):
        failed_regions = [k for k, v in region_pass.items() if not v]
        return {"ok": False, "reason": f"Failed region check(s): {', '.join(failed_regions)}", "inliers": inliers, "edge_similarity": edge_sim, "region_scores": region_scores, "region_pass": region_pass}

    # All checks passed
    return {"ok": True, "reason": "Verification Passed", "inliers": inliers, "edge_similarity": edge_sim, "region_scores": region_scores, "region_pass": region_pass}

# -------------------- Routes & UI --------------------
HTML_FORM = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>CNIC Verification</title>
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <style>
    body { font-family: system-ui, Arial, sans-serif; background:#0b1020; color:#e7ecff; margin:0; }
    .wrap { max-width: 600px; margin: 40px auto; padding: 24px; background: #151b31; border-radius: 16px; box-shadow: 0 10px 30px rgba(0,0,0,.35);}
    h1 { margin-top:0; }
    label { display:block; margin: 14px 0 6px; font-weight:600;}
    input[type="file"] {
      width:100%; padding:12px; border-radius:12px; border:1px solid #2a355f; background:#0f1430; color:#e7ecff;
    }
    button {
      margin-top:18px; padding:12px 16px; background:#4c6fff; color:white; border:none; border-radius:12px; cursor:pointer; font-weight:700;
    }
    .note { color:#a7b1d6; font-size: 0.95rem; }
    .flash { background:#263058; padding:12px; border-radius:10px; margin:10px 0;}
    .result {
      margin-top:20px; padding:16px; border-radius:12px; text-align: center;
      font-size: 1.5rem; font-weight: bold;
    }
    .result.pass { background-color: #0c4d29; border: 1px solid #148b3b; color: #a1ffc4; }
    .result.fail { background-color: #5d1720; border: 1px solid #942b2b; color: #ffadad; }
    table { width:100%; border-collapse: collapse; margin-top: 12px; font-size: 0.95rem;}
    th, td { padding: 10px; border-bottom: 1px solid #2a355f; text-align:left; }
    th { background:#11183a; }
  </style>
</head>
<body>
  <div class="wrap">
    <h1>CNIC Document Check</h1>
    {% with messages = get_flashed_messages() %}
      {% if messages %}
        {% for m in messages %}
          <div class="flash">{{ m }}</div>
        {% endfor %}
      {% endif %}
    {% endwith %}
    <form action="{{ url_for('verify') }}" method="post" enctype="multipart/form-data">
      <div>
        <label>ID Card Image (JPG/PNG)</label>
        <input type="file" name="id_image" accept=".jpg,.jpeg,.png" required />
      </div>
      <button type="submit">Run Verification</button>
      <div class="note">Files &lt; 8 MB. We only accept JPG/JPEG/PNG.</div>
    </form>

    {% if result is not none %}
      <div class="result {{ 'pass' if result.ok else 'fail' }}">
        {% if result.ok %}
          ✅ ORIGINAL: NO FRAUD DETECTED
        {% else %}
          ❌ FRAUD DETECTED
        {% endif %}
      </div>
      <br/>
      {% if 'inliers' in result %}
      <div class="card">
        <h3>Debug Scores</h3>
        <table>
          <thead>
            <tr>
              <th>Check</th>
              <th>Score</th>
              <th>Threshold</th>
              <th>Result</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>RANSAC Inliers</td>
              <td>{{ result.inliers }}</td>
              <td>>= {{ HOMOGRAPHY_MIN_INLIERS }}</td>
              <td>{{ 'Pass' if result.inliers >= HOMOGRAPHY_MIN_INLIERS else 'Fail' }}</td>
            </tr>
            <tr>
              <td>Edge Similarity</td>
              <td>{{ '%.2f' % result.edge_similarity }}</td>
              <td>>= {{ FULLFRAME_EDGE_SIM_THRESH }}</td>
              <td>{{ 'Pass' if result.edge_similarity >= FULLFRAME_EDGE_SIM_THRESH else 'Fail' }}</td>
            </tr>
            {% if 'region_scores' in result %}
            {% for name, score in result.region_scores.items() %}
            <tr>
              <td>{{ name }}</td>
              <td>{{ '%.2f' % score }}</td>
              <td>>= {{ REGION_THRESHOLDS.get(name) }}</td>
              <td>{{ 'Pass' if result.get('region_pass', {}).get(name) else 'Fail' }}</td>
            </tr>
            {% endfor %}
            {% endif %}
          </tbody>
        </table>
      </div>
      {% endif %}
    {% endif %}
  </div>
</body>
</html>
"""

@app.get("/")
def index():
    return render_template_string(HTML_FORM, result=None)

@app.post("/verify")
def verify():
    try:
        id_file = request.files.get("id_image")
        if id_file is None:
            flash("Please attach an image.")
            return redirect(url_for("index"))
        id_path = save_upload(id_file)
        result = template_alignment_check(id_path)
        try:
            os.remove(id_path)
        except Exception:
            pass
        return render_template_string(
            HTML_FORM,
            result=result,
            HOMOGRAPHY_MIN_INLIERS=HOMOGRAPHY_MIN_INLIERS,
            FULLFRAME_EDGE_SIM_THRESH=FULLFRAME_EDGE_SIM_THRESH,
            REGION_THRESHOLDS=REGION_THRESHOLDS
        )
    except ValueError as ve:
        flash(str(ve))
        return redirect(url_for("index"))
    except Exception as e:
        flash(f"An unexpected error occurred: {e}. Please ensure the template file exists and is readable.")
        return redirect(url_for("index"))

if __name__ == "__main__":
    app.run(debug=True)