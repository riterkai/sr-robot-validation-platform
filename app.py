import csv
import json
import os
from flask import Flask, render_template, request, send_from_directory, jsonify, Response
import mimetypes

app = Flask(__name__)

_sync_cache = {}  # { (video_path, csv_path): result }

def compute_sync_offset():
    try:
        import cv2
        import numpy as np
    except ImportError as e:
        return {"error": str(e)}

    video_path = resolve_video('webcam')
    csv_path   = os.path.join(UPLOAD_FOLDER, "log.csv")
    if not video_path or not os.path.exists(csv_path):
        return {"error": "missing files"}

    cache_key = (video_path, os.path.getmtime(video_path),
                 csv_path,   os.path.getmtime(csv_path))
    if cache_key in _sync_cache:
        return _sync_cache[cache_key]

    # ── 1. 從影片擷取 frame difference 作為「運動強度」信號 ──────────
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    sample_every = max(1, int(fps / 5))   # 以 ~5fps 取樣

    video_times, video_motion = [], []
    ret, prev = cap.read()
    if not ret:
        cap.release()
        return {"error": "cannot read video"}

    def preprocess(f):
        small = cv2.resize(f, (320, 180))
        gray  = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        gray  = cv2.GaussianBlur(gray, (5, 5), 0)
        return gray[90:, :]   # 只取下半部（機器人區域）

    prev_gray = preprocess(prev)
    idx = 1
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % sample_every == 0:
            gray = preprocess(frame)
            diff = float(np.mean(cv2.absdiff(gray, prev_gray).astype(np.float32)))
            video_times.append(idx / fps)
            video_motion.append(diff)
            prev_gray = gray
        idx += 1
    cap.release()

    if len(video_times) < 20:
        return {"error": "insufficient video data"}

    # ── 2. 讀取 CSV：合併 sqrt(v² + w²) 作為「總運動量」信號 ──────────
    csv_times, csv_motion = [], []
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            t_key = next((k for k in ["elapsed_time","time","t","timestamp"] if k in row), None)
            w_key = next((k for k in ["real_w","w","cmd_w","angular"]          if k in row), None)
            v_key = next((k for k in ["real_v","v","cmd_v","linear"]           if k in row), None)
            try:
                if not t_key:
                    continue
                t_val = float(row[t_key])
                w_val = float(row[w_key]) if w_key else 0.0
                v_val = float(row[v_key]) if v_key else 0.0
                csv_times.append(t_val)
                csv_motion.append((v_val**2 + w_val**2) ** 0.5)
            except (ValueError, TypeError):
                continue

    if len(csv_times) < 20:
        return {"error": "insufficient csv data"}

    # ── 3. 平滑 + 偵測「運動開始時間點」────────────────────────────────
    def smooth(arr, win=5):
        return np.convolve(np.array(arr), np.ones(win)/win, 'same')

    def first_motion_time(times, signal, baseline_secs=2.0, multiplier=2.5, sustained=4):
        """
        先用前 baseline_secs 秒估計靜止基準值，
        再找到第一次連續 sustained 個樣本超過 baseline × multiplier 的時間。
        """
        s     = smooth(signal)
        tarr  = np.array(times)
        base_mask = tarr <= baseline_secs
        baseline  = float(np.mean(s[base_mask])) if base_mask.any() else float(s[:5].mean())
        thresh    = baseline * multiplier
        above     = s > thresh
        for i in range(len(above) - sustained):
            if above[i:i+sustained].all():
                return float(times[i])
        return None

    t_csv_start   = first_motion_time(csv_times,   csv_motion,   baseline_secs=1.0)
    t_video_start = first_motion_time(video_times, video_motion, baseline_secs=0.5)

    if t_csv_start is None or t_video_start is None:
        return {"error": "cannot detect motion start"}

    motion_offset = round(t_csv_start - t_video_start, 2)

    # ── 4. Cross-correlation 交叉驗證（±30s 範圍）──────────────────────
    dt     = 0.2
    t_max  = min(video_times[-1], csv_times[-1])
    t_grid = np.arange(0, t_max, dt)

    v_sig = smooth(np.interp(t_grid, video_times, video_motion))
    w_sig = smooth(np.interp(t_grid, csv_times,  csv_motion))

    def norm(s):
        s = s - s.mean()
        std = s.std()
        return s / std if std > 1e-8 else s

    corr = np.correlate(norm(v_sig), norm(w_sig), mode="full")
    lags = (np.arange(len(corr)) - (len(w_sig) - 1)) * dt
    mask = np.abs(lags) <= 30
    best_idx  = np.argmax(corr * mask)
    best_lag  = float(lags[best_idx])
    corr_offset    = round(best_lag, 2)   # 不取負號
    confidence = float(corr[best_idx] / len(t_grid))

    # motion-start 超出合理範圍 → 改用 cross-correlation
    if abs(motion_offset) > 5.0:
        final_offset = corr_offset
        method = "cross-correlation (motion-start unreliable)"
    elif abs(corr_offset - motion_offset) < 0.5:
        final_offset = corr_offset
        method = "cross-correlation"
    else:
        final_offset = motion_offset
        method = "motion-start"

    result = {
        "offset":     final_offset,
        "confidence": round(confidence, 3),
        "method":     method,
        "motion_start_offset": motion_offset,
        "corr_offset": corr_offset,
    }
    _sync_cache[cache_key] = result
    return result

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "static")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/static/<path:filename>")
def static_files(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


@app.route("/api/pose")
def get_pose():
    """座標軌跡 CSV（two_point_loop_run001_...csv）"""
    # 先找 static/ 裡有沒有放，否則掃 Documents/Ros/
    candidates = [
        os.path.join(UPLOAD_FOLDER, "pose.csv"),
        os.path.join(UPLOAD_FOLDER, "pos.csv"),
    ]
    import glob, pathlib
    candidates += sorted(glob.glob(str(pathlib.Path.home() / "Documents/Ros/two_point_loop_run*.csv")))

    csv_path = next((p for p in candidates if os.path.exists(p)), None)
    if not csv_path:
        return jsonify([])

    rows = []
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                x, y = float(row.get("x", 0)), float(row.get("y", 0))
                tx, ty = float(row.get("target_x", 0)), float(row.get("target_y", 0))
                dist_raw = row.get("distance", "")
                distance = float(dist_raw) if dist_raw != "" else ((x - tx)**2 + (y - ty)**2)**0.5
                rows.append({
                    "time":       float(row.get("elapsed_time") or row.get("time", 0)),
                    "x":          x,
                    "y":          y,
                    "yaw":        float(row.get("current_yaw_deg", 0)),
                    "target_x":   tx,
                    "target_y":   ty,
                    "distance":   round(distance, 3),
                    "waypoint":   row.get("waypoint", ""),
                    "loop":       int(float(row.get("loop", 0))),
                    "strategy":   row.get("strategy", ""),
                })
            except (ValueError, TypeError):
                continue
    return jsonify(rows)


@app.route("/api/csv")
def get_csv():
    csv_path = os.path.join(UPLOAD_FOLDER, "log.csv")
    if not os.path.exists(csv_path):
        import math
        rows = []
        for i in range(200):
            t = round(i * 0.05, 3)
            rows.append({
                "time": t,
                "pwm_left":    round(0.08 + 0.03 * math.sin(t * 1.2), 4),
                "pwm_right":   round(0.10 + 0.03 * math.sin(t * 1.2 + 0.3), 4),
                "real_v":      round(0.0 + 0.3 * math.sin(t * 0.8), 4),
                "real_w":      round(0.0 + 0.2 * math.sin(t * 0.8 + 1.0), 4),
                "left_speed":  round(0.3 * math.sin(t * 0.8 + 0.2), 4),
                "right_speed": round(0.3 * math.sin(t * 0.8 - 0.2), 4),
            })
        return jsonify(rows)

    def pick(row, *names):
        for n in names:
            if n in row:
                return n
        return None

    rows = []
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            t_key  = pick(row, "elapsed_time", "time", "t", "timestamp")
            pl_key  = pick(row, "physical_left_pwm_signed",  "pwm_left",  "pvm_left",  "left_pwm")
            pr_key  = pick(row, "physical_right_pwm_signed", "pwm_right", "pvm_right", "right_pwm")
            v_key   = pick(row, "real_v", "v", "cmd_v", "linear")
            w_key   = pick(row, "real_w", "w", "cmd_w", "angular")
            lspd_key = pick(row, "physical_left_speed",  "left_speed")
            rspd_key = pick(row, "physical_right_speed", "right_speed")
            try:
                rows.append({
                    "time":        float(row[t_key])     if t_key     else len(rows) * 0.05,
                    "pwm_left":    float(row[pl_key])    if pl_key    else 0,
                    "pwm_right":   float(row[pr_key])    if pr_key    else 0,
                    "real_v":      float(row[v_key])     if v_key     else 0,
                    "real_w":      float(row[w_key])     if w_key     else 0,
                    "left_speed":  float(row[lspd_key])  if lspd_key  else 0,
                    "right_speed": float(row[rspd_key])  if rspd_key  else 0,
                })
            except (ValueError, TypeError):
                continue
    return jsonify(rows)


@app.route("/api/sync-offset")
def sync_offset():
    return jsonify({"error": "disabled"}), 503


def resolve_video(basename):
    """Return actual path, trying .mp4 then .mov variants."""
    stem = os.path.splitext(basename)[0]
    for ext in (".mp4", ".mov"):
        p = os.path.join(UPLOAD_FOLDER, stem + ext)
        if os.path.exists(p):
            return p
    return None


@app.route("/video/<filename>")
def video(filename):
    path = resolve_video(filename)
    if not path:
        return "", 404
    filename = os.path.basename(path)

    file_size = os.path.getsize(path)
    range_header = request.headers.get("Range")
    mime = mimetypes.guess_type(filename)[0] or "video/mp4"

    if range_header:
        byte_start, byte_end = 0, file_size - 1
        parts = range_header.replace("bytes=", "").split("-")
        byte_start = int(parts[0])
        if parts[1]:
            byte_end = int(parts[1])
        length = byte_end - byte_start + 1

        def generate():
            with open(path, "rb") as f:
                f.seek(byte_start)
                remaining = length
                while remaining:
                    chunk = f.read(min(65536, remaining))
                    if not chunk:
                        break
                    remaining -= len(chunk)
                    yield chunk

        headers = {
            "Content-Range": f"bytes {byte_start}-{byte_end}/{file_size}",
            "Accept-Ranges": "bytes",
            "Content-Length": length,
            "Content-Type": mime,
        }
        return Response(generate(), status=206, headers=headers)

    return send_from_directory(UPLOAD_FOLDER, filename)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    debug = os.environ.get("FLASK_DEBUG", "").lower() in {"1", "true", "yes"}
    print(f"Place webcam.mp4, aruco.mp4, and log.csv inside the 'static/' folder.")
    print(f"Then open http://127.0.0.1:{port}")
    app.run(debug=debug, port=port)
