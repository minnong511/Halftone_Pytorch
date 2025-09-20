# ============================================================
# [핵심 로직 주석]
# 입력(웹캠 프레임, BGR) → Grayscale → 격자(Cell)로 분할 →
# 각 셀의 평균 밝기 I(0~255) → 반경 r = map(I; r_min, r_max, gamma) →
# 빈 캔버스에 셀 중심마다 원(도트) 채우기 → Halftone 프레임 디스플레이
# 어두울수록 큰 점, 밝을수록 작은 점.
# ============================================================

import atexit
import os
import socket
import threading
import time
from pathlib import Path

import cv2 as cv
import numpy as np
from flask import Flask, Response, jsonify, render_template, request

# ---- PyTorch(옵션) : GPU 가속용 ----
try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
    DEVICE = torch.device(
        "cuda" if torch.cuda.is_available() else
        ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")
    )
except Exception as _e:
    TORCH_AVAILABLE = False
    DEVICE = None

# ---------------------------
# 파라미터(합리적 기본값)
# ---------------------------
CAP_INDEX   = 1          # 내장 카메라 기본 인덱스
FRAME_W     = 720
FRAME_H     = 480
CELL        = 36         # 셀 크기(픽셀)
R_MIN       = 0          # 최소 반경
R_MAX       = CELL // 2  # 최대 반경(셀의 절반 이하 권장)
GAMMA       = 1.0        # 감마(>1이면 밝은 영역 점 더 작아짐)
BLUR_KSIZE  = 3          # 0이면 블러 없음(예: 3,5로 설정하면 약간 매끈해짐)
USE_CAMERA  = True       # 카메라 사용. False면 테스트용 합성/이미지 사용
SAVE_DIR    = Path("halftone_snaps")
SAVE_DIR.mkdir(exist_ok=True)

PROC_SCALE = 0.5
WEB_HOST = os.getenv("HALFTONE_HOST", "0.0.0.0")


def _parse_port(value: str | None, default: int) -> int:
    if not value:
        return default
    try:
        port = int(value)
        if 1 <= port <= 65535:
            return port
    except (TypeError, ValueError):
        pass
    return default


def _find_available_port(preferred: int, host: str, attempts: int = 20) -> tuple[int, bool]:
    """Return a port and whether it differs from preferred."""
    host_to_check = "0.0.0.0" if host in ("", "0.0.0.0") else host
    port = preferred
    for _ in range(max(1, attempts)):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind((host_to_check, port))
            except OSError:
                port += 1
                if port > 65535:
                    port = 1024
                continue
            else:
                return port, port != preferred
    return preferred, False


WEB_PORT_INPUT = _parse_port(os.getenv("HALFTONE_PORT"), 5050)
WEB_PORT, WEB_PORT_CHANGED = _find_available_port(WEB_PORT_INPUT, WEB_HOST)

ASCII_LOGO = r"""

 _   _         _   __  _                         _____   ___  ___  ___
| | | |       | | / _|| |                       /  __ \ / _ \ |  \/  |
| |_| |  __ _ | || |_ | |_   ___   _ __    ___  | /  \// /_\ \| .  . |
|  _  | / _` || ||  _|| __| / _ \ | '_ \  / _ \ | |    |  _  || |\/| |
| | | || (_| || || |  | |_ | (_) || | | ||  __/ | \__/\| | | || |  | |
\_| |_/ \__,_||_||_|   \__| \___/ |_| |_| \___|  \____/\_| |_/\_|  |_/
                                                                      
                                                                
"""                                                               
                                                                 



def _gather_program_settings() -> list[str]:
    blur_desc = f"{BLUR_KSIZE}" if BLUR_KSIZE and BLUR_KSIZE > 1 else "off"
    return [
        f"Camera Index    : {CAP_INDEX}",
        f"Frame Size      : {FRAME_W}x{FRAME_H}",
        f"Cell Size       : {CELL}",
        f"Radius Range    : {R_MIN}~{R_MAX}",
        f"Gamma           : {GAMMA}",
        f"Blur Kernel     : {blur_desc}",
        f"Processing Scale: {PROC_SCALE}",
        f"Camera Mode     : {'enabled' if USE_CAMERA else 'static demo'}",
    ]


def _gather_acceleration_info() -> list[tuple[str, str]]:
    lines = []
    if TORCH_AVAILABLE and DEVICE is not None:
        device_name = str(DEVICE)
        device_label = "GPU" if device_name in ("cuda", "mps") else "CPU"
        lines.append(("PyTorch", f"{device_name} · {device_label}"))
    else:
        lines.append(("PyTorch", "not available (CPU fallback)"))

    cv_cuda_available = False
    cuda_desc = "not available"
    if hasattr(cv, "cuda"):
        try:
            count = cv.cuda.getCudaEnabledDeviceCount()
            if count and count > 0:
                cv_cuda_available = True
                cuda_desc = "cuda"
            elif count == 0:
                cuda_desc = "detected, but no enabled devices"
        except Exception:
            cuda_desc = "error while probing"
    lines.append(("OpenCV CUDA", cuda_desc))

    opencl_desc = "not available"
    try:
        if cv.ocl.haveOpenCL():
            opencl_desc = "enabled" if cv.ocl.useOpenCL() else "available (disabled)"
    except Exception:
        opencl_desc = "error while probing"
    lines.append(("OpenCL", opencl_desc))

    return lines


def print_startup_banner():
    print(ASCII_LOGO.strip("\n"))
    print("=" * 78)
    print("PROGRAM SETTINGS")
    print("-" * 78)
    for line in _gather_program_settings():
        print(line)
    print("=" * 78)
    print("HARDWARE ACCELERATION")
    print("-" * 78)
    for label, status in _gather_acceleration_info():
        print(f"{label:<14} | {status}")
    print("=" * 78)
    print()

def _gaussian_kernel2d(ksize: int, sigma: float | None = None):
    """Create a 2D Gaussian kernel as torch.Tensor shape (1,1,k,k)."""
    if ksize <= 1:
        k = torch.ones((1, 1, 1, 1), dtype=torch.float32)
        return k
    if sigma is None:
        sigma = 0.3 * ((ksize - 1) * 0.5 - 1) + 0.8  # OpenCV의 기본 추정식과 유사
    ax = torch.arange(ksize, dtype=torch.float32) - (ksize - 1) / 2.0
    xx, yy = torch.meshgrid(ax, ax, indexing="xy")
    ker = torch.exp(-(xx * xx + yy * yy) / (2 * sigma * sigma))
    ker = ker / ker.sum()
    return ker.view(1, 1, ksize, ksize)

def halftone_gray_gpu(gray_np: np.ndarray, cell: int, r_min: int, r_max: int, gamma: float, blur_ksize: int, device):
    """
    GPU 텐서 연산 기반 halftone 렌더링.
    입력: gray_np (H,W) uint8
    반환: BGR uint8 이미지
    """
    H, W = gray_np.shape
    # (1,1,H,W) float32 [0,1] 텐서
    x = torch.from_numpy(gray_np).to(device=device, dtype=torch.float32).div_(255.0).unsqueeze(0).unsqueeze(0)

    # (옵션) Gaussian Blur: conv2d로 수행 (same padding)
    if blur_ksize and blur_ksize > 1:
        k = blur_ksize if blur_ksize % 2 == 1 else (blur_ksize + 1)
        ker = _gaussian_kernel2d(k).to(device=device)
        pad = k // 2
        x = F.conv2d(x, ker, padding=pad)

    # 감마 보정
    if gamma != 1.0:
        x = x.clamp_(0.0, 1.0).pow_(gamma)

    # 셀 평균 맵 (Hc,Wc) = (H//cell, W//cell)
    # stride=kernel_size=cell 로 겹치지 않는 평균
    mean = F.avg_pool2d(x, kernel_size=cell, stride=cell)

    # 반경 맵(셀 해상도)
    r = r_max * (1.0 - mean)  # 밝을수록 작게
    r = r.clamp(min=float(r_min), max=float(r_max))

    # (H,W)로 최근접 업샘플
    r_full = F.interpolate(r, size=(H, W), mode="nearest")  # (1,1,H,W)

    # 각 픽셀의 셀 로컬 좌표 (셀 중심 기준)
    # modulo 좌표를 [-cell//2, +cell//2] 범위로 이동
    ix = torch.arange(W, device=device, dtype=torch.float32).view(1, 1, 1, W)
    iy = torch.arange(H, device=device, dtype=torch.float32).view(1, 1, H, 1)
    cx = (ix % cell) - (cell // 2)
    cy = (iy % cell) - (cell // 2)

    # 사각형 판정: max(|dx|, |dy|) <= r  (체비셰프 노름 → 정사각형 도트)
    abs_cx = cx.abs()
    abs_cy = cy.abs()
    cheb = torch.maximum(abs_cx, abs_cy)
    mask = (cheb <= r_full).to(torch.uint8)  # (1,1,H,W)

    # 3채널 BGR로 확장 - Matrix 스타일 네온 그린 (B=0, G=255, R=0)
    g = mask.mul(255)  # green channel
    z = torch.zeros_like(g)
    out = torch.cat([z, g, z], dim=1).squeeze(0).permute(1, 2, 0).contiguous()

    # CPU로 가져와 numpy 변환
    return out.to("cpu").numpy()

def halftone_gray(gray, cell=CELL, r_min=R_MIN, r_max=R_MAX, gamma=GAMMA, blur_ksize=0):
    """
    gray: (H,W) uint8
    반환: halftone BGR 이미지
    """
    H, W = gray.shape
    # 필요한 경우 약간의 블러로 노이즈 완화
    if blur_ksize and blur_ksize > 1:
        g = cv.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)
    else:
        g = gray

    # 감마 보정: 입력 밝기 I(0~255)를 [0,1]로 정규화 → pow(gamma) → 다시 0~255
    # (시각적 콘트라스트를 제어)
    g_norm = g.astype(np.float32) / 255.0
    if gamma != 1.0:
        g_norm = np.clip(g_norm, 0, 1) ** gamma
    g_proc = (g_norm * 255.0).astype(np.uint8)

    # 출력 캔버스(검정 배경, BGR)
    out = np.zeros((H, W, 3), dtype=np.uint8)

    # 셀 단위로 평균을 구해 반경으로 매핑해서 원을 그린다
    for y in range(0, H, cell):
        y2 = min(y + cell, H)
        cy = (y + y2) // 2  # 셀 중심 y

        for x in range(0, W, cell):
            x2 = min(x + cell, W)
            cx = (x + x2) // 2  # 셀 중심 x

            patch = g_proc[y:y2, x:x2]
            I = float(patch.mean())  # 평균 밝기(0~255)

            # 밝기 → 반경 매핑: 밝을수록 반경↓, 어두울수록 반경↑
            r = r_max * (1.0 - I / 255.0)  # 위에서 감마 적용했으므로 선형
            r = max(r_min, min(r, r_max))

            if r > 0.5:
                side = int(round(2 * r))  # 정사각형 한 변 길이
                if side > 0:
                    half = side // 2
                    x1 = max(0, cx - half)
                    y1 = max(0, cy - half)
                    x2 = min(W - 1, cx + half)
                    y2 = min(H - 1, cy + half)
                    cv.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), thickness=-1)  # 네온 그린(BGR)

    return out

def frame_to_gray(frame_bgr):
    """BGR → Gray, 리사이즈 포함"""
    f = cv.resize(frame_bgr, (FRAME_W, FRAME_H), interpolation=cv.INTER_AREA)
    return cv.cvtColor(f, cv.COLOR_BGR2GRAY)


def overlay_params(img, cell, rmin, rmax, gamma, blur_ksize, fps=None):
    """현재 파라미터를 화면 좌상단에 텍스트로 표시."""
    htxt = [
        f"CELL={cell}  R_MIN={rmin}  R_MAX={rmax}",
        f"GAMMA={gamma:.2f}  BLUR_KSIZE={blur_ksize if blur_ksize>0 else 0}",
    ]
    if fps is not None:
        htxt.append(f"FPS={fps:.1f}")
    y = 20
    for line in htxt:
        cv.putText(img, line, (10, y), cv.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv.LINE_AA)
        y += 20


class ParameterState:
    def __init__(self):
        self.lock = threading.Lock()
        self.cell = CELL
        self.r_min = R_MIN
        self.r_max = min(R_MAX, CELL // 2)
        self.gamma = GAMMA
        self.blur_level = 0 if (BLUR_KSIZE is None or BLUR_KSIZE < 2) else (BLUR_KSIZE - 1) // 2

    def _normalize_locked(self):
        self.cell = int(max(4, min(64, self.cell)))
        self.r_max = int(max(1, min(self.r_max, self.cell // 2, 32)))
        self.r_min = int(max(0, min(self.r_min, self.r_max)))
        self.gamma = float(max(0.2, min(3.0, self.gamma)))
        self.blur_level = int(max(0, min(7, self.blur_level)))

    def snapshot(self) -> dict:
        with self.lock:
            self._normalize_locked()
            blur_ksize = 0 if self.blur_level <= 0 else 2 * self.blur_level + 1
            return {
                "cell": self.cell,
                "r_min": self.r_min,
                "r_max": self.r_max,
                "gamma": self.gamma,
                "blur_level": self.blur_level,
                "blur_ksize": blur_ksize,
            }

    def update(self, payload: dict) -> list[str]:
        if not isinstance(payload, dict):
            return []

        changed = set()
        with self.lock:
            original = {
                "cell": self.cell,
                "r_min": self.r_min,
                "r_max": self.r_max,
                "gamma": self.gamma,
                "blur_level": self.blur_level,
            }

            if "cell" in payload:
                try:
                    new_val = int(payload["cell"])
                except (TypeError, ValueError):
                    new_val = original["cell"]
                if new_val != self.cell:
                    self.cell = new_val
                    changed.add("cell")

            if "r_max" in payload:
                try:
                    new_val = int(payload["r_max"])
                except (TypeError, ValueError):
                    new_val = original["r_max"]
                if new_val != self.r_max:
                    self.r_max = new_val
                    changed.add("r_max")

            if "r_min" in payload:
                try:
                    new_val = int(payload["r_min"])
                except (TypeError, ValueError):
                    new_val = original["r_min"]
                if new_val != self.r_min:
                    self.r_min = new_val
                    changed.add("r_min")

            if "gamma" in payload:
                try:
                    new_val = float(payload["gamma"])
                except (TypeError, ValueError):
                    new_val = original["gamma"]
                if new_val != self.gamma:
                    self.gamma = new_val
                    changed.add("gamma")

            if "blur_level" in payload:
                try:
                    new_val = int(payload["blur_level"])
                except (TypeError, ValueError):
                    new_val = original["blur_level"]
                if new_val != self.blur_level:
                    self.blur_level = new_val
                    changed.add("blur_level")

            self._normalize_locked()

            normalized = {
                "cell": self.cell,
                "r_min": self.r_min,
                "r_max": self.r_max,
                "gamma": self.gamma,
                "blur_level": self.blur_level,
            }

        for key, value in normalized.items():
            if original[key] != value:
                changed.add(key)

        return sorted(changed)


class FrameSource:
    def __init__(self, use_camera: bool):
        self.lock = threading.Lock()
        self.use_camera = use_camera
        self.cap = None
        self._warned_camera_failure = False
        self.static_gray = self._build_static_pattern()
        if use_camera:
            cap = cv.VideoCapture(CAP_INDEX)
            cap.set(cv.CAP_PROP_FRAME_WIDTH, FRAME_W)
            cap.set(cv.CAP_PROP_FRAME_HEIGHT, FRAME_H)
            if cap.isOpened():
                self.cap = cap
            else:
                print("[경고] 카메라를 열 수 없습니다. 정적 패턴으로 전환합니다.")
                self.use_camera = False
        if not self.use_camera or self.cap is None:
            self.use_camera = False

    @staticmethod
    def _build_static_pattern():
        H, W = FRAME_H, FRAME_W
        grad = np.tile(np.linspace(0, 255, W, dtype=np.uint8), (H, 1))
        y0, y1 = H // 3, 2 * H // 3
        x0, x1 = W // 3, 2 * W // 3
        grad[y0:y1, x0:x1] = 0
        return grad

    def get_gray(self) -> np.ndarray:
        if self.cap is not None:
            with self.lock:
                ok, frame = self.cap.read()
            if ok:
                return frame_to_gray(frame)
            if not self._warned_camera_failure:
                print("[에러] 카메라 프레임 캡처 실패. 정적 패턴으로 전환합니다.")
                self._warned_camera_failure = True
            self.release()
        return self.static_gray.copy()

    def release(self):
        if self.cap is not None:
            with self.lock:
                self.cap.release()
            self.cap = None


class SnapshotBuffer:
    def __init__(self):
        self.lock = threading.Lock()
        self.frame: np.ndarray | None = None

    def update(self, frame: np.ndarray):
        with self.lock:
            self.frame = frame.copy()

    def save(self):
        with self.lock:
            if self.frame is None:
                return False, "no_frame"
            ts = time.strftime("%Y%m%d_%H%M%S")
            path = SAVE_DIR / f"halftone_{ts}.png"
            ok = cv.imwrite(str(path), self.frame)
            if not ok:
                return False, "write_failed"
            return True, str(path)


def render_halftone_frame(gray: np.ndarray, params: dict) -> np.ndarray:
    cell = params["cell"]
    rmin = params["r_min"]
    rmax = params["r_max"]
    gamma = params["gamma"]
    blur_ksize = params["blur_ksize"]
    use_gpu = TORCH_AVAILABLE and DEVICE is not None and str(DEVICE) in ("cuda", "mps")

    if PROC_SCALE < 1.0:
        W2 = max(64, int(gray.shape[1] * PROC_SCALE))
        H2 = max(48, int(gray.shape[0] * PROC_SCALE))
        gray_s = cv.resize(gray, (W2, H2), interpolation=cv.INTER_AREA)
        cell_s = max(2, int(round(cell * PROC_SCALE)))
        rmax_s = max(1, int(round(rmax * PROC_SCALE)))
        rmin_s = max(0, int(round(rmin * PROC_SCALE)))
        rmax_s = min(rmax_s, cell_s // 2, 32)
        rmin_s = min(rmin_s, rmax_s)
        if blur_ksize and blur_ksize > 1:
            k = max(1, int(round(blur_ksize * PROC_SCALE)))
            blur_s = k if (k % 2 == 1) else (k + 1)
        else:
            blur_s = 0
        if use_gpu:
            ht_small = halftone_gray_gpu(gray_s, cell_s, rmin_s, rmax_s, gamma, blur_s, DEVICE)
        else:
            ht_small = halftone_gray(gray_s, cell=cell_s, r_min=rmin_s, r_max=rmax_s, gamma=gamma, blur_ksize=blur_s)
        frame = cv.resize(ht_small, (FRAME_W, FRAME_H), interpolation=cv.INTER_NEAREST)
    else:
        if use_gpu:
            frame = halftone_gray_gpu(gray, cell, rmin, rmax, gamma, blur_ksize, DEVICE)
        else:
            frame = halftone_gray(gray, cell=cell, r_min=rmin, r_max=rmax, gamma=gamma, blur_ksize=blur_ksize)

    return frame


PARAM_STATE = ParameterState()
FRAME_SOURCE = FrameSource(USE_CAMERA)
SNAPSHOT_BUFFER = SnapshotBuffer()
atexit.register(FRAME_SOURCE.release)


def stream_frames():
    fps = 0.0
    last_t = time.time()
    while True:
        params = PARAM_STATE.snapshot()
        gray = FRAME_SOURCE.get_gray()
        if gray is None:
            time.sleep(0.01)
            continue
        frame = render_halftone_frame(gray, params)
        now = time.time()
        dt = now - last_t
        last_t = now
        if dt > 0:
            fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps > 0 else (1.0 / dt)
        overlay_params(frame, params["cell"], params["r_min"], params["r_max"], params["gamma"], params["blur_ksize"], fps=fps)
        SNAPSHOT_BUFFER.update(frame)
        ok, buffer = cv.imencode(".jpg", frame, [cv.IMWRITE_JPEG_QUALITY, 90])
        if not ok:
            continue
        yield b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"


app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/stream")
def stream():
    return Response(stream_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/settings", methods=["GET", "POST"])
def api_settings():
    if request.method == "GET":
        params = PARAM_STATE.snapshot()
        hardware = [{"label": label, "status": status} for label, status in _gather_acceleration_info()]
        return jsonify({
            "params": params,
            "hardware": hardware,
            "program_settings": _gather_program_settings(),
        })

    payload = request.get_json(silent=True) or {}
    changed = PARAM_STATE.update(payload)
    params = PARAM_STATE.snapshot()
    return jsonify({
        "params": params,
        "changed": changed,
    })


@app.route("/api/snapshot", methods=["POST"])
def api_snapshot():
    ok, detail = SNAPSHOT_BUFFER.save()
    if ok:
        return jsonify({"success": True, "path": detail})
    return jsonify({"success": False, "error": detail}), 400


if __name__ == "__main__":
    print_startup_banner()
    display_host = "127.0.0.1" if WEB_HOST in ("", "0.0.0.0") else WEB_HOST
    if WEB_PORT_CHANGED:
        print(f"[INFO] 요청한 포트가 사용 중이라 {WEB_PORT}로 변경되었습니다.")
    print(f"[INFO] 웹 인터페이스: http://{display_host}:{WEB_PORT}/")
    app.run(host=WEB_HOST, port=WEB_PORT, threaded=True)
