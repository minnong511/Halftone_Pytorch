# ============================================================
# [핵심 로직 주석]
# 입력(웹캠 프레임, BGR) → Grayscale → 격자(Cell)로 분할 →
# 각 셀의 평균 밝기 I(0~255) → 반경 r = map(I; r_min, r_max, gamma) →
# 빈 캔버스에 셀 중심마다 원(도트) 채우기 → Halftone 프레임 디스플레이
# 어두울수록 큰 점, 밝을수록 작은 점.
# ============================================================

import cv2 as cv
import numpy as np
import time
from pathlib import Path

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
SAVE_DIR    = Path("./halftone_snaps")
SAVE_DIR.mkdir(exist_ok=True)

# ---------------------------
# 통합 UI 레이아웃(1개 창): 좌측 1/3 = GUI, 우측 2/3 = 영상
# ---------------------------
UI_W_RATIO  = 1/3.0
VID_W_RATIO = 1 - UI_W_RATIO
CANVAS_W    = FRAME_W
CANVAS_H    = FRAME_H
VID_W       = int(CANVAS_W * VID_W_RATIO)
UI_W        = CANVAS_W - VID_W

PROC_SCALE = 0.5

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

# ---------------------------
# 커스텀 슬라이더(캔버스 내 그리기 + 마우스 이벤트)
# ---------------------------
class Slider:
    def __init__(self, name, vmin, vmax, vinit, x, y, w, h, step=1.0, as_int=False):
        self.name = name
        self.vmin = vmin
        self.vmax = vmax
        self.value = np.clip(vinit, vmin, vmax)
        self.x, self.y = x, y
        self.w, self.h = w, h
        self.step = step
        self.as_int = as_int
        self.active = False  # 드래그 중 여부

    def _value_to_px(self, v):
        t = (v - self.vmin) / (self.vmax - self.vmin) if self.vmax > self.vmin else 0.0
        return int(self.x + 10 + t * (self.w - 20))

    def _px_to_value(self, px):
        t = (px - (self.x + 10)) / max(1, (self.w - 20))
        v = self.vmin + np.clip(t, 0, 1) * (self.vmax - self.vmin)
        if self.as_int:
            v = round(v / self.step) * self.step
        return np.clip(v, self.vmin, self.vmax)

    def handle_mouse(self, event, mx, my):
        if event == cv.EVENT_LBUTTONDOWN:
            if self.y <= my <= self.y + self.h and self.x <= mx <= self.x + self.w:
                self.active = True
                self.value = self._px_to_value(mx)
        elif event == cv.EVENT_MOUSEMOVE and self.active:
            self.value = self._px_to_value(mx)
        elif event == cv.EVENT_LBUTTONUP:
            self.active = False

    def read(self):
        return int(self.value) if self.as_int else float(self.value)

    def draw(self, img):
        # 바탕
        cv.rectangle(img, (self.x, self.y), (self.x + self.w, self.y + self.h), (35, 35, 35), -1)
        # 라벨
        label = f"{self.name}: {int(self.value) if self.as_int else self.value:.2f}" if not self.as_int else f"{self.name}: {int(self.value)}"
        cv.putText(img, label, (self.x + 8, self.y + 18), cv.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv.LINE_AA)
        # 트랙
        cy = self.y + self.h // 2 + 8
        cv.line(img, (self.x + 10, cy), (self.x + self.w - 10, cy), (80, 80, 80), 2)
        # 핸들
        hx = self._value_to_px(self.value)
        cv.circle(img, (hx, cy), 7, (180, 180, 180), -1)


class UIButton:
    def __init__(self, label: str, x: int, y: int, w: int, h: int):
        self.label = label
        self.x, self.y = x, y
        self.w, self.h = w, h
        self.active = False
        self.hover = False
        self._clicked = False

    def _contains(self, mx: int, my: int) -> bool:
        return self.x <= mx <= self.x + self.w and self.y <= my <= self.y + self.h

    def handle_mouse(self, event, mx, my):
        inside = self._contains(mx, my)

        if event == cv.EVENT_MOUSEMOVE:
            self.hover = inside or self.active
        elif event == cv.EVENT_LBUTTONDOWN:
            if inside:
                self.active = True
                self.hover = True
            else:
                self.hover = False
        elif event == cv.EVENT_LBUTTONUP:
            if self.active and inside:
                self._clicked = True
            self.active = False
            self.hover = inside

        # 마우스가 버튼을 벗어났고 드래그 중이 아니면 hover 해제
        if not inside and not self.active and event == cv.EVENT_MOUSEMOVE:
            self.hover = False

    def consume_click(self) -> bool:
        if self._clicked:
            self._clicked = False
            return True
        return False

    def draw(self, img):
        base_color = (60, 60, 60)
        hover_color = (80, 80, 80)
        active_color = (100, 100, 100)

        color = base_color
        if self.active:
            color = active_color
        elif self.hover:
            color = hover_color

        cv.rectangle(img, (self.x, self.y), (self.x + self.w, self.y + self.h), color, -1)
        cv.rectangle(img, (self.x, self.y), (self.x + self.w, self.y + self.h), (140, 140, 140), 1)

        text_scale = 0.6
        text_thickness = 2
        text_color = (230, 230, 230)
        (tw, th), baseline = cv.getTextSize(self.label, cv.FONT_HERSHEY_SIMPLEX, text_scale, text_thickness)
        tx = self.x + (self.w - tw) // 2
        ty = self.y + (self.h + th) // 2 - baseline
        cv.putText(img, self.label, (tx, ty), cv.FONT_HERSHEY_SIMPLEX, text_scale, text_color, text_thickness, cv.LINE_AA)

def build_sliders():
    # 레이아웃
    margin = 12
    sw = UI_W - 2 * margin
    sh = 36
    x0 = margin
    y0 = 20
    gap = 10

    sliders = {}
    sliders["CELL"]      = Slider("CELL",       4, 64, CELL, x0, y0 + 0*(sh+gap), sw, sh, step=1, as_int=True)
    sliders["R_MIN"]     = Slider("R_MIN",      0, 32, R_MIN, x0, y0 + 1*(sh+gap), sw, sh, step=1, as_int=True)
    sliders["R_MAX"]     = Slider("R_MAX",      1, 32, min(R_MAX, CELL//2), x0, y0 + 2*(sh+gap), sw, sh, step=1, as_int=True)
    sliders["GAMMA"]     = Slider("GAMMA",    0.2, 3.0, GAMMA, x0, y0 + 3*(sh+gap), sw, sh, step=0.1, as_int=False)
    sliders["BLUR_LVL"]  = Slider("BLUR_LVL",   0, 7, 0 if (BLUR_KSIZE is None or BLUR_KSIZE < 2) else (BLUR_KSIZE - 1)//2,
                                   x0, y0 + 4*(sh+gap), sw, sh, step=1, as_int=True)
    return sliders

def read_params_from_sliders(sliders):
    cell  = int(sliders["CELL"].read())
    rmax  = int(sliders["R_MAX"].read())
    rmin  = int(sliders["R_MIN"].read())
    gamma = float(sliders["GAMMA"].read())
    blur_lvl = int(sliders["BLUR_LVL"].read())

    # 안전 보정
    cell = max(4, min(64, cell))
    rmax = min(rmax, cell // 2, 32)
    rmin = min(rmin, rmax)
    gamma = max(0.2, min(3.0, gamma))
    blur_ksize = 0 if blur_lvl <= 0 else 2*blur_lvl + 1
    return cell, rmin, rmax, gamma, blur_ksize

def draw_ui_panel(panel, sliders, exit_button, fps=None):
    # 패널 배경
    panel[:] = (20, 20, 20)
    # 타이틀
    cv.putText(panel, "HALFTONE CONTROLS", (12, 16), cv.FONT_HERSHEY_SIMPLEX, 0.55, (240,240,240), 1, cv.LINE_AA)
    # 슬라이더들
    for sl in sliders.values():
        sl.draw(panel)
    # FPS
    if fps is not None:
        text_y = exit_button.y - 12
        if text_y < 40:
            text_y = 40
        cv.putText(panel, f"FPS: {fps:.1f}", (12, text_y), cv.FONT_HERSHEY_SIMPLEX, 0.55, (200,200,200), 1, cv.LINE_AA)
    exit_button.draw(panel)

def make_mouse_callback(sliders, exit_button):
    def _cb(event, x, y, flags, param):
        exit_button.handle_mouse(event, x, y)
        for sl in sliders.values():
            sl.handle_mouse(event, x, y)
    return _cb

def run_unified_loop(get_gray_frame_func):
    # 슬라이더 준비
    sliders = build_sliders()
    button_margin = 12
    button_height = 44
    button_width = UI_W - 2 * button_margin
    button_y = CANVAS_H - button_height - button_margin
    exit_button = UIButton("EXIT", button_margin, button_y, button_width, button_height)
    win_name = "Halftone (Unified)"
    cv.namedWindow(win_name, cv.WINDOW_NORMAL)
    cv.resizeWindow(win_name, CANVAS_W, CANVAS_H)
    cv.setMouseCallback(win_name, make_mouse_callback(sliders, exit_button))

    # 디바이스 안내 1회 출력
    if TORCH_AVAILABLE:
        print(f"[Halftone GPU] Using device: {DEVICE}")
    else:
        print("[Halftone GPU] PyTorch not available -> CPU path")

    fps = 0.0
    last_t = time.time()

    while True:
        gray = get_gray_frame_func()
        if gray is None:
            break

        cell, rmin, rmax, gamma, blur_ksize = read_params_from_sliders(sliders)
        H, W = gray.shape
        use_gpu = TORCH_AVAILABLE and DEVICE is not None and str(DEVICE) in ("cuda", "mps")

        if PROC_SCALE < 1.0:
            W2 = max(64, int(W * PROC_SCALE))
            H2 = max(48, int(H * PROC_SCALE))
            gray_s = cv.resize(gray, (W2, H2), interpolation=cv.INTER_AREA)
            # 파라미터 스케일 적용
            cell_s = max(2, int(round(cell * PROC_SCALE)))
            rmax_s = max(1, int(round(rmax * PROC_SCALE)))
            rmin_s = max(0, int(round(rmin * PROC_SCALE)))
            rmax_s = min(rmax_s, cell_s // 2, 32)
            rmin_s = min(rmin_s, rmax_s)
            # 블러 커널 스케일
            if blur_ksize and blur_ksize > 1:
                k = max(1, int(round(blur_ksize * PROC_SCALE)))
                blur_ksize_s = k if (k % 2 == 1) else (k + 1)
            else:
                blur_ksize_s = 0
            if use_gpu:
                ht_small = halftone_gray_gpu(gray_s, cell_s, rmin_s, rmax_s, gamma, blur_ksize_s, DEVICE)
            else:
                ht_small = halftone_gray(gray_s, cell=cell_s, r_min=rmin_s, r_max=rmax_s, gamma=gamma, blur_ksize=blur_ksize_s)
            ht_resized = cv.resize(ht_small, (VID_W, CANVAS_H), interpolation=cv.INTER_NEAREST)
        else:
            if use_gpu:
                ht = halftone_gray_gpu(gray, cell, rmin, rmax, gamma, blur_ksize, DEVICE)
            else:
                ht = halftone_gray(gray, cell=cell, r_min=rmin, r_max=rmax, gamma=gamma, blur_ksize=blur_ksize)
            ht_resized = cv.resize(ht, (VID_W, CANVAS_H), interpolation=cv.INTER_AREA)

        # UI 패널 생성 및 그리기
        panel = np.zeros((CANVAS_H, UI_W, 3), dtype=np.uint8)
        # FPS 업데이트
        now = time.time()
        dt = now - last_t
        last_t = now
        if dt > 0:
            fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps > 0 else (1.0 / dt)
        draw_ui_panel(panel, sliders, exit_button, fps=fps)

        # 합성: [패널 | 영상]
        canvas = np.hstack([panel, ht_resized])
        cv.imshow(win_name, canvas)

        key = cv.waitKey(10) & 0xFF
        if exit_button.consume_click():
            break
        if key == ord('q'):
            break
        elif key == ord('s'):
            ts = time.strftime("%Y%m%d_%H%M%S")
            path = SAVE_DIR / f"halftone_{ts}.png"
            cv.imwrite(str(path), ht)
            print(f"[저장] {path}")

    cv.destroyWindow(win_name)

# ---------------------------
# OpenCV 트랙바 기반 GUI
# ---------------------------
def _noop(x):
    pass

def _blurlevel_to_ksize(level: int) -> int:
    """0 -> 0(블러 없음), n -> 2*n+1 (홀수 커널)"""
    return 0 if level <= 0 else 2 * level + 1

def create_control_panel():
    cv.namedWindow("Controls", cv.WINDOW_NORMAL)
    cv.resizeWindow("Controls", 400, 250)

    # 초기값 준비
    init_cell = int(max(4, min(64, CELL)))
    init_rmax = int(min(R_MAX, init_cell // 2, 32))
    init_rmin = int(min(R_MIN, init_rmax))
    init_gamma10 = int(max(2, min(30, round(GAMMA * 10))))
    init_blur_lvl = 0 if (BLUR_KSIZE is None or BLUR_KSIZE < 2) else (BLUR_KSIZE - 1) // 2
    init_blur_lvl = int(max(0, min(7, init_blur_lvl)))

    cv.createTrackbar("CELL",       "Controls", init_cell,   64, _noop)      # 4~64 (정규화에서 하한 강제)
    cv.createTrackbar("R_MIN",      "Controls", init_rmin,   32, _noop)      # 0~32
    cv.createTrackbar("R_MAX",      "Controls", init_rmax,   32, _noop)      # 1~32 (정규화에서 CELL/2와 동기)
    cv.createTrackbar("GAMMA_x10",  "Controls", init_gamma10, 30, _noop)     # 0~30 → 0.0~3.0 (정규화에서 0.2~3.0)
    cv.createTrackbar("BLUR_LVL",   "Controls", init_blur_lvl, 7, _noop)     # 0(off)~7 → ksize=0/3/5/.../15

def read_safe_params_from_trackbar():
    """트랙바 값을 읽은 뒤 안전 구간으로 정규화하여 반환."""
    cell     = cv.getTrackbarPos("CELL",      "Controls")
    rmin     = cv.getTrackbarPos("R_MIN",     "Controls")
    rmax     = cv.getTrackbarPos("R_MAX",     "Controls")
    gamma10  = cv.getTrackbarPos("GAMMA_x10", "Controls")
    blur_lvl = cv.getTrackbarPos("BLUR_LVL",  "Controls")

    # 정규화/보정
    cell = max(4, min(64, cell if cell > 0 else 4))
    rmax = min(rmax, cell // 2, 32)
    rmin = min(rmin, rmax)
    gamma = max(0.2, min(3.0, gamma10 / 10.0))
    blur_ksize = _blurlevel_to_ksize(blur_lvl)

    return cell, rmin, rmax, gamma, blur_ksize

# ---------------------------
# Split-window UI: Controls (trackbars) + Render
# ---------------------------
def run_split_windows_loop(get_gray_frame_func):
    """Split-window UI: 'Controls' (trackbars) + 'Halftone' (render).
    Trackbar changes apply immediately because we poll them every loop with waitKey(1).
    """
    # Ensure control window exists
    try:
        cv.getTrackbarPos("CELL", "Controls")
    except Exception:
        create_control_panel()

    render_win = "Halftone"
    cv.namedWindow(render_win, cv.WINDOW_NORMAL)
    cv.resizeWindow(render_win, FRAME_W, FRAME_H)

    fps = 0.0
    last_t = time.time()

    use_gpu_possible = TORCH_AVAILABLE and DEVICE is not None and str(DEVICE) in ("cuda", "mps")

    while True:
        gray = get_gray_frame_func()
        if gray is None:
            break

        cell, rmin, rmax, gamma, blur_ksize = read_safe_params_from_trackbar()

        # Optional downscale for speed
        if PROC_SCALE < 1.0:
            W = max(64, int(gray.shape[1] * PROC_SCALE))
            H = max(48, int(gray.shape[0] * PROC_SCALE))
            gray_s = cv.resize(gray, (W, H), interpolation=cv.INTER_AREA)
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
            if use_gpu_possible:
                out_small = halftone_gray_gpu(gray_s, cell_s, rmin_s, rmax_s, gamma, blur_s, DEVICE)
            else:
                out_small = halftone_gray(gray_s, cell=cell_s, r_min=rmin_s, r_max=rmax_s, gamma=gamma, blur_ksize=blur_s)
            out = cv.resize(out_small, (FRAME_W, FRAME_H), interpolation=cv.INTER_NEAREST)
        else:
            if use_gpu_possible:
                out = halftone_gray_gpu(gray, cell, rmin, rmax, gamma, blur_ksize, DEVICE)
            else:
                out = halftone_gray(gray, cell=cell, r_min=rmin, r_max=rmax, gamma=gamma, blur_ksize=blur_ksize)

        # FPS overlay (small, top-left)
        now = time.time()
        dt = now - last_t
        last_t = now
        if dt > 0:
            fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps > 0 else (1.0 / dt)
        cv.putText(out, f"FPS:{fps:.1f}", (10, 20), cv.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1, cv.LINE_AA)

        cv.imshow(render_win, out)

        # Super-short GUI timeslice keeps UI snappy
        key = cv.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            ts = time.strftime("%Y%m%d_%H%M%S")
            path = SAVE_DIR / f"halftone_{ts}.png"
            cv.imwrite(str(path), out)
            print(f"[저장] {path}")

    cv.destroyWindow(render_win)

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

def demo_camera():
    cap = cv.VideoCapture(CAP_INDEX)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, FRAME_H)

    if not cap.isOpened():
        print("[경고] 카메라를 열 수 없습니다. 권한 설정 또는 CAP_INDEX 확인.")
        return

    def _get_gray():
        ok, frame = cap.read()
        if not ok:
            print("[에러] 프레임 캡처 실패")
            return None
        return frame_to_gray(frame)

    print("[키/버튼] q 또는 EXIT: 종료, s: 스냅샷 저장")
    create_control_panel()
    run_split_windows_loop(_get_gray)
    cap.release()

def demo_test_static():
    H, W = FRAME_H, FRAME_W
    grad = np.tile(np.linspace(0, 255, W, dtype=np.uint8), (H, 1))
    y0, y1 = H // 3, 2 * H // 3
    x0, x1 = W // 3, 2 * W // 3
    grad[y0:y1, x0:x1] = 0

    def _get_gray():
        return grad

    print("[키/버튼] q 또는 EXIT: 종료, s: 스냅샷 저장")
    create_control_panel()
    run_split_windows_loop(_get_gray)

if __name__ == "__main__":
    # macOS에서 첫 실행 시 '터미널(또는 파이썬)'의 카메라 권한을 허용해야 함
    if USE_CAMERA:
        demo_camera()
    else:
        demo_test_static()
