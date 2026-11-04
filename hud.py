"""HandLiveness HUD (ported from the FINAL design): Instrument Telemetry in ice cyan. C's information architecture
in A's cinematic language (cool spotlight grade, ice-cyan instruments with a low-res glow, reticles, closing
brackets) with B's plain-language result rows. Amber only ever means "needs your attention".

render(img, hand, ui) -> new BGR image; pure and deterministic, animation reads only ui["t"] / ui["t_in_phase"];
thresholds and durations come only from ui ("fusion", "palm_min_ratio", "protocol"); non-finite numbers are shown
as missing. draw(img, fr, pose, out, stats, debug, cfg, st) adapts the M2 session output (build_ui) and keeps the
display-only state the renderer cannot hold (HudState: HOLD tilt baseline / range, last RESULT debug, the last
hand for <= HAND_HOLD_S over a dropped detection)."""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

import gesture_templates as gt
from gesture_templates import HAND_CONNECTIONS
from liveness.frame import GESTURES, palm_center_px, palm_width_px

GESTURE_NAMES = {"OPEN_PALM": "Open palm", "FIST": "Fist", "THUMBS_UP": "Thumbs up",
                 "V_SIGN": "V sign", "POINT_UP": "Point up", "OK": "OK sign"}
PROMPTS = {"IDLE": "Press SPACE to start", "COUNTDOWN": "Show one hand",
           "HOLD": "Hold it. Slowly tilt your hand left and right"}

# ---------------------------------------------------------------- fonts / text
_FONTS: dict[str, cv2.FontFace] = {}
_TTF = {"mono": "/System/Library/Fonts/SFNSMono.ttf", "sfpro": "/System/Library/Fonts/SFNS.ttf"}


def font(kind: str = "sans") -> cv2.FontFace:
    """'sans' | 'italic' (OpenCV builtin, variable weight) | 'mono' (SF Mono) | 'sfpro' (SF Pro).
    TTF kinds fall back to builtin 'sans' when the file is missing or fails to load."""
    if kind not in _FONTS:
        ff = None
        if kind in _TTF and Path(_TTF[kind]).exists():
            ff = cv2.FontFace(_TTF[kind])
            if not ff.getName():
                ff = None
        _FONTS[kind] = ff or cv2.FontFace("italic" if kind == "italic" else "sans")
    return _FONTS[kind]


def text_rect(text: str, kind: str, size_px: int, weight: int = 400) -> tuple[int, int, int, int]:
    """(x, y, w, h) of the ink box relative to a baseline-left origin; y < 0 is above the baseline."""
    return tuple(int(v) for v in cv2.getTextSize((4096, 4096), text, (0, 0), font(kind), int(size_px), int(weight)))


def put(img, text, org, color, kind="sans", size_px=20, weight=400, align="left"):
    """Draw text with baseline at org[1]; align left/center/right around org[0]. Returns end point."""
    flags = {"left": cv2.PUT_TEXT_ALIGN_LEFT, "center": cv2.PUT_TEXT_ALIGN_CENTER,
             "right": cv2.PUT_TEXT_ALIGN_RIGHT}[align]
    end, _ = cv2.putText(img, text, (int(org[0]), int(org[1])), color, font(kind), int(size_px), int(weight), flags)
    return end


# ---------------------------------------------------------------- geometry helpers
def hand_bbox(lm_px: np.ndarray, pad: float = 0.0) -> tuple[int, int, int, int]:
    x0, y0 = lm_px[:, :2].min(0); x1, y1 = lm_px[:, :2].max(0)
    p = pad * max(x1 - x0, y1 - y0)
    return int(x0 - p), int(y0 - p), int(x1 + p), int(y1 + p)


DISPLAY_ROLL = {"THUMBS_UP": 90.0}   # templates live in a fingers-up hand frame; roll so the thumb points up


def project_template(gid: str, size: float, center, yaw_deg: float = 0.0, pitch_deg: float = 0.0,
                     mirror: bool = True, roll_deg: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Wireframe icon points for a gesture template (the CHALLENGE card icon, PROMPT 3.1).
    Returns pts (21, 2) px fitted into a size x size box at center, and depth (21,) in [0, 1] (1 = nearest).
    Fingers point up; palm faces the viewer; mirror=True puts the thumb on screen-right (user's right
    hand in the mirrored preview). yaw spins around the vertical axis, pitch around the horizontal axis;
    roll_deg (default DISPLAY_ROLL[gid]) turns the gesture in the image plane first."""
    p = gt.template(gid).astype(np.float64)
    r = np.radians(DISPLAY_ROLL.get(gid, 0.0) if roll_deg is None else roll_deg)
    rz = np.array([[np.cos(r), -np.sin(r), 0], [np.sin(r), np.cos(r), 0], [0, 0, 1]])
    p = p @ rz.T
    cy, sy = np.cos(np.radians(yaw_deg)), np.sin(np.radians(yaw_deg))
    cp, sp = np.cos(np.radians(pitch_deg)), np.sin(np.radians(pitch_deg))
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]]); rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
    q = p @ (rx @ ry).T
    x, y, z = q[:, 0], q[:, 1], q[:, 2]
    if mirror:
        x = -x
    s = 0.35 / (0.35 + z)                      # weak perspective, camera 35 cm in front of the palm
    u, v = x * s, y * s
    k = size / max(np.ptp(u), np.ptp(v), 1e-9)
    pts = np.stack([u * k, v * k], 1)
    pts -= (pts.max(0) + pts.min(0)) / 2
    pts += np.asarray(center, dtype=np.float64)
    depth = (z.max() - z) / max(np.ptp(z), 1e-9)
    return pts, depth


def _bgr(h: str) -> tuple[int, int, int]:
    return int(h[5:7], 16), int(h[3:5], 16), int(h[1:3], 16)
# ---------------------------------------------------------------- tokens: colour
PANEL = _bgr("#0B1117")        # plate and chip fill (blue-black graphite)
TRACK = _bgr("#1C2733")        # empty meter track
EDGE = _bgr("#2A3D50")         # plate hairline, separators
TICK = _bgr("#4E6275")         # minor ticks and dim strokes, never text
GREY = _bgr("#9AACBD")         # secondary text
WHITE = _bgr("#F2F6FA")        # primary text, numerals, needle
CYAN = _bgr("#4FD8FF")         # the one accent: every live instrument
CYAN_DIM = _bgr("#1D5A6E")     # done steps, icon palm, far icon edges
RAIL = _bgr("#318CA8")         # unlit HOLD ring rail: >= 3:1 on a dark frame (on skin the INK under-stroke carries it)
BONE = _bgr("#78C4D8")         # skeleton bones
PASS_C = _bgr("#3DDC84")
FAIL_C = _bgr("#FF5468")
AMBER = _bgr("#FFB938")        # attention only: hints, INSUFFICIENT, turn still needed, last second
INK = _bgr("#10141A")          # text on amber / cyan fills, reticle halo, ring under-stroke
VERDICT = {"PASS": PASS_C, "FAIL": FAIL_C, "INSUFFICIENT": AMBER}
VERDICT_LINE = {"PASS": "Liveness verified", "FAIL": "Not verified", "INSUFFICIENT": "Not enough evidence"}
# ---------------------------------------------------------------- tokens: feed grade, glass, glow
VIG_CENTER = 0.86              # centre gain while a hand is present (the spotlight lifts the hand to 1.0)
VIG_CENTER_IDLE = 0.97         # no hand: mild vignette only
VIG_EDGE = 0.66                # corner gain; with CHAN_GAMMA every channel stays >= 0.6
VIG_EDGE_IDLE = 0.74
CHAN_GAMMA = (1.0, 1.05, 1.12) # B, G, R exponents on the gain: shadows turn cool
SPOT = 1.0                     # gain inside the hand spotlight
FROST_SIGMA = 2.5              # plate glass blur at 1/4 scale
GLOW_SIGMA = 1.6               # bloom blur at 1/4 scale
PLATE_A = 0.90                 # plate opacity over the frosted frame
PLATE_M = np.hstack([np.eye(3) * (1 - PLATE_A), np.float32(PANEL)[:, None] * PLATE_A]).astype(np.float32)
# ---------------------------------------------------------------- tokens: layout (px at 720p)
# Vertical lengths scale by H / 720; x, widths, radii, strokes and type by min(H / 720, W / 1280).
M = 24                         # outer margin
HDR_H = 40                     # top status bar
TOP = 52                       # one top y for every plate and the target card
STRIP_H = 64                   # bottom instruction strip
GAP = 14
PAD = 16                       # plate padding
CORNER = 10                    # plate corner accent length
COL_W = 284                    # side column width
RES_W = 296                    # result panel; its left edge sits on the central-zone boundary (x = 960)
CARD_W, CARD_H, WELL = 580, 132, 136   # top target card and its two side wells
HOLD_W, HOLD_H = 560, 84       # compact HOLD card: no icon well, carries the turn readout
ICON_PX = 104                  # wireframe icon box: >= 10 px padding inside the 136 x 132 well (incl. tip dots)
RING_R, HOLD_RING_R = 38, 30   # card ring timer radius (full / compact card)
TRACKING = 0.16                # tracked caps: extra advance as a fraction of the font size
RES_HEAD, RES_LINE, RES_SCORES, RES_NOTE, RES_LEGEND, RES_EMPTY, RES_FOOT = 128, 26, 252, 22, 22, 70, 44
TYPE = {  # role: (kind, size px at 720p, weight); 13 px is the floor at 720p, 10 px at 540p
    "verdict": ("sans", 52, 800), "id": ("sans", 40, 700),
    "num_l": ("mono", 30, 600), "num_m": ("mono", 22, 600), "prompt": ("sans", 24, 600),
    "title": ("sans", 20, 700), "body": ("sans", 16, 500), "num_s": ("mono", 15, 500),
    "label": ("mono", 13, 600), "small": ("mono", 13, 500), "key": ("mono", 13, 700),
}
# ---------------------------------------------------------------- tokens: motion (seconds / Hz)
CLOSE_S = 0.5                  # countdown brackets close in
ENTER_S = 0.35                 # result panel slide-in
FILL_DELAY, FILL_S, ROW_STAGGER = 0.10, 0.40, 0.06   # result bars fill after the panel lands, rows staggered
BADGE_DELAY, BADGE_S = 0.05, 0.40                    # verdict badge draw-in
BADGE_RING, BADGE_GLYPH = 0.6, 0.4                   # ring takes the first 60 %, glyph starts at 40 %
LAST_S = 1.0                   # the first-match deadline turns amber in its last second
GHOST_HZ = 1.2                 # "Move closer" ghost brackets
ICON_YAW, ICON_YAW_RATE = 20, 1.25                   # icon yaw swing (deg) and rate (rad/s): stays readable
MIN_DUR = 1e-3                 # durations <= 0 from config are clamped here (no division by zero)
# ---------------------------------------------------------------- tokens: hold turntable ring (yaw, vertical axis)
RING_GAIN, RING_SPAN = 3, 25   # ring angle = 3 x tilt; +-25 deg of tilt shown (+-75 deg on the ring); 0 = front
RING_K, RING_B = 0.75, 0.26    # half width = 0.75 x palm width (05: ring = 1.19 x hand box); depth b = 0.26 a
RING_DROP = 0.10               # ring centre = palm centre + 0.10 x palm width down
RING_MIN, RING_EDGE = 80, 16   # minimum half width; clearance to the frame's left / right edge (px at 720p)
RING_SHRINK = 0.6              # at a side edge the half width shrinks to 0.6 x nominal at most, then the frame clips
RING_LIFT = 12                 # ring front kept >= this above the strip top (soft clamp, px at 720p)
RING_DASH = 6                  # dash pitch in ring degrees (back half, amber gap)
RING_HEAD = 18                 # arrowhead length and width (px at 720p)
STEPS = (("01", "READY", "COUNTDOWN"), ("02", "MATCH", "CHALLENGE"), ("03", "TURN", "HOLD"), ("04", "RESULT", "RESULT"))
PHASE_KEY = {"COUNTDOWN": "countdown_s", "CHALLENGE": "challenge_s", "HOLD": "hold_s", "RESULT": "result_s"}
CARD_PHASES = ("COUNTDOWN", "CHALLENGE", "HOLD")
HINT_PHASES = ("CHALLENGE", "HOLD")   # the session only sends hints here (M2 0.3); strip, ghost and amber follow it
TIPS = (4, 8, 12, 16, 20)
ROWS = (("planarity", "3D shape"), ("texture", "Skin texture"), ("nonrigid", "Natural motion"),
        ("temporal", "Response timing"))
VETO_ROW = {"PRERECORDED": "temporal", "INSTANT_SWITCH": "temporal", "PLANAR": "planarity"}   # hard check -> row
MATCH_FAIL, HOLD_FAIL = ("TIMEOUT", "WRONG_GESTURE", "NO_TRANSITION"), ("NEED_ROTATION", "HOLD_BROKEN")
HINT_SUB = {"More light": "The scene is too dark for a reliable check",
            "Too bright": "Reduce glare or strong light behind the hand",
            "Show only one hand": "Two hands detected: keep one hand in view"}
BORDERLINE = "Hand size is borderline - move a little closer"   # 'Move closer' while the live ratio already passes
TURN_SUB = "Turn your palm left and right, not just lean it"     # the HUD's own line under a NEED_ROTATION result
KEYS = (("D", "debug"), ("Q", "quit"))   # SPACE is shown only in the IDLE strip (keycap + prompt)
GLYPHS = {"PASS": [[(-0.45, 0.02), (-0.12, 0.34), (0.48, -0.32)]],          # unit-radius polylines
          "FAIL": [[(-0.36, -0.36), (0.36, 0.36)], [(0.36, -0.36), (-0.36, 0.36)]],
          "INSUFFICIENT": [[(0, -0.48), (0, 0.14)]]}                         # stem; the dot is added in _glyph
NUM_KEYS = ("tilt_deg", "tilt_min_deg", "tilt_max_deg", "tilt_target_deg", "total", "palm_ratio", "fps", "proc_ms",
            "stable_progress")
ELL = "..."

_CACHE: dict = {}              # caches keyed by frame size only

# ---------------------------------------------------------------- helpers
def _clamp(x, a=0.0, b=1.0):
    return a if x < a else b if x > b else x
def _ease(x):                             # ease-out cubic on a clamped 0..1 input
    return 1 - (1 - _clamp(x)) ** 3
def _mix(a, b, k):
    return tuple(int(a[i] + (b[i] - a[i]) * _clamp(k)) for i in range(3))
def _scale(col, k):
    return tuple(int(v * k) for v in col)
def _fin(v):
    return isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool) and math.isfinite(v)
def _n(v, fmt="{:.2f}"):
    return "--" if v is None else fmt.format(v)
def _dur(ui, key):                        # config duration; <= 0 or unusable values clamp to MIN_DUR
    v = ui["protocol"][key]
    return float(v) if _fin(v) and v > 0 else MIN_DUR
def _plen(ui):
    return _dur(ui, PHASE_KEY[ui["phase"]])
def _verdict(ui):
    return ui["verdict"] if ui["verdict"] in VERDICT else "INSUFFICIENT"
def _low(s, fl):
    return s is not None and fl is not None and s < fl
def _trunc(v, d=2):                       # a value next to its threshold: floored at the shown precision, never up
    return f"{math.floor(v * 10 ** d + 1e-9) / 10 ** d:.{d}f}"
def _thr(v, d=2):                         # a threshold: >= d decimals, more when needed to be exact (12.5 stays 12.5)
    k = next((k for k in range(d, 5) if abs(round(v, k) - v) < 1e-9), 4)
    return f"{v:.{k}f}"
def _left(v):                            # seconds left, rounded UP to 0.1: reads 0.0 only at the deadline itself
    return math.ceil(max(0.0, v) * 10 - 1e-6) / 10
def _pol(ctr, r, deg):
    a = math.radians(deg)
    return int(ctr[0] + r * math.cos(a)), int(ctr[1] + r * math.sin(a))
def _g4(v):                               # a debug number: 4 significant digits, '--' when missing or not finite
    return f"{v:.4g}" if _fin(v) else "--"
def _fmt(v):
    """Debug value as text that always fits: numbers with 4 significant digits, a sequence as count + range of its
    finite numbers ('10: 12.2..21.53'), None / NaN as '--', anything else str()."""
    if isinstance(v, np.ndarray):
        v = v.tolist()
    if isinstance(v, (list, tuple)):
        f = [float(a) for a in v if _fin(a)]
        if not f:
            return f"{len(v)}: --" if v else "[]"
        return f"{len(v)}: " + (_g4(f[0]) if min(f) == max(f) else f"{_g4(min(f))}..{_g4(max(f))}")
    if v is None or isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool):
        return _g4(v)
    return str(v)
def _clean(ui):
    """Shallow copy of ui with every non-finite number turned into None (the repo marks gaps with NaN)."""
    u = dict(ui)
    for k in NUM_KEYS:
        u[k] = float(u[k]) if _fin(u.get(k)) else None
    if u["stable_progress"] is not None:
        u["stable_progress"] = _clamp(u["stable_progress"])
    u["scores"] = {k: (float(v) if _fin(v) else None) for k, v in (u.get("scores") or {}).items()}
    fu = u.get("fusion") or {}
    u["fusion"] = {"weights": {k: float(v) if _fin(v) else None for k, v in (fu.get("weights") or {}).items()},
                   "floors": {k: float(v) if _fin(v) else None for k, v in (fu.get("floors") or {}).items()},
                   "pass_total": float(fu["pass_total"]) if _fin(fu.get("pass_total")) else None}
    u["palm_min_ratio"] = float(u["palm_min_ratio"]) if _fin(u.get("palm_min_ratio")) else None
    if u.get("e") is not None:
        u["e"] = [float(v) if _fin(v) else None for v in u["e"]]
    return u
def _tilt(ui):
    """(lo, hi, target, sides still needed, ok); target None = not judged. While the range is short BOTH sides are
    still needed (no per-frame side choice: jitter cannot flip it). A missing bound is filled from the other values
    (and 0); a bound that is present is kept."""
    d, tgt, lo, hi = ui["tilt_deg"], ui["tilt_target_deg"], ui["tilt_min_deg"], ui["tilt_max_deg"]
    tgt = tgt if tgt is not None and tgt > 0 else None
    vals = [v for v in (lo, hi, d, 0.0) if v is not None]
    lo, hi = min(vals) if lo is None else lo, max(vals) if hi is None else hi
    lo, hi = min(lo, hi), max(lo, hi)
    if tgt is None or hi - lo >= tgt:
        return lo, hi, tgt, (), tgt is not None
    return lo, hi, tgt, (-1, 1), False
def _grade(img, lm, W, H):
    """Vignette + hand spotlight + cool shadows as one uint8 multiply; masks built at 1/8 scale."""
    sw, sh, key = max(2, W // 8), max(2, H // 8), ("rad", W, H)
    if key not in _CACHE:
        yy, xx = np.mgrid[0:sh, 0:sw].astype(np.float32)
        d = np.sqrt(((xx + 0.5) / sw - 0.5) ** 2 + ((yy + 0.5) / sh - 0.5) ** 2) / 0.7071
        _CACHE[key] = np.clip(d, 0, 1) ** 1.6
    cen, edge = (VIG_CENTER, VIG_EDGE) if lm is not None else (VIG_CENTER_IDLE, VIG_EDGE_IDLE)
    s = cen + (edge - cen) * _CACHE[key]
    if lm is not None:
        x0, y0, x1, y1 = hand_bbox(lm, 0.22)
        spot = np.zeros((sh, sw), np.float32)
        cv2.ellipse(spot, ((x0 + x1) // 16, (y0 + y1) // 16), (max(1, (x1 - x0) // 14), max(1, (y1 - y0) // 14)),
                    0, 0, 360, SPOT, -1)
        s = np.maximum(s, cv2.GaussianBlur(spot, (0, 0), max(sw, sh) / 45))
    m = np.minimum(np.dstack([s ** g for g in CHAN_GAMMA]) * 256, 255).astype(np.uint8)   # 1/256 is the fast path
    return cv2.multiply(img, cv2.resize(m, (W, H), interpolation=cv2.INTER_LINEAR), scale=1 / 256.0)

class _Ctx:
    """One frame being drawn: graded output, 1/4-res frost source and glow layer, unit helpers."""
    def __init__(self, img, hand, ui):
        self.H, self.W = img.shape[:2]
        self.u, self.ux = self.H / 720.0, min(self.H / 720.0, self.W / 1280.0)
        self.ui, self.hand = ui, hand
        self.t, self.tp = float(ui["t"]), float(ui["t_in_phase"])
        lm = hand.get("lm_px")
        ok = bool(hand.get("present")) and lm is not None and bool(np.isfinite(lm).all())
        self.lm = lm if ok else None
        self.out = _grade(img, self.lm, self.W, self.H)
        sm = cv2.resize(self.out, (max(1, self.W // 4), max(1, self.H // 4)), interpolation=cv2.INTER_AREA)
        self.sm = cv2.GaussianBlur(sm, (0, 0), FROST_SIGMA)
        self.glow, self.gbox = np.zeros_like(sm), None
        self.foot_top = self.H - self.s(M + STRIP_H)
        self.card_bot = self.s(TOP + (HOLD_H if ui["phase"] == "HOLD" else CARD_H))
    def s(self, v):                       # vertical 720p length on this frame
        return int(round(v * self.u))
    def sx(self, v):                      # horizontal length / radius on this frame
        return int(round(v * self.ux))
    def lw(self, v=1.0):                  # stroke width, never below 1 px
        return max(1, self.sx(v))
    def px(self, size):                   # font size, never below 10 px
        return max(10, self.sx(size))

# ---------------------------------------------------------------- primitives
def _txt(c, s, x, y, col, role="body", align="left", size=None, px=None):
    kind, sz, w = TYPE[role]
    px = px or c.px(size or sz)            # px: exact pixel size from _shrink (no rounding through c.px)
    if align != "left":                   # OpenCV 5.0 ALIGN_CENTER is unreliable: align by measuring
        bx, _, tw, _ = text_rect(s, kind, px, w)
        x = x - bx - (tw // 2 if align == "center" else tw)
    return put(c.out, s, (int(x), int(y)), col, kind, px, w)
def _tw(c, s, role="body", size=None):
    kind, sz, w = TYPE[role]
    return text_rect(s, kind, c.px(size or sz), w)[2]
def _shrink(c, s, role, width, size=None):
    """Largest pixel size <= the role size at which s fits width, stepping 1 px, floor = the 13 px minimum."""
    kind, sz, w = TYPE[role]
    px, lo = c.px(size or sz), c.px(13)
    while px > lo and text_rect(s, kind, px, w)[2] > width:
        px -= 1
    return px
def _lab(c, s, role, width, size=None):
    """Fixed label: whole, else drop trailing words ('QUALITY GATE' -> 'QUALITY'), else ''; never an ellipsis."""
    while s and _tw(c, s, role, size) > width:
        cut = max(s.rfind(" "), s.rfind("_"))
        s = s[:cut].rstrip(" /-") if cut > 0 else ""
    return s
def _fit(c, s, role, width, size=None):
    """Clip to width with an ASCII ellipsis. ponytail: linear trim after a proportional first cut."""
    tw = _tw(c, s, role, size)
    if tw <= width:
        return s
    s = s[:max(0, int(len(s) * width / max(1, tw)))]
    while s and _tw(c, s + ELL, role, size) > width:
        s = s[:-1]
    return s.rstrip() + ELL if _tw(c, ELL, role, size) <= width else ""
def _track(c, s, x, y, col, maxw=None, short=None):
    """Tracked mono caps drawn glyph by glyph (A's micro-label craft). Returns the width."""
    px, w = c.px(TYPE["label"][1]), TYPE["label"][2]
    step = text_rect("MM", "mono", px, w)[2] - text_rect("M", "mono", px, w)[2] + TRACKING * px
    width = int(step * len(s) - TRACKING * px)
    if maxw is not None and width > maxw:  # no room for tracking: plain label, then its short variant
        s = s if _tw(c, s, "label") <= maxw or not short else short
        s = _lab(c, s, "label", maxw)
        return _txt(c, s, x, y, col, "label")[0] - int(x) if s else 0
    for i, ch in enumerate(s):
        if ch != " ":
            put(c.out, ch, (int(x + i * step), int(y)), col, "mono", px, w)
    return width
def _lines(c, s, role, width, size=None):  # greedy word wrap, nothing clipped
    lines = []
    for word in s.split():
        if lines and _tw(c, lines[-1] + " " + word, role, size) <= width:
            lines[-1] += " " + word
        else:
            lines.append(word)
    return lines
def _wrap(c, s, role, width, max_lines=99, size=None):
    lines = _lines(c, s, role, width, size)
    if len(lines) > max_lines:
        lines = lines[:max_lines - 1] + [" ".join(lines[max_lines - 1:])]
    return [_fit(c, ln, role, width, size) for ln in lines]
def _arc(c, ctr, r, a0, a1, col, th, img=None):
    cv2.ellipse(c.out if img is None else img, (int(ctr[0]), int(ctr[1])), (max(1, int(r)),) * 2, 0, a0, a1, col,
                th, cv2.LINE_AA)
def _gmark(c, x0, y0, x1, y1):            # grow the glow bounding box (1/4-res coordinates)
    b = c.gbox
    c.gbox = (x0, y0, x1, y1) if b is None else (min(b[0], x0), min(b[1], y0), max(b[2], x1), max(b[3], y1))
def _gline(c, a, b, col, th=2):           # bloom layer at 1/4 res
    p, q = (int(a[0]) // 4, int(a[1]) // 4), (int(b[0]) // 4, int(b[1]) // 4)
    _gmark(c, min(p[0], q[0]), min(p[1], q[1]), max(p[0], q[0]), max(p[1], q[1]))
    cv2.line(c.glow, p, q, col, th, cv2.LINE_AA)
def _garc(c, ctr, r, a0, a1, col, th=2):
    q, rq = (int(ctr[0]) // 4, int(ctr[1]) // 4), max(1, int(r) // 4)
    _gmark(c, q[0] - rq, q[1] - rq, q[0] + rq, q[1] + rq)
    _arc(c, q, rq, a0, a1, col, th, c.glow)
def _bloom(c):
    """Flush the glow layer inside its bounding box only (blur, upscale, add) before plates are drawn."""
    if c.gbox is None:
        return
    m, (gh, gw) = int(3 * GLOW_SIGMA) + 4, c.glow.shape[:2]
    x0, y0 = max(0, c.gbox[0] - m), max(0, c.gbox[1] - m)
    x1, y1 = min(gw, c.gbox[2] + m + 1), min(gh, c.gbox[3] + m + 1)
    c.gbox = None
    if x1 <= x0 or y1 <= y0:
        return
    g = cv2.GaussianBlur(c.glow[y0:y1, x0:x1], (0, 0), GLOW_SIGMA)
    c.glow[y0:y1, x0:x1] = 0
    X1, Y1 = min(c.W, 4 * x1), min(c.out.shape[0], 4 * y1)   # c.out may be the hand layer's row view
    if X1 <= 4 * x0 or Y1 <= 4 * y0:
        return
    up = cv2.resize(g, (4 * (x1 - x0), 4 * (y1 - y0)), interpolation=cv2.INTER_LINEAR)
    roi = c.out[4 * y0:Y1, 4 * x0:X1]
    cv2.add(roi, up[:Y1 - 4 * y0, :X1 - 4 * x0], dst=roi)
def _plate(c, x0, y0, x1, y1, accent=CYAN):
    """Frosted graphite plate: 1/4-scale blurred frame under the fill, hairline, L-shaped corner accents."""
    x0, y0, x1, y1 = max(0, int(x0)), max(0, int(y0)), min(c.W, int(x1)), min(c.H, int(y1))
    if x1 - x0 < 2 or y1 - y0 < 2:
        return
    sh, sw = c.sm.shape[:2]
    a, b = min(x0 // 4, sw - 1), min(y0 // 4, sh - 1)
    fr = cv2.resize(c.sm[b:max(b + 1, (y1 + 3) // 4), a:max(a + 1, (x1 + 3) // 4)], (x1 - x0, y1 - y0),
                    interpolation=cv2.INTER_LINEAR)
    cv2.transform(fr, PLATE_M, dst=c.out[y0:y1, x0:x1])   # PANEL over frost in one pass, written into the ROI
    cv2.rectangle(c.out, (x0, y0), (x1 - 1, y1 - 1), EDGE, 1, cv2.LINE_AA)
    if accent:
        _corners(c, (x0, y0, x1 - 1, y1 - 1), c.sx(CORNER), accent, c.lw(2))
def _corners(c, box, k, col, th, glow=0.0):
    """L-shaped corner marks: plate accents and the hand brackets."""
    x0, y0, x1, y1 = box
    for px, py, dx, dy in ((x0, y0, 1, 1), (x1, y0, -1, 1), (x0, y1, 1, -1), (x1, y1, -1, -1)):
        pts = [(px, py + dy * k), (px, py), (px + dx * k, py)]
        cv2.polylines(c.out, [np.array(pts, np.int32)], False, col, th, cv2.LINE_AA)
        if glow:
            _gline(c, pts[0], pts[1], _scale(col, glow))
            _gline(c, pts[1], pts[2], _scale(col, glow))
def _plate_head(c, x0, y0, x1, label, right=None, rcol=GREY, rshort=None):
    """Section label row: tracked caps title, optional right tag (whole, else its short variant, else none)."""
    p = c.sx(PAD)
    e = _track(c, label, x0 + p, y0 + c.s(24), GREY, x1 - x0 - 2 * p)
    room = x1 - x0 - 2 * p - e - c.sx(12)
    right = next((r for r in (right, rshort) if r and _tw(c, r, "label") <= room), None)
    if right:
        _txt(c, right, x1 - p, y0 + c.s(24), rcol, "label", "right")
    cv2.line(c.out, (x0 + p, y0 + c.s(34)), (x1 - p, y0 + c.s(34)), EDGE, 1, cv2.LINE_AA)
    return y0 + c.s(34)
def _split(c, w, label, val, vrole, keep):
    """Does the value need its own continuation row? Yes when it is wider than the row at the floor size, or
    (keep) when it does not fit beside the whole label."""
    vw = text_rect(val, TYPE[vrole][0], _shrink(c, val, vrole, w), TYPE[vrole][2])[2]
    return vw > w or keep and _tw(c, label, "small") + c.sx(10) + vw > w
def _row(c, x0, x1, y, label, val, vcol=WHITE, vrole="num_s", keep=False, dy=0):
    """Telemetry row: value right-aligned, never clipped (shrinks to the floor); the label gets the room left over
    (keep: whole). A value _split() moves goes on a continuation row dy lower. Nothing is drawn left of x0.
    Returns the rows used (1 or 2)."""
    w, two = x1 - x0, _split(c, x1 - x0, label, val, vrole, keep)
    px = _shrink(c, val, vrole, w)
    vw = text_rect(val, TYPE[vrole][0], px, TYPE[vrole][2])[2]
    _txt(c, _lab(c, label, "small", w if two else w - vw - c.sx(10)), x0, y, GREY, "small")
    _txt(c, val, max(x1, x0 + vw), y + dy * two, vcol, vrole, "right", px=px)
    return 1 + two
def _meter(c, x, y, w, h, frac, col, marks=(), hatch=False):
    """Horizontal meter. marks: [(value 0..1, colour)] ticks that overhang the track."""
    cv2.rectangle(c.out, (x, y), (x + w, y + h), TRACK, -1)
    if hatch:                              # "no data": diagonal hatch instead of a fill
        for hx in range(x, x + w, max(2, c.sx(8))):
            cv2.line(c.out, (hx, y + h), (min(x + w, hx + h), y), _scale(AMBER, 0.55), 1, cv2.LINE_AA)
    elif frac > 0:
        cv2.rectangle(c.out, (x, y), (x + int(w * min(1.0, frac)), y + h), col, -1)
    for v, mc in marks:
        mx = x + int(w * _clamp(v))
        cv2.line(c.out, (mx, y - c.s(4)), (mx, y + h + c.s(4)), mc, c.lw(2), cv2.LINE_AA)
def _glyph(c, kind, ctr, r, col, lw, k=1.0):
    """Check / cross / exclamation drawn with lines; k < 1 draws each stroke in (B's draw-on)."""
    for line in GLYPHS.get(kind, GLYPHS["INSUFFICIENT"]):
        p = np.array(line, float) * r + np.array(ctr, float)
        m = _clamp(k) * (len(p) - 1)       # draw-in progress measured in segments
        i = min(int(m), len(p) - 2)
        q = np.vstack([p[:i + 1], p[i] + (p[i + 1] - p[i]) * (m - i)])
        if m > 0:
            cv2.polylines(c.out, [q.astype(np.int32)], False, col, lw, cv2.LINE_AA)
    if kind not in ("PASS", "FAIL") and k > 0.8:
        cv2.circle(c.out, (int(ctr[0]), int(ctr[1] + 0.40 * r)), max(1, lw * 2 // 3), col, -1, cv2.LINE_AA)
def _badge(c, verdict, ctr, r, col, k=1.0):
    """Verdict ring + glyph so colour is never the only cue; k draws the ring in, then the glyph."""
    _arc(c, ctr, r, -90, -90 + 360 * _ease(k / BADGE_RING), col, c.lw(2.5))
    _glyph(c, verdict, ctr, r, col, c.lw(3), _ease((k - BADGE_GLYPH) / (1 - BADGE_GLYPH)))
def _chevron(c, x, y, s, d, col, lw):
    p = np.array([[-d * 0.5, -1], [d * 0.5, 0], [-d * 0.5, 1]]) * s + (x, y)
    cv2.polylines(c.out, [p.astype(np.int32)], False, col, lw, cv2.LINE_AA)
def _key_w(c, label):
    return max(c.sx(22), _tw(c, label, "key") + c.sx(12))
def _key(c, label, x, y, hot=False, ink=False):
    """Keycap: outlined box with the key name (filled cyan when hot); returns its right edge."""
    w, h = _key_w(c, label), c.s(22)
    if hot:
        cv2.rectangle(c.out, (x, y - h), (x + w, y), CYAN, -1)
    cv2.rectangle(c.out, (x, y - h), (x + w, y), INK if ink else CYAN if hot else GREY, 1, cv2.LINE_AA)
    _txt(c, label, x + w // 2, y - c.s(6), INK if hot or ink else WHITE, "key", "center")
    return x + w
def _chip(c, s, x, y, col, align="left", role="label", xmax=None):
    """Text on a solid scrim (anything that sits over the live image); y is the chip bottom, xmax its right limit."""
    p, fs = c.sx(7), c.px(TYPE[role][1])
    s = _fit(c, s, role, c.W - 2 * p - 4)
    w = _tw(c, s, role)
    x0 = int(max(0, min((xmax or c.W) - w - 2 * p, x - (w // 2 if align == "center" else 0) - p)))
    y = int(max(fs + c.s(12), y))
    cv2.rectangle(c.out, (x0, y - fs - c.s(12)), (x0 + w + 2 * p, y), PANEL, -1)
    cv2.line(c.out, (x0, y - fs - c.s(12)), (x0, y - 1), col, c.lw(2))
    _txt(c, s, x0 + p, y - c.s(6) - max(0, fs - c.s(13)) // 4, col, role)

# ---------------------------------------------------------------- hand layer
def _matched(ui):                         # CHALLENGE: first match AND the target still seen (raw or smoothed);
    return bool(ui["first_match"]) and ui["target"] in (ui["raw_label"], ui["smooth_label"])   # else "lost" (U1)
def _lock(ui):
    """0 = searching (loose, breathing brackets) .. 1 = locked (tight)."""
    if ui["phase"] == "CHALLENGE":
        return 0.15 + 0.85 * (ui["stable_progress"] or 0.0) if _matched(ui) else 0.0
    return {"COUNTDOWN": 0.25, "HOLD": 1.0, "RESULT": 1.0}.get(ui["phase"], 0.0)
def _state_col(c):
    ui, ph = c.ui, c.ui["phase"]
    if ph == "RESULT":
        return VERDICT[_verdict(ui)]
    if (ui["hint"] and ph in HINT_PHASES) or (ph == "COUNTDOWN" and c.lm is not None and ui["quality_ok"] is False):
        return AMBER                       # CHALLENGE / HOLD follow the debounced hint only (HINT-FLICKER)
    return WHITE if ph == "CHALLENGE" and not _matched(ui) else CYAN
def _hand_layer(c, col):
    """Hairline skeleton, five fingertip reticles, glowing corner brackets; in HOLD the turntable ring replaces the
    brackets. Draws into c.out as given (render passes the row view above the strip). Returns (box, palm centre)."""
    if c.lm is None:
        return None
    ip = [tuple(p) for p in c.lm[:, :2].astype(int).tolist()]
    ui, lock, dbg, ph = c.ui, _lock(c.ui), c.ui["debug"], c.ui["phase"]
    ring = _ring_geom(c) if ph == "HOLD" else None
    if ring:
        _ring(c, ring, back=True)          # order: far half, skeleton, front half, reticles
    for a, b in HAND_CONNECTIONS:
        cv2.line(c.out, ip[a], ip[b], _mix(BONE, WHITE, 0.35) if dbg else BONE, c.lw(1.6 if dbg else 1.2), cv2.LINE_AA)
    for i in set(range(21)) - set(TIPS):
        cv2.circle(c.out, ip[i], c.lw(3 if dbg else 2), WHITE if dbg else _mix(BONE, WHITE, 0.5), -1, cv2.LINE_AA)
    if ring:
        _ring(c, ring, back=False)
    r = int(round(_clamp(0.035 * palm_width_px(c.lm) / c.ux, 4, 6) * c.ux))   # 4..6 px at 720p, follows the palm
    rc = _mix(BONE, WHITE, 0.5) if ph == "RESULT" else col
    for i in TIPS:                         # light reticle in the hand colour, 1 px ink edge so it reads on skin
        cv2.circle(c.out, ip[i], r, INK, c.lw(1.2) + c.lw(2), cv2.LINE_AA)
        cv2.circle(c.out, ip[i], r, rc, c.lw(1.2), cv2.LINE_AA)
        for k in range(4):                 # four ticks that rotate 45 deg as the lock rises
            a = 45 * lock + 90 * k
            cv2.line(c.out, _pol(ip[i], r + c.sx(2), a), _pol(ip[i], r + c.sx(5), a), rc, c.lw(1.2), cv2.LINE_AA)
    x0, y0, x1, y1 = hand_bbox(c.lm)
    if ring:
        return (x0, y0, x1, y1), palm_center_px(c.lm)
    pad = (0.17 - 0.07 * lock) * max(x1 - x0, y1 - y0, 1)
    if ph == "COUNTDOWN":                  # brackets close in on the hand
        pad += (1 - _ease(c.tp / CLOSE_S)) * c.sx(60)
    elif ph == "CHALLENGE" and not _matched(ui):
        pad += c.sx(5) * math.sin(c.t * 4.0)
    bx0, bx1 = (int(_clamp(v, c.sx(4), c.W - c.sx(4))) for v in (x0 - pad, x1 + pad))
    by1, top, ov = int(y1 + pad), c.s(HDR_H + 6), 1.0     # no bottom clamp: the row view cuts them at the strip
    if ph in CARD_PHASES:                  # stay under the card where the box is under it (eased over 40 px of overlap)
        cx0 = c.W // 2 - c.sx(CARD_W) // 2
        top, ov = c.card_bot + c.s(10), _clamp((min(bx1, c.W - cx0) - max(bx0, cx0)) / max(1, c.sx(40)))
    by0 = y0 - pad
    box = (bx0, int(by0 + max(0.0, min(top, by1 - c.s(40)) - by0) * ov), bx1, by1)
    k = int(min(box[2] - box[0], box[3] - box[1]) * 0.2 + c.sx(10))
    _corners(c, box, k, col, c.lw(2), glow=0.45)
    if ui["hint"] == "Move closer" and ph in HINT_PHASES:   # amber ghost brackets expand outward: "make it bigger"
        g = (c.t * GHOST_HZ) % 1.0
        o = int(c.sx(32) * _ease(g))
        _corners(c, (box[0] - o, box[1] - o, box[2] + o, box[3] + o), k, _scale(AMBER, 1 - g), c.lw(1.5))
    return box, palm_center_px(c.lm)
def _lock_ring(c, ctr, r, frac):
    p, r = (int(ctr[0]), int(ctr[1])), max(c.sx(20), int(r))
    cv2.circle(c.out, p, r, TICK, 1, cv2.LINE_AA)
    for i in range(24):
        cv2.line(c.out, _pol(p, r, i * 15), _pol(p, r - c.sx(6 if i % 6 == 0 else 3), i * 15), TICK, 1, cv2.LINE_AA)
    if frac > 0:
        _arc(c, p, r, -90, -90 + 360 * frac, CYAN, c.lw(3.5))
        _garc(c, p, r, -90, -90 + 360 * frac, _scale(CYAN, 0.7), 3)
def _ring_geom(c):
    """(cx, cy, a, b) from the palm centre (P0/5/9/13/17) and palm width (max(|P5-P17|, |P0-P9|)); continuous.
    At a side edge a shrinks to RING_SHRINK x nominal at most, then the frame clips the ring. Near the strip the
    centre is lifted by a soft clamp (softplus) so the front stays RING_LIFT above the strip top."""
    pw, (cx, cy) = palm_width_px(c.lm), palm_center_px(c.lm).astype(float)
    a0 = max(RING_K * pw, c.sx(RING_MIN), 1.0)
    a = max(RING_SHRINK * a0, min(a0, cx - c.sx(RING_EDGE), c.W - c.sx(RING_EDGE) - cx))
    b, k = RING_B * a, max(1.0, c.sx(RING_LIFT))
    lim = c.foot_top - b - k
    cy = lim - k * float(np.logaddexp(0.0, (lim - cy - RING_DROP * pw) / k))   # = cy + drop well above lim
    return cx, cy, a, b
def _rp(g, t):                            # ring point at ring angle t (deg): front = lowest point, t > 0 = right
    r = np.radians(t)
    return np.stack([g[0] + g[2] * np.sin(r), g[1] + g[3] * np.cos(r)], -1)
def _rl(c, g, t0, t1, col, th, img=None):  # ring arc t0..t1 as a sub-pixel polyline (~2 deg steps)
    q = _rp(g, np.linspace(t0, t1, max(2, int(abs(t1 - t0) / 2) + 2)))
    cv2.polylines(c.out if img is None else img, [np.int32(np.round(q * 16))], False, col, th, cv2.LINE_AA, 4)
def _ring(c, g, back):
    """HOLD turntable in perspective: ring angle = RING_GAIN x tilt (the card keeps true degrees). Covered = cyan
    span; while short, still needed = a dashed amber run of half the missing ring angle + arrowhead from EACH span
    end (both until the target is reached); live tilt = white marker."""
    cx, cy, a, b = g
    if back:                               # far half (behind the hand): thin dashes
        for t in range(90, 270, RING_DASH):
            _rl(c, g, t, t + RING_DASH / 2, TICK, 1)
        return
    (lo, hi, tgt, miss, ok), S, G = _tilt(c.ui), RING_SPAN, RING_GAIN
    _rl(c, g, -90, 90, INK, c.lw(2) + 2)      # ink under-stroke: the rail reads on skin
    _rl(c, g, -90, 90, RAIL, c.lw(2))         # and >= 3:1 on a dark frame where it leaves the hand
    for t in range(-S, S + 1, 5):          # a tick every 5 deg of tilt along the outward normal, major at 0
        p, r = _rp(g, G * t), math.radians(G * t)
        n = np.array([b * math.sin(r), a * math.cos(r)])
        q = p + n / max(1e-6, float(np.hypot(*n))) * c.sx(9 if t == 0 else 5)
        cv2.line(c.out, tuple(np.int32(p * 16)), tuple(np.int32(q * 16)), GREY if t == 0 else TICK, 1, cv2.LINE_AA, 4)
    lo_s, hi_s = G * _clamp(lo, -S, S), G * _clamp(hi, -S, S)
    _rl(c, g, lo_s, hi_s, CYAN, c.lw(5))
    _rl(c, (cx / 4, cy / 4, a / 4, b / 4), lo_s, hi_s, _scale(CYAN, 0.8), 3, c.glow)
    _gmark(c, int(cx - a) // 4, int(cy - b) // 4, int(cx + a) // 4 + 1, int(cy + b) // 4 + 1)
    _bloom(c)                              # flush the span glow now: amber runs, heads, marker stay unwashed on top
    gap = 0.0 if ok or tgt is None else G * (tgt - (hi - lo)) / max(1, len(miss))
    for side in miss:                      # dashed amber run from each span end, arrowhead the way to keep turning
        e0 = lo_s if side < 0 else hi_s
        e1 = _clamp(e0 + side * gap, -G * S, G * S)
        for t in np.arange(0, abs(e1 - e0), RING_DASH):
            _rl(c, g, e0 + side * t, e0 + side * min(t + RING_DASH * 0.6, abs(e1 - e0)), AMBER, c.lw(2))
        run = _rp(g, np.linspace(e0, e1, 12))
        head, short = c.sx(RING_HEAD), float(np.hypot(*np.diff(run, axis=0).T).sum()) < c.sx(RING_HEAD)
        r = math.radians(e0 if short else e1)  # a gap shorter than the head: the head starts at the span end
        d = side * np.array([a * math.cos(r), -b * math.sin(r)])
        d /= max(1e-6, float(np.hypot(*d)))
        tip = _rp(g, e0) + d * head if short else run[-1]
        base, wing = tip - d * head, np.array([-d[1], d[0]]) * head / 2
        tri = [np.int32(np.round(np.array([tip, base + wing, base - wing]) * 16))]
        cv2.fillPoly(c.out, tri, AMBER, cv2.LINE_AA, 4)
        cv2.polylines(c.out, tri, True, INK, 1, cv2.LINE_AA, 4)
    if ok:                                 # range reached: cyan check under the ring front, never green
        _glyph(c, "PASS", (cx, cy + b + c.sx(17)), c.sx(9), CYAN, c.lw(2))
    if c.ui["tilt_deg"] is not None:       # live tilt: white marker with an ink edge
        p = tuple(np.int32(np.round(_rp(g, G * _clamp(c.ui["tilt_deg"], -S, S)) * 16)))
        cv2.circle(c.out, p, 16 * (c.sx(5) + 1), INK, -1, cv2.LINE_AA, 4)
        cv2.circle(c.out, p, 16 * c.sx(5), WHITE, -1, cv2.LINE_AA, 4)

# ---------------------------------------------------------------- header / footer
def _steps(ui, act):
    """Step bar marks (colour, filled, cross): the live step in cyan; on RESULT, where the round ended."""
    done, todo = (CYAN_DIM, True, False), (TICK, False, False)
    if ui["phase"] != "RESULT":
        return [done if i < act else (CYAN, True, False) if i == act else todo for i in range(4)]
    v, rs = _verdict(ui), set(ui["reasons"] or ())
    if v != "PASS" and rs & set(MATCH_FAIL):
        return [done, (VERDICT[v], False, True), todo, todo]
    if v != "PASS" and rs & set(HOLD_FAIL):
        return [done, done, (VERDICT[v], True, False), todo]
    return [done] * 3 + [(VERDICT[v], True, False)]
def _header(c):
    ui, W, h = c.ui, c.W, c.s(HDR_H)
    _plate(c, 0, 0, W, h, accent=None)
    x, y, b = c.sx(M), c.s(26), c.sx(12)
    cv2.rectangle(c.out, (x, y - b), (x + b, y), CYAN, c.lw(2), cv2.LINE_AA)
    cv2.circle(c.out, (x + b // 2, y - b // 2), c.lw(2), CYAN, -1, cv2.LINE_AA)
    e = x + c.sx(22) + _track(c, "HAND LIVENESS", x + c.sx(22), y, WHITE)
    act, sw = next((i for i, st in enumerate(STEPS) if st[2] == ui["phase"]), -1), c.sx(124)
    s0, q = W // 2 - 2 * sw, c.sx(11)
    if e + c.sx(12) + _tw(c, "CHALLENGE / RESPONSE", "small") < s0 - c.sx(12):   # subtitle only when it fits
        _txt(c, "CHALLENGE / RESPONSE", e + c.sx(12), y, GREY, "small")
    for i, ((num, name, _), (col, fill, cross)) in enumerate(zip(STEPS, _steps(ui, act))):
        bx = s0 + i * sw
        cv2.rectangle(c.out, (bx, y - q), (bx + q, y), col, -1 if fill else 1)
        if cross:
            _glyph(c, "FAIL", (bx + q / 2, y - q / 2), q, col, c.lw(1.5))
        end = _txt(c, f"{num} {name}", bx + c.sx(18), y, GREY if col in (CYAN_DIM, TICK) else col, "label")
        if i < 3 and end[0] + c.sx(8) < bx + sw - c.sx(8):
            cv2.line(c.out, (end[0] + c.sx(8), y - c.s(5)), (bx + sw - c.sx(8), y - c.s(5)), EDGE, 1, cv2.LINE_AA)
    rx = W - c.sx(M)
    if ui["recording"]:                    # red lamp (the convention); the word stays white
        _txt(c, "REC", rx, y, WHITE, "label", "right")
        dc = (rx - _tw(c, "REC", "label") - c.sx(12), y - c.s(6))
        cv2.circle(c.out, dc, c.sx(5), FAIL_C, -1, cv2.LINE_AA)
        cv2.circle(c.out, dc, c.sx(7 + 3 * (c.t % 1.0)), _mix(PANEL, GREY, 1 - c.t % 1.0), 1, cv2.LINE_AA)
        rx = dc[0] - c.sx(20)
    _txt(c, f"CAM {_n(ui['fps'], '{:.0f}')} FPS", rx, y, GREY, "small", "right")
def _footer(c, text, icon, line2=None, l2col=GREY):
    """Bottom strip: glyph, instruction (+ line 2) left of the key legend; a hint turns it amber."""
    ui, hint, pr, pm = c.ui, c.ui["hint"], c.ui["palm_ratio"], c.ui["palm_min_ratio"]
    if hint and ui["phase"] in HINT_PHASES:
        text, icon, line2, l2col = hint, "warn", HINT_SUB.get(hint), INK
        if hint == "Move closer" and pr is not None:   # numbers only while they say "too small"
            line2 = f"Hand too small: {_trunc(pr * 100, 1)}% of frame, need {_thr(pm * 100, 0)}%" \
                if pm is not None and pr < pm else BORDERLINE
    x0, x1, y0, y1 = c.sx(M), c.W - c.sx(M), c.foot_top, c.H - c.s(M)
    warn, cy, r = icon == "warn", (y0 + y1) // 2, c.sx(14)
    if warn:
        cv2.rectangle(c.out, (x0, y0), (x1, y1), AMBER, -1)
    else:
        _plate(c, x0, y0, x1, y1)
    x, ctr = x0 + c.sx(PAD), (x0 + c.sx(PAD) + r, cy)
    if icon == "space":
        x = _key(c, "SPACE", x, cy + c.s(11), hot=True) + c.sx(14)
    elif warn:
        tri = np.array([[ctr[0], cy - r], [ctr[0] + r, cy + r - c.sx(2)], [ctr[0] - r, cy + r - c.sx(2)]], np.int32)
        cv2.polylines(c.out, [tri], True, INK, c.lw(2.5), cv2.LINE_AA)
        _glyph(c, "INSUFFICIENT", (ctr[0], cy + c.sx(3)), r * 0.8, INK, c.lw(2.5))
    elif icon == "tilt":
        for s in (-1, 1):
            _chevron(c, ctr[0] + s * c.sx(9), cy, c.sx(9), s, CYAN, c.lw(2.5))
    elif icon in VERDICT:
        _badge(c, icon, ctr, r, VERDICT[icon])
    else:                                  # reticle
        cv2.circle(c.out, ctr, r, CYAN, c.lw(2), cv2.LINE_AA)
        cv2.circle(c.out, ctr, c.sx(4), CYAN, -1, cv2.LINE_AA)
    x = x if icon == "space" else x + 2 * r + c.sx(14)
    if ui["phase"] == "RESULT":            # PROMPT 3.1: SPACE only starts from IDLE; R ends the round -> IDLE
        first = [("X", "discard sample")] * bool(ui["recording"]) + [("R", "back to start")]
    else:
        first = [("R", "reset")]
    for short in (False, True):            # long key labels, else their first word
        keys = [(k, n.split()[0] if short else n) for k, n in first] + list(KEYS)
        widths = [_key_w(c, k) + c.sx(20) + _tw(c, n, "small") for k, n in keys]
        kx, ky = x1 - c.sx(PAD) - sum(widths), cy + c.s(11)
        room = kx - c.sx(20) - x
        if _tw(c, text, "prompt") <= room:
            break
    role = "prompt"
    if not line2 and _tw(c, text, "prompt") > room:          # a long sentence wraps to two body lines, no "..."
        lines = _wrap(c, text, "body", room, 2)
        text, line2, l2col, role = lines[0], (lines[1:] or [None])[0], INK if warn else WHITE, "body"
    _txt(c, _fit(c, text, role, room), x, cy + c.s(3) if line2 else cy + c.s(9), INK if warn else WHITE, role)
    if line2:
        _txt(c, _fit(c, line2, "body", room), x, cy + c.s(22), INK if warn else l2col, "body")
    for (k, n), w in zip(keys, widths):
        _txt(c, n, _key(c, k, kx, ky, ink=warn) + c.sx(6), ky - c.s(6),
             INK if warn else GREY, "small")
        kx += w

# ---------------------------------------------------------------- top target card
def _icon(c, gid, ctr, size, yaw):
    """Wireframe of the template: glowing bones far to near (far side dim), BONE joints, larger CYAN tips."""
    pts, dep = project_template(gid, size, ctr, yaw_deg=yaw, pitch_deg=-12)
    q = [tuple(p) for p in np.int32(np.round(pts * 16)).tolist()]
    for a, b in sorted(HAND_CONNECTIONS, key=lambda ab: dep[ab[0]] + dep[ab[1]]):
        col = _mix(CYAN_DIM, CYAN, 0.8 * (dep[a] + dep[b]) - 0.3)
        cv2.line(c.out, q[a], q[b], col, c.lw(2), cv2.LINE_AA, 4)
        _gline(c, pts[a], pts[b], _scale(col, 0.4))
    for i in range(21):
        cv2.circle(c.out, q[i], 16 * (c.sx(3) if i in TIPS else c.lw(1.5)), CYAN if i in TIPS else BONE, -1,
                   cv2.LINE_AA, 4)
    _bloom(c)
def _ring_timer(c, ctr, r, frac, col, n=20):
    """Segmented ring that empties counter-clockwise from 12 o'clock."""
    for i in range(n):
        a0 = -90 + i * 360 / n + 2
        a1 = a0 + 360 / n - 4
        _arc(c, ctr, r, a0, a1, TRACK, c.lw(3))
        if i < frac * n:
            _arc(c, ctr, r, a0, a0 + (a1 - a0) * min(1.0, frac * n - i), col, c.lw(3))
def _secs(c, v, ctr, role):
    """Seconds left inside a ring: big number (rounded up, _left) + small 's', centred as one unit."""
    s, gap = f"{_left(v):.1f}", c.sx(3)
    a = _tw(c, s, role)
    x, y = ctr[0] - (a + gap + _tw(c, "s", "small")) // 2, ctr[1] + c.sx(0.36 * TYPE[role][1])
    _txt(c, s, x, y, WHITE, role)
    _txt(c, "s", x + a + gap, y, GREY, "small")
def _fit_id(c, s, width, base):
    """Big ID: 1 px steps down until it fits (4 px margin), floor 13 px, clipped only below. Returns (text, px)."""
    room = width - c.lw(4)
    px = _shrink(c, s, "id", room, base)
    return (s if text_rect(s, "sans", px, TYPE["id"][2])[2] <= room else _fit(c, s, "id", room, 13)), px
def _hold_card(c):
    """HOLD: compact card - target ID, what to do (TURN LEFT / RIGHT, then KEEP HOLDING), the RANGE readout,
    seconds left (never amber)."""
    ui, w, h, p = c.ui, c.sx(HOLD_W), c.s(HOLD_H), c.sx(PAD)
    x0, y0 = c.W // 2 - w // 2, c.s(TOP)
    _plate(c, x0, y0, x0 + w, y0 + h)
    rr, L = c.sx(HOLD_RING_R), _plen(ui)
    rc = (x0 + w - p - rr, y0 + h // 2)
    _ring_timer(c, rc, rr, _clamp((L - c.tp) / L), CYAN)
    _secs(c, max(0.0, L - c.tp), rc, "num_s")
    tx, bw, lw = x0 + p + c.sx(4), c.sx(176), _tw(c, "TIME LEFT", "small")
    rx = rc[0] - rr - c.sx(12)
    if rx - lw - c.sx(12) - bw - c.sx(16) - tx >= c.sx(150):     # "TIME LEFT" only while the ID keeps 150 px
        _txt(c, "TIME LEFT", rx, rc[1] + c.s(5), GREY, "small", "right")
        rx -= lw + c.sx(12)
    bx = rx - bw
    aw = bx - c.sx(16) - tx
    cv2.line(c.out, (bx - c.sx(8), y0 + c.s(14)), (bx - c.sx(8), y0 + h - c.s(14)), EDGE, 1, cv2.LINE_AA)
    _track(c, "HOLD THIS GESTURE", tx, y0 + c.s(28), CYAN, aw)
    title, px = _fit_id(c, ui["target"] or "--", aw, 30)
    _txt(c, title, tx - c.sx(1), y0 + c.s(66), WHITE, "id", px=px)
    lo, hi, tgt, miss, ok = _tilt(ui)      # no side word: the range is not a direction (both arrows on the ring)
    do, dcol, short = ("KEEP HOLDING", CYAN, "HOLD") if ok else ("TURN LEFT / RIGHT", AMBER if miss else GREY, "TURN")
    _track(c, do, bx, y0 + c.s(28), dcol, bw, short)
    t = "--" if tgt is None else _thr(tgt, 0)
    val = f"{t if ok else _trunc(hi - lo, len(t.partition('.')[2]))} / {t} deg"    # capped once reached
    rx = _txt(c, "RANGE", bx, y0 + c.s(64), GREY, "small")[0] + c.sx(6)
    e = _txt(c, val, rx, y0 + c.s(64), WHITE, "num_m", px=_shrink(c, val, "num_m", bx + bw - c.sx(24) - rx))
    if ok:
        _glyph(c, "PASS", (e[0] + c.sx(12), y0 + c.s(57)), c.sx(8), CYAN, c.lw(2))
def _card(c):
    """Top centre: what to do (wireframe icon + ID) and how long (ring = seconds left for the current goal)."""
    ui, ph = c.ui, c.ui["phase"]
    if ph == "HOLD":
        return _hold_card(c)
    w, h, wl = c.sx(CARD_W), c.s(CARD_H), c.sx(WELL)
    x0, y0 = c.W // 2 - w // 2, c.s(TOP)
    x1, y1 = x0 + w, y0 + h
    _plate(c, x0, y0, x1, y1)
    for xd in (x0 + wl, x1 - wl):
        cv2.line(c.out, (xd, y0 + c.s(14)), (xd, y1 - c.s(14)), EDGE, 1, cv2.LINE_AA)
    gid = "OPEN_PALM" if ph == "COUNTDOWN" else ui["target"]
    if gid in GESTURES:
        yaw = ICON_YAW * math.sin(c.t * ICON_YAW_RATE)
        _icon(c, gid, (x0 + wl // 2, (y0 + y1) // 2), c.sx(ICON_PX), yaw)
        if ui["debug"]:
            _chip(c, f"TURN {yaw:+.0f} deg", x0, y1 + c.s(30), GREY)
    name, matched = GESTURE_NAMES.get(gid, ""), ph == "CHALLENGE" and _matched(ui)
    lost, first = ph == "CHALLENGE" and ui["first_match"] and not matched, ph == "CHALLENGE" and not ui["first_match"]
    L = _plen(ui)                          # CHALLENGE: one ring + number to challenge_s for the whole phase
    left, fm = max(0.0, L - c.tp), _dur(ui, "first_match_s")
    if ph == "COUNTDOWN":
        eyebrow, ecol, title, name, rlab = "GET READY", GREY, "Challenge in", "A random gesture comes next", ()
    elif matched:                          # the number keeps counting; a badge marks the match
        eyebrow, ecol, title, rlab = "MATCH FOUND - HOLD STEADY", CYAN, gid or "--", ("MATCHED",)
    elif lost:                             # matched once, gesture gone: the number keeps counting, no check (U1)
        eyebrow, ecol, title, rlab = "MATCH LOST - SHOW IT AGAIN", AMBER, gid or "--", ("MATCH LOST", "LOST")
    else:                                  # the label counts to the first-match deadline (the notch on the ring)
        m = f"{_left(fm - c.tp):.1f} s"
        eyebrow, ecol, title, rlab = "MAKE THIS GESTURE", GREY, gid or "--", (f"MATCH IN {m}", m)
    late = first and fm - c.tp < LAST_S
    rcol, lcol = (AMBER, AMBER) if late or lost else (CYAN, GREY)
    tx, tw = x0 + wl + c.sx(18), w - 2 * wl - c.sx(36)
    _track(c, eyebrow, tx, y0 + c.s(32), ecol, tw)
    title, px = _fit_id(c, title, tw, TYPE["id"][1])
    _txt(c, title, tx - c.sx(2), y0 + c.s(80), WHITE, "id", px=px)
    for i, ln in enumerate(_wrap(c, name, "body", tw, 2, 17)):    # a sentence wraps instead of "..."
        _txt(c, ln, tx, y0 + c.s(108) + i * (c.px(17) + c.s(3)), GREY, "body", size=17)
    rc, rr = (x1 - wl // 2, y0 + c.s(56)), c.sx(RING_R)
    _ring_timer(c, rc, rr, _clamp(left / L), rcol)
    if first and 0 < fm < L:   # first-match deadline notch, white on the ring
        a = -90 + 360 * (L - fm) / L
        cv2.line(c.out, _pol(rc, rr - c.sx(6), a), _pol(rc, rr + c.sx(6), a), WHITE, c.lw(2), cv2.LINE_AA)
    _secs(c, left, rc, "num_m")
    if matched:                            # cyan check badge on the ring, beside the number (never replacing it)
        bc, br = _pol(rc, rr, 45), c.sx(11)
        cv2.circle(c.out, bc, br + c.lw(1), PANEL, -1, cv2.LINE_AA)
        cv2.circle(c.out, bc, br, CYAN, -1, cv2.LINE_AA)
        _glyph(c, "PASS", bc, br * 0.62, INK, c.lw(2))
    if rlab:                               # long label, else its short variant ("MATCH IN 2.1 s" -> "2.1 s")
        w = wl - c.sx(8)
        _txt(c, _lab(c, next((r for r in rlab if _tw(c, r, "small") <= w), rlab[-1]), "small", w), rc[0],
             y1 - c.s(11), lcol, "small", "center")

# ---------------------------------------------------------------- side plates
def _col(c, side):
    x0 = c.sx(M) if side == "L" else c.W - c.sx(M + COL_W)
    return x0, x0 + c.sx(COL_W)
def _signal_plate(c, y0):
    """Debug only (right column): per-frame quantities as telemetry rows (status dot, label, value)."""
    ui, n, pr, q, pm = c.ui, c.hand.get("n_hands"), c.ui["palm_ratio"], c.ui["quality_ok"], c.ui["palm_min_ratio"]
    n = int(n) if _fin(n) else None
    rows = [("RAW", str(ui["raw_label"] or "NONE"), None), ("SMOOTH", str(ui["smooth_label"] or "NONE"), None),
            ("HANDS", _n(n, "{}"), None if n is None else n == 1),
            ("PALM WIDTH", "--" if pr is None else f"{_trunc(pr * 100, 1)}%", None if pr is None or pm is None
             else pr >= pm),
            ("QUALITY GATE", "--" if q is None else "OK" if q else "LOW", None if q is None else bool(q)),
            ("FPS / PROC", f"{_n(ui['fps'], '{:.0f}')} / {_n(ui['proc_ms'], '{:.1f}')} ms", None)]
    if ui["phase"] == "HOLD":              # the live turn value lives here now (no hand tag in HOLD)
        rows.append(("TURN", _n(ui["tilt_deg"], "{:+.1f} deg"), None))
    (x0, x1), p = _col(c, "R"), c.sx(PAD)
    lx0, lx1 = x0 + p + c.sx(16), x1 - p
    y1 = y0 + c.s(34 + 24 * sum(1 + _split(c, lx1 - lx0, a, b, "num_s", False) for a, b, _ in rows) + 16)
    _plate(c, x0, y0, x1, y1)
    y = _plate_head(c, x0, y0, x1, "SIGNAL", "DEBUG")
    for label, val, ok in rows:
        y += c.s(24)
        cv2.circle(c.out, (x0 + p + c.sx(4), y - c.s(5)), c.sx(3.5), TICK if ok is None else CYAN if ok else AMBER, -1)
        y += c.s(24) * (_row(c, lx0, lx1, y, label, val, dy=c.s(24)) - 1)
    return y1
def _finger_plate(c, y0):
    """Debug only: five finger-extension gauges."""
    e, (x0, x1), p = c.ui["e"], _col(c, "L"), c.sx(PAD)
    _plate(c, x0, y0, x1, y0 + c.s(214))
    yh = _plate_head(c, x0, y0, x1, "FINGER EXTENSION", "0 - 1")
    sh, g, cw = c.s(9), c.s(3), (x1 - x0 - 2 * p) / 5
    bot = yh + c.s(14) + 10 * (sh + g)
    for i, name in enumerate("TIMRP"):
        cx, bw = int(x0 + p + cw * (i + 0.5)), int(cw * 0.21)
        v = None if e is None or len(e) <= i else e[i]
        for s in range(10):
            cv2.rectangle(c.out, (cx - bw, bot - s * (sh + g) - sh), (cx + bw, bot - s * (sh + g)),
                          CYAN if v is not None and v * 10 > s + 0.5 else TRACK, -1)
        _txt(c, name, cx, bot + c.s(22), WHITE, "label", "center")
        s = _n(v) if v is None or _tw(c, _n(v), "small") <= cw - 2 else _trunc(v, 1)   # narrow: one decimal
        _txt(c, s, cx, bot + c.s(40), GREY, "small", "center")
    return y0 + c.s(214)
def _debug_values_plate(c, y0, title="RESULT VALUES"):
    """Debug values as label / value rows; a value that does not fit beside its label takes a continuation row.
    Rows that do not fit above the strip are left out whole."""
    dv, (x0, x1), p, rh = c.ui["debug_values"] or {}, _col(c, "L"), c.sx(PAD), c.s(18)
    cap, items, n = (c.foot_top - c.s(GAP + 50) - y0) // max(1, rh), [], 0
    for k, v in dv.items():
        k, v = str(k), _fmt(v)
        r = 1 + _split(c, x1 - x0 - 2 * p, k, v, "small", True)
        if n + r > cap:
            break
        items.append((k, v))
        n += r
    if not items:                          # no room above the strip: draw nothing rather than overlap it
        return
    _plate(c, x0, y0, x1, y0 + c.s(50) + rh * n)
    y = _plate_head(c, x0, y0, x1, title, "DEBUG") + c.s(4)
    for k, v in items:
        y += rh
        y += rh * (_row(c, x0 + p, x1 - p, y, k, v, WHITE, "small", True, rh) - 1)
def _idle_plate(c):
    ui, (x0, x1), p, y0 = c.ui, _col(c, "L"), c.sx(PAD), c.s(TOP)
    lines = _wrap(c, "Show the gesture you are given, then turn your palm left and right.", "body", x1 - x0 - 2 * p)
    steps = (("Show one hand", f"{_dur(ui, 'countdown_s'):.1f} s"),
             ("Make the gesture shown", f"within {_dur(ui, 'first_match_s'):.1f} s"),
             ("Hold it and turn your palm", f"{_dur(ui, 'hold_s'):.1f} s"))
    sx, step = x0 + p + c.sx(34), c.px(17) + c.s(2)
    wraps = [_wrap(c, a, "body", x1 - p - sx, 2, 17) for a, _ in steps]     # a step wraps instead of "..."
    more = step * sum(len(w) - 1 for w in wraps)
    y1 = y0 + c.s(34 + 38 + 22 * len(lines) + 16 + 52 * len(steps) + 8) + more
    _plate(c, x0, y0, x1, y1)
    y = _plate_head(c, x0, y0, x1, "PROTOCOL", f"{len(steps)} STEPS") + c.s(36)
    _txt(c, _lab(c, "Hand liveness check", "title", x1 - x0 - 2 * p, 22), x0 + p, y, WHITE, "title", size=22)
    for ln in lines:
        y += c.s(22)
        _txt(c, ln, x0 + p, y, GREY, "body")
    cy = y + c.s(34)
    for i, ((a, b), al) in enumerate(zip(steps, wraps)):
        cv2.circle(c.out, (x0 + p + c.sx(12), cy), c.sx(12), CYAN, c.lw(1.5), cv2.LINE_AA)
        _txt(c, str(i + 1), x0 + p + c.sx(12), cy + c.s(5), WHITE, "label", "center")
        for j, ln in enumerate(al):
            _txt(c, ln, sx, cy + c.s(2) + j * step, WHITE, "body", size=17)
        _txt(c, _lab(c, b, "small", x1 - p - sx), sx, cy + c.s(21) + (len(al) - 1) * step, GREY, "small")
        cy += c.s(52) + (len(al) - 1) * step
    return y1
def _idle_reticle(c):
    ctr, r = (c.W // 2, int(c.H * 0.47)), c.sx(92)
    cv2.circle(c.out, ctr, r, TICK, 1, cv2.LINE_AA)
    for k in range(4):                     # slow standby rotation, 24 deg/s
        a = c.t * 24 + k * 90
        _arc(c, ctr, r + c.sx(8), a - 12, a + 12, CYAN, c.lw(2))
        _garc(c, ctr, r + c.sx(8), a - 12, a + 12, _scale(CYAN, 0.6))
    cv2.circle(c.out, ctr, c.sx(3), GREY, -1, cv2.LINE_AA)
    _chip(c, "PLACE ONE HAND HERE", ctr[0], ctr[1] + r + c.s(48), WHITE, "center")

# ---------------------------------------------------------------- result panel
def _result_panel(c):
    """Right panel: verdict + glyph + reason, liveness score vs the pass line, plain-language check rows."""
    ui, v = c.ui, _verdict(c.ui)
    col, sc, dbg, floors = VERDICT[v], ui["scores"], ui["debug"], ui["fusion"]["floors"]
    w, p = c.sx(RES_W), c.sx(PAD)
    x0 = c.W - c.sx(M) - w + int((1 - _ease(c.tp / ENTER_S)) * c.sx(48))
    x1, iw = x0 + w, w - 2 * p
    reason, rw = ui["reason_text"] or ("Live hand confirmed" if v == "PASS" else ""), iw - c.sx(44)
    rrole = "title" if len(_lines(c, reason, "title", rw)) <= 2 else "body"   # 2 title lines, else 2 body lines
    lines = _wrap(c, reason, rrole, rw, 2)
    rows = [(k, name, sc.get(k), floors.get(k)) for k, name in ROWS]
    notes = [(dbg and k != "temporal") or _low(s, fl) for k, _, s, fl in rows]
    body = RES_SCORES + RES_NOTE * sum(notes) + (RES_LEGEND if dbg else 0) if sc else RES_EMPTY
    y0 = c.s(TOP)
    y1 = y0 + c.s(RES_HEAD + RES_LINE * max(1, len(lines)) + body + RES_FOOT)
    _plate(c, x0, y0, x1, y1, col)
    cv2.rectangle(c.out, (x0, y0), (x1 - 1, y0 + c.lw(3) - 1), col, -1)   # 3 px verdict top bar (inclusive)
    yh = _plate_head(c, x0, y0, x1, "RESULT", f"TARGET {ui['target'] or '--'}", WHITE, ui["target"])
    vs = min(TYPE["verdict"][1], TYPE["verdict"][1] * iw / max(1, _tw(c, v, "verdict")))
    _txt(c, v, x0 + p - c.sx(2), yh + c.s(14) + c.sx(vs * 0.74), col, "verdict", size=vs)
    ly = yh + c.s(RES_HEAD - 38)
    _badge(c, v, (x0 + p + c.sx(15), ly - c.s(7)), c.sx(15), col, (c.tp - BADGE_DELAY) / BADGE_S)
    for i, ln in enumerate(lines):
        _txt(c, ln, x0 + p + c.sx(44), ly + c.s(RES_LINE * i), WHITE, rrole)
    y = ly + c.s(RES_LINE * max(1, len(lines)) - 8)
    cv2.line(c.out, (x0 + p, y), (x1 - p, y), EDGE, 1)
    if not sc:                             # TIMEOUT / WRONG_GESTURE: nothing was measured
        _txt(c, _lab(c, "No checks were run", "title", iw), x0 + p, y + c.s(32), WHITE, "title")
        _txt(c, _fit(c, f"Target was {ui['target'] or '--'}", "body", iw), x0 + p, y + c.s(56), GREY, "body")
    else:
        _score_block(c, x0, x1, y, v, rows, notes)
    L, ry = _plen(ui), y1 - c.s(20)
    _txt(c, "Back to start in", x0 + p, ry, GREY, "small")
    _txt(c, f"{_left(L - c.tp):.1f} s", x1 - p, ry, WHITE, "num_s", "right")
    _meter(c, x0 + p, ry + c.s(7), iw, c.lw(2), 1 - c.tp / L, col)
def _score_block(c, x0, x1, y, v, rows, notes):
    """LIVENESS SCORE vs the pass line, then one row per check; a hard veto marks its row (red bar, band)."""
    ui, p, col = c.ui, c.sx(PAD), VERDICT[v]
    tot, dbg, sc, fu = ui["total"], ui["debug"], ui["scores"], ui["fusion"]
    veto = v == "FAIL" and tot is None     # failed a hard check before fusion: not scored, not "no data"
    vrow = {VETO_ROW.get(r) for r in ui["reasons"] or ()} if veto else set()
    mw, kt = x1 - x0 - 2 * p, _ease((c.tp - FILL_DELAY) / FILL_S)
    y += c.s(34)
    val = "" if veto else _n(tot) if tot is None else _trunc(tot)
    _track(c, "LIVENESS SCORE", x0 + p, y, GREY, mw - _tw(c, val or " ", "num_l") - c.sx(10), "SCORE")
    if val:
        _txt(c, val, x1 - p, y + c.s(4), GREY if tot is None else col, "num_l", "right")
    ty, th = y + c.s(16), c.s(12)
    parts = [(k, fu["weights"][k] * sc[k]) for k, _ in ROWS if None not in (fu["weights"].get(k), sc.get(k))]
    if dbg and tot is not None:            # debug: weighted contributions, same hue at falling strength
        _meter(c, x0 + p, ty, mw, th, 0, col)
        xa = x0 + p
        for j, (_, share) in enumerate(parts):
            xb = xa + int(mw * _clamp(share) * kt)
            cv2.rectangle(c.out, (xa, ty), (max(xa, xb - c.sx(2)), ty + th), _mix(PANEL, col, 1 - 0.3 * j), -1)
            xa = xb
    elif veto:                             # not scored: the statement takes the meter's place
        s = "Not scored - failed a hard check"
        _txt(c, s if _tw(c, s, "body") <= mw else _lab(c, "Not scored", "body", mw), x0 + p, ty + th, FAIL_C, "body")
    else:                                  # normal: one verdict-colour fill to the total
        _meter(c, x0 + p, ty, mw, th, 0 if tot is None else tot * kt, col, hatch=tot is None)
    pt = fu["pass_total"]
    y = ty + th + c.s(20)
    if tot is not None and pt is not None:  # the pass line only means something next to a total
        mx = x0 + p + int(mw * _clamp(pt))
        cv2.line(c.out, (mx, ty - c.s(5)), (mx, ty + th + c.s(5)), WHITE, c.lw(2), cv2.LINE_AA)
        _txt(c, f"pass at {_thr(pt)}", mx, y, GREY, "small", "center")
    if dbg and tot is not None:            # contribution legend (debug)
        lx, y = x0 + p, y + c.s(22)
        for j, (k, share) in enumerate(parts):
            lab = f"{k[0].upper()} {share:.2f}"
            if lx + c.sx(14) + _tw(c, lab, "small") > x1 - p:     # legend entries that do not fit are dropped
                break
            cv2.rectangle(c.out, (lx, y - c.sx(9)), (lx + c.sx(9), y), _mix(PANEL, col, 1 - 0.3 * j), -1)
            lx = _txt(c, lab, lx + c.sx(14), y, GREY, "small")[0] + c.sx(14)
    y += c.s(16)
    for i, ((key, name, s, fl), note) in enumerate(zip(rows, notes)):
        k, low, hard = _ease((c.tp - FILL_DELAY - ROW_STAGGER * i) / FILL_S), _low(s, fl), key in vrow
        y += c.s(34)
        if hard:                           # the check that vetoed the round
            y0r, y1r = y - c.s(24), y + c.s(10 if key == "temporal" else 18)
            roi = c.out[y0r:y1r, x0 + c.lw(1):x1 - c.lw(1)]
            cv2.addWeighted(roi, 0.94, np.full_like(roi, FAIL_C), 0.06, 0, dst=roi)
            cv2.rectangle(c.out, (x0 + c.lw(1), y0r), (x0 + c.lw(1) + c.lw(3) - 1, y1r - 1), FAIL_C, -1)
        _txt(c, name, x0 + p, y, WHITE if hard else GREY, "body")   # red label hit 4.4:1 on the band
        if key == "temporal":              # a gate, not a score: passed / failed with a drawn glyph
            ok = s is not None and s >= 1.0
            val, gc = ("NO DATA", AMBER) if s is None else ("passed", PASS_C if v == "PASS" else CYAN) if ok \
                else ("failed", FAIL_C)    # green only on a PASS card
            _txt(c, val, x1 - p - (0 if s is None else c.sx(22)), y, gc if s is None or hard else WHITE, "num_s",
                 "right")
            if s is not None:
                _glyph(c, "PASS" if ok else "FAIL", (x1 - p - c.sx(8), y - c.s(5)), c.sx(10), gc, c.lw(2), k)
        else:
            _txt(c, "NO DATA" if s is None else _trunc(s), x1 - p, y,
                 AMBER if s is None else FAIL_C if low or hard else WHITE, "num_s", "right")
            _meter(c, x0 + p, y + c.s(9), mw, c.s(5), 0 if s is None else s * k, FAIL_C if low else CYAN,
                   hatch=s is None, marks=[(fl, WHITE)] if fl is not None else [])
            y += c.s(6)
        if note:
            tech, fs = f"{key}  w {_n(fu['weights'].get(key))}", f"below floor {_thr(fl)}" if low else ""
            if dbg and _tw(c, tech, "small") + _tw(c, fs or " ", "small") + c.sx(10) <= mw:   # debug name + weight
                _txt(c, tech, x0 + p, y + c.s(26), GREY, "small")
            if low:
                _txt(c, fs, x1 - p, y + c.s(26), FAIL_C, "small", "right")
            y += c.s(RES_NOTE)

# ---------------------------------------------------------------- phases (hand layer clipped at the strip, then plates)
def _tag(c, hb, text, col):
    """Status chip on a continuous clamp (never relocates): min(box bottom + 32, strip - 10), left of the panel."""
    if hb is None or not text:
        return
    xmax = c.W - c.sx(M + RES_W + 8) if c.ui["phase"] == "RESULT" else None
    _chip(c, text.upper(), hb[0][0], min(hb[0][3] + c.s(32), c.foot_top - c.s(10)), col, xmax=xmax)
def _hand_hud(c, col):
    """Everything on the live hand, drawn into the row view out[:foot_top] (same coordinates). Returns hb."""
    full, c.out = c.out, c.out[:max(1, c.foot_top)]
    ui, ph, lab = c.ui, c.ui["phase"], str(c.ui["smooth_label"] or "NONE")
    hb = _hand_layer(c, col)
    text = {"COUNTDOWN": "HAND FOUND", "RESULT": _verdict(ui), "HOLD": None}.get(ph, "HAND FOUND")
    if ph == "CHALLENGE":                  # after the first match the lock ring alone shows the hold progress
        if hb is not None and _matched(ui):
            b = hb[0]
            _lock_ring(c, hb[1], max(c.sx(40), 0.30 * max(b[2] - b[0], b[3] - b[1])), ui["stable_progress"] or 0.0)
        text = None if _matched(ui) else ui["hint"] or ("NO GESTURE" if lab == "NONE" else f"SEEING {lab}")
    elif hb is None and ph not in PHASES:  # IDLE without a hand: standby reticle
        _idle_reticle(c)
    _bloom(c)
    _tag(c, hb, text, CYAN if ph not in PHASES else col)
    c.out = full
    return hb
def _phase_idle(c):
    y = _idle_plate(c)
    if c.ui["debug"]:
        _signal_plate(c, c.s(TOP))
        if c.ui["debug_values"]:           # live gesture-tuning values (dev-5) under the protocol plate
            _debug_values_plate(c, y + c.s(GAP), "GESTURE VALUES")
    _footer(c, c.ui["prompt"] or PROMPTS["IDLE"], "space")
def _phase_card(c):                        # COUNTDOWN / CHALLENGE / HOLD: target card, debug columns, strip
    ui, ph = c.ui, c.ui["phase"]
    _card(c)
    if ui["debug"]:
        y = _finger_plate(c, c.s(TOP))
        _signal_plate(c, c.s(TOP))
        if ui["debug_values"]:             # live gesture-tuning values (dev-5) under the finger plate
            _debug_values_plate(c, y + c.s(GAP), "GESTURE VALUES")
    text = ui["prompt"] or PROMPTS[ph] if ph != "CHALLENGE" else \
        "Match found. Hold steady" if _matched(ui) else f"Lost it - show {ui['target'] or '--'} and hold steady" \
        if ui["first_match"] else "Make the gesture shown above"
    _footer(c, text, "tilt" if ph == "HOLD" else "target")
def _phase_result(c):
    v = _verdict(c.ui)
    if c.ui["debug"] and c.ui["debug_values"]:
        _debug_values_plate(c, c.s(TOP))
    _result_panel(c)
    _footer(c, VERDICT_LINE[v], v, TURN_SUB if "NEED_ROTATION" in (c.ui["reasons"] or ()) else None)

PHASES = {"COUNTDOWN": _phase_card, "CHALLENGE": _phase_card, "HOLD": _phase_card, "RESULT": _phase_result}

# ---------------------------------------------------------------- entry point
def render(img: np.ndarray, hand: dict, ui: dict) -> np.ndarray:
    c = _Ctx(img, hand, _clean(ui))
    _hand_hud(c, _state_col(c))
    PHASES.get(c.ui["phase"], _phase_idle)(c)
    _header(c)
    if c.ui["phase"] == "RESULT":          # verdict frame for the whole phase
        cv2.rectangle(c.out, (1, 1), (c.W - 2, c.H - 2), VERDICT[_verdict(c.ui)], c.lw(2))
    return c.out



# ---------------------------------------------------------------- adapter: M2 session output -> ui contract
# ponytail: margin over planarity.min_normal_deg so that >= min_pairs frame pairs clear the 12 deg rule;
# tune in M4 if NEED_ROTATION is common.
TILT_TARGET_K = 1.5
HAND_HOLD_S = 0.15                         # a dropped detection keeps the last hand on screen this long (frame time)
HINT_ON_S, HINT_OFF_S = 0.2, 0.6           # a session hint shows after HINT_ON_S seen, stays HINT_OFF_S after it clears


@dataclass
class HudState:
    """Display-only state across frames: HOLD tilt baseline and running range, debug of the last RESULT, the last
    hand with finite landmarks and its frame time."""
    yaw0: float | None = None              # yaw of the first HOLD frame with a usable palm normal
    tilt_min: float = 0.0
    tilt_max: float = 0.0
    result_debug: dict | None = None
    last_hand: dict | None = None
    last_t: float = 0.0
    hint: str | None = None                # debounced hint on screen and the last frame time it was seen
    hint_seen: float = 0.0
    cand: str | None = None                # pending hint, first and last frame time it was seen
    cand_t0: float = 0.0
    cand_seen: float = 0.0


def palm_yaw_deg(lm_world) -> float | None:
    """Palm yaw about the vertical axis, degrees; turning the camera-facing side toward screen-right increases it,
    for either hand and with the palm or the back to the camera. Axes (checked on data/smoke and the
    gesture_templates conventions): lm_world x points like lm_px x (screen right), y down, smaller z = nearer
    the camera. None for NaN or degenerate input. The camera-facing flip makes yaw jump by 180 deg when the
    hand passes edge-on, so yaw is only defined mod 180: differences must be wrapped mod 180 (build_ui)."""
    p = np.asarray(lm_world, np.float64)
    if p.shape != (21, 3) or not np.isfinite(p[[0, 5, 17]]).all():
        return None
    n = np.cross(p[5] - p[0], p[17] - p[0])
    if np.linalg.norm(n) < 1e-12:
        return None
    toward = -n[2]                         # component toward the camera
    if toward < 0:
        n, toward = -n, -toward
    return float(np.degrees(np.arctan2(n[0], toward)))


def _shown_hint(out, st: HudState) -> str | None:
    """HINT-FLICKER: display-only debounce of the per-frame session hint (M2 sends quality.frame_hint every frame).
    It shows once seen for HINT_ON_S (gaps <= HINT_ON_S keep it pending) and stays HINT_OFF_S after it was last
    seen, so one-frame blips never reach the screen and dropped frames (no hand = no hint) keep the banner."""
    h, t = out.hint, out.t
    if out.phase not in HINT_PHASES:      # a new round starts clean (a new Session always passes IDLE first)
        st.hint = st.cand = None
    if h is not None and out.phase in HINT_PHASES:
        if h != st.cand or t - st.cand_seen > HINT_ON_S:
            st.cand, st.cand_t0 = h, t
        st.cand_seen = t
        if h == st.hint or t - st.cand_t0 >= HINT_ON_S - 1e-6:
            st.hint, st.hint_seen = h, t
    if st.hint is not None and t - st.hint_seen > HINT_OFF_S:
        st.hint = None
    return st.hint


def _dv(v, fmt) -> str:
    return "--" if v is None or not np.isfinite(v) else format(v, fmt)


def _gesture_values(fr, pose, g: dict) -> dict:
    """Debug plate for live phases: everything the M1 / dev-5 overlay showed that the HUD plates do not."""
    ok = fr is not None and fr.present
    nan4 = (float("nan"),) * 4
    e = pose.e if pose is not None else (float("nan"),) * 5
    p2, p3 = (pose.pip2d_deg, pose.pip_deg) if pose is not None else (nan4, nan4)
    th = (pose.thumb_tip_ratio, pose.thumb_wrist_ratio, pose.thumb_cross_cos, pose.tip_gap_ratio) \
        if pose is not None else nan4
    return {
        "hand score": _dv(fr.hand_score if ok else None, ".2f"),
        "roi mean / sat": f"{_dv(fr.roi_mean if ok else None, '.0f')} / "
                          f"{_dv(fr.roi_sat_ratio * 100 if ok else None, '.1f')}%",
        "e TIMRP": " ".join(_dv(v, ".1f") for v in e),   # one decimal: fits the 4:3 column
        f"pip2d >{g['pip_extended_deg']:g}": " ".join(_dv(v, ".0f") for v in p2),
        f"pip3d >{g['pip3d_min_deg']:g}": " ".join(_dv(v, ".0f") for v in p3),
        f"thumb tip >{g['thumb_tip_ratio']:g}": _dv(th[0], ".2f"),
        "thumb wrist": _dv(th[1], ".2f"),
        "thumb cross": _dv(th[2], "+.2f"),
        f"tip gap <{g['ok_tip_ratio']:g}": _dv(th[3], ".2f"),
    }


def build_ui(out, fr, pose, stats: dict, cfg: dict, debug: bool, st: HudState) -> dict:
    """One frame of the HUD ui contract (every key always present) from a SessionOutput (M2 0.3, duck-typed)."""
    s, fu = cfg["session"], cfg["fusion"]
    tilt = lo = hi = None
    if out.phase == "HOLD":
        yaw = palm_yaw_deg(fr.lm_world) if fr is not None and fr.present else None
        if st.yaw0 is None and yaw is not None:
            st.yaw0, st.tilt_min, st.tilt_max = yaw, 0.0, 0.0
        if st.yaw0 is not None:
            if yaw is not None:
                tilt = (yaw - st.yaw0 + 90.0) % 180.0 - 90.0   # mod 180: no jump edge-on (YAW-FOLD)
                st.tilt_min, st.tilt_max = min(st.tilt_min, tilt), max(st.tilt_max, tilt)
            lo, hi = st.tilt_min, st.tilt_max
    else:
        st.yaw0, st.tilt_min, st.tilt_max = None, 0.0, 0.0
    if out.phase == "RESULT":
        for ev in out.events:
            if ev.kind == "RESULT":
                st.result_debug = ev.data["result"].debug
        dv = st.result_debug
    else:
        st.result_debug = None
        dv = _gesture_values(fr, pose, cfg["gesture"]) if debug else None
    return dict(
        phase=out.phase, t=out.t, t_in_phase=out.t_in_phase, phase_len=out.phase_len, prompt=out.prompt,
        target=out.target, stable_progress=out.stable_progress, first_match=out.first_match, hint=_shown_hint(out, st),
        verdict=out.verdict, reasons=out.reasons, reason_text=out.reason_text, scores=out.scores, total=out.total,
        raw_label=out.raw_label, smooth_label=out.smooth_label, e=out.e, palm_ratio=out.palm_ratio,
        quality_ok=out.quality_ok, fps=stats["fps"], proc_ms=stats["proc_ms"], debug=debug,
        recording=stats["recording"],
        tilt_deg=tilt, tilt_range_deg=None if lo is None else hi - lo,
        tilt_target_deg=TILT_TARGET_K * cfg["planarity"]["min_normal_deg"], debug_values=dv,
        tilt_min_deg=lo, tilt_max_deg=hi,
        protocol=dict(countdown_s=s["countdown_s"], first_match_s=s["first_match_max_s"],
                      challenge_s=s["challenge_timeout_s"], stable_s=s["stable_hold_s"], hold_s=s["hold_s"],
                      result_s=s["result_s"]),
        fusion=dict(weights=dict(planarity=fu["w_planarity"], texture=fu["w_texture"], nonrigid=fu["w_nonrigid"]),
                    floors=dict(planarity=fu["planarity_floor"], texture=fu["texture_floor"]),
                    pass_total=fu["pass_total"]),
        palm_min_ratio=cfg["quality"]["min_palm_frac"])


def draw(img_bgr, fr, pose, out, stats, debug, cfg, st: HudState) -> np.ndarray:
    """ui.draw replacement: hand from the Frame (no hand when fr is None / absent), then render()."""
    ok = fr is not None and bool(fr.present)
    hand = {"present": ok, "n_hands": fr.n_hands if fr is not None else 0,
            "hand_score": fr.hand_score if ok else None,
            "lm_px": fr.lm_px if ok else None, "lm_world": fr.lm_world if ok else None}
    # ponytail: display-only hold - a dropped detection (absent / NaN landmarks) shows the last finite hand for
    # <= HAND_HOLD_S of frame time so the feed grade and hand layer do not blink; build_ui still sees the real fr.
    if ok and fr.lm_px is not None and bool(np.isfinite(fr.lm_px).all()):
        st.last_hand, st.last_t = hand, out.t
    elif st.last_hand is not None and 0.0 <= out.t - st.last_t <= HAND_HOLD_S:
        hand = st.last_hand
    return render(img_bgr, hand, build_ui(out, fr, pose, stats, cfg, debug, st))
