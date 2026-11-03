# PLAN — PROMPT.md §0.1 plan-mode 产出

状态：用户 2026-11-03 指示"在本会话直接运行"，视为对本计划的确认；本文件留档，随时可改方向。里程碑范围以 PROMPT.md §0.3 为准。

---

## ① 数据结构（§5.0）

已落地为代码（M1）：`liveness/frame.py`
- `Frame`：t, n_hands, present, lm_px(21,3), lm_world(21,3), hand_score, raw_label, roi_mean, roi_sat_ratio, frame_wh
- 常量：`NONE`、`UNKNOWN`、`GESTURES`（6 个 ID）、`LABELS`
- 几何：`palm_width_px`（max(|P5−P17|, |P0−P9|)）、`palm_center_px`（P0/5/9/13/17 均值）
- ROI：`crop_roi_gray(img_bgr, center, size)`（原分辨率、贴边平移不填充）、`roi_stats`
- 样本读写：`save_sample` / `load_sample`（frames.npz + meta.json；video.mp4 由录制器写）

M2/M3 草案（届时放进 `liveness/frame.py` 或 `liveness/__init__.py`）：

```python
@dataclass
class Sequence:
    frames: list[Frame]          # COUNTDOWN 开始 → HOLD 结束（或超时）
    roi: np.ndarray              # (M,128,128) uint8，[t0, HOLD 结束] 每 ≥100 ms 一块，最多 70
    roi_t: np.ndarray            # (M,) float64
    targets: list[str]
    t0: float
    t_hold_start: float | None
    t_hold_end: float | None
    hold_match_ratio: float | None
    frame_wh: tuple[int, int]
    # t_switch 不存：score() 从 frames 的 lm_px 现算（dev-3）

@dataclass
class LivenessResult:
    verdict: str                 # "PASS" | "FAIL" | "INSUFFICIENT"
    total: float | None
    scores: dict[str, float | None]   # temporal(1/0/None), nonrigid, planarity, texture
    reasons: list[str]
    debug: dict

def score(seq: Sequence, cfg: dict) -> LivenessResult: ...
```

## ② config.yaml 键（默认值 / 单位 / 来源）

来源标记：[brief] 任务书给定 · [synth] 合成推导 · [calib] M4 用 calib 半区校准 · [impl] 实现参数 · [dev-N] 见 ⑤

**M1（已实现，`config.yaml` + `config.py` 严格校验）**

| 键 | 默认 | 单位 | 来源 |
|---|---|---|---|
| camera.width / height | 1280 / 720 | px | brief §2 |
| camera.max_read_failures | 30 | 次 | brief §2 |
| camera.fps_nominal | 30.0 | fps | impl（video.mp4 写入帧率） |
| tracker.model_path / model_url | models/hand_landmarker.task / float16/1 URL | — | brief §2 |
| tracker.running_mode | VIDEO | — | brief §2 |
| tracker.num_hands | 2 | — | brief §5.5 |
| tracker.min_*_confidence ×3 | 0.5 | — | impl（MediaPipe 默认） |
| gesture.pip_extended_deg | 160 | °（2D, lm_px） | brief §4, dev-5, calib |
| gesture.pip3d_min_deg | 110 | °（3D, lm_world） | dev-5, calib |
| gesture.thumb_tip_ratio | 0.75 | ×掌宽 | dev-5, calib |
| gesture.ok_tip_ratio | 0.3 | ×掌宽 | brief §4 |
| gesture.ext_pip_deg | [90, 170] | °（2D） | brief §5.1, dev-5 |
| gesture.ext_pip3d_deg | [100, 140] | °（3D） | dev-5, calib |
| gesture.ext_thumb_ratio | [0.4, 0.9] | ×掌宽 | dev-5, calib |
| gesture.smooth_window_s | 0.2 | s | brief §4 |
| roi.size / sat_level | 128 / 250 | px / 灰度 | brief §5.0 / §5.5 |

**M2（计划）**：session.countdown_s 1.0 · challenge_timeout_s 5.0 · first_match_max_s 4.0 · stable_hold_s 0.5 · stable_gap_tol_s 0.1 · hold_s 1.5 · hold_match_min 0.7 · result_s 3.0 · reveal_lookback_s 0.5 · reveal_e_tol 0.5 · wrong_gesture_window_s 1.0 · wrong_gesture_frac 0.5 · roi_interval_s 0.1 · roi_max_blocks 70（均 brief §3/§5.0）；quality.min_palm_frac 0.08 · roi_mean_min 40 · roi_mean_max 220 · sat_ratio_max 0.10 · hand_score_min 0.7 · qualified_ratio_min 0.6 · qualified_min_frames 15 · two_hands_ratio_max 0.2（均 brief §5.5）。

**M3（计划）**：temporal.prerecorded_s 0.3 · match_frames 3 · window_s 1.0 · max_gap_s 0.15 · f_delta 0.5 · mid_band [0.2, 0.8] · max_ext_rate 15（calib）· switch_jump 1.0（brief §5.0/§5.1）；nonrigid.c / s（synth→calib）；planarity.min_pair_dt_s 0.3 · min_normal_deg 12 · max_pairs 10 · min_pairs 3 · tau_hard 0.02 · c 0.04 · s 0.01 · ransac_thresh 0.02 · ransac_iters 200 · ransac_seed 0（brief §5.3，tau_hard/c/s calib）；texture.band [0.15, 0.45] · c / s（synth→calib）· min_blocks 5；fusion.w_planarity 0.65 · w_texture 0.20 · w_nonrigid 0.15 · planarity_floor 0.4 · texture_floor 0.3 · pass_total 0.6（brief §5.6，floors calib）。

**M4（计划）**：session.challenge_len 1 · seq_second_react_max_s 2.5（brief §3.4）。

## ③ 测试清单（测试名 → 一句话断言）

| 模块 | 测试 | 断言 |
|---|---|---|
| liveness/frame.py | test_frame.py::test_crop_roi_border_shift | 贴边中心仍返回 128×128，窗口内容等于平移后的原图块 |
| | ::test_save_load_roundtrip | 含无手帧的序列写入再读回逐字段相等（NaN 视为相等） |
| | ::test_palm_width_max_rule | 转腕缩短 P5–P17 时取 P0–P9 |
| config.py | test_config.py::test_default_loads | config.yaml 通过校验 |
| | ::test_missing_any_key_fails | 逐个删除任一叶子键都抛 ConfigError |
| | ::test_wrong_type_fails | bool/字符串冒充数字、PAIR 逆序都抛错 |
| gesture.py | test_gesture.py::test_templates_classify | 6 模板 ×{原样, 镜像} ×10 随机变换（±30°、0.7–1.3 倍、σ=1% 掌宽）100% 正确 |
| | ::test_template_margins | 每个模板的判定量离阈值有余量（PIP、拇指两特征、OK 间距） |
| | ::test_foreshortened_curl_needs_3d | FIST/THUMBS_UP 模板绕 x 转 70° 后 2D 角被压扁成直，靠 3D 下限仍判对；关掉下限则误判（dev-5 回归） |
| | ::test_extension_degree | e ∈ [0,1]，伸直≈1、蜷曲≈0，lm_world 或 lm_px 含 NaN 返回 NONE/NaN |
| | test_smoke_replay.py::test_smoke_replay_accuracy | 本机 smoke 样本按人工真值分段逐帧准确率 ≥ 90%；样本不在本机时整文件 skip（dev-5） |
| | ::test_smoother_* | 0.2 s 窗内多数、并列 → UNKNOWN、丢检 > 50% → NONE、窗口按时间不按帧 |
| camera.py | test_camera.py::test_video_dir_uses_npz_t | 样本目录读出的 t 等于 frames.npz 的 t，帧不翻转 |
| | ::test_video_dir_count_mismatch | npz 行数 ≠ mp4 帧数抛错 |
| | ::test_raw_mp4_t_from_fps | 裸 mp4 的 t = idx / fps |
| hand_tracker.py | test_tracker.py::test_black_frame_no_hand | 全黑 1280×720 → present False、n_hands 0（模型缺失 skip） |
| | ::test_timestamps_strictly_increase | 相同 t 连续调用不报错 |
| | ::test_ensure_model_existing | 已有模型不下载，补写 .sha256 |
| main.py | test_main.py::test_headless_requires_video | --headless 无 --video → 退出码 2 |
| | ::test_headless_raw_mp4 | 合成 mp4 跑完，逐帧行数 = 帧数，--bench 打印 p50/p95 |
| | ::test_headless_emoji_hand | 表情 ✋ 视频至少一半帧 present（字体缺失 skip） |
| | ::test_save_frames | --save-frames 产出 PNG |
| session.py（M2） | test_session.py | 见 PROMPT.md §0.3 M2 |
| liveness/*（M3） | test_temporal / nonrigid / planarity / texture / fusion | 见 PROMPT.md §5.1–5.6 合成用例 |
| 回放 | test_roundtrip.py（M2） | 见 PROMPT.md §0.3 M2 |

## ④ 最可能达不到 §7.5 的 3 个风险

1. **真人误拒来自手势/时序规则（FRR 与 INSUFFICIENT）**。拇指规则若把握拳读成伸直，FIST↔THUMBS_UP 互判；e_thumb 用腕距比时握拳≈0.7（合成模板），表情拳头样本里更是饱和到 1.0，FIST→THUMBS_UP 过渡 Δe < 0.5 会落入 NO_TRANSITION。快速比划的运动模糊会丢检，0.15 s 空档就是 HAND_LOST。缓解：dev-1；M1 smoke 录制量出真实分布；M3 必须重定义 e_thumb（把横跨方向 cross 并进去），不能沿用腕距比。
2. **5.3 平面性对真手的分辨余量**。MediaPipe 2D 关键点带手形先验，小角度转腕时真手视差可能被压到接近噪声；OPEN_PALM 本身近平面。表现为真手 NEED_ROTATION 偏多或 R 与照片重叠。缓解：HOLD 提示加转腕幅度可视化；按手势报告 R 分布，必要时按 §5.3 去掉 OPEN_PALM。
3. **回放 B ≤ 20% 大概率达不到**。v1 只有纹理 FFT，手机屏在正常距离下摩尔纹不可见。按任务书如实报告，不回调融合；v2 候选是主动光与高光。

## ⑤ 与 PROMPT.md 的偏差

- **dev-1 拇指判定加一个条件**：拇指伸直 ⇔ |P4−P0|/掌宽 > thumb_ratio **且** cos(P4−P2, P17−P5) < thumb_cross_cos。理由：几何推算显示真实握拳时拇指尖压在食指/中指中节，|P4−P0|/掌宽 ≈ 1.35–1.45，已超过 1.3，单一腕距规则会把 FIST 读成 THUMBS_UP；而折叠拇指一定朝小指侧横跨掌面，方向特征与指尖落点无关。两个阈值都是 calib。M1 debug 层同时显示两项特征，smoke 录制用来核对。（dev-5 已取代其判定，cross 仅保留为调试量）
- **dev-1 的 M1 证据（表情渲染样本，不是真人）**：✊ 45 帧中位数腕距比 1.97、cross +0.55。只用腕距规则时 100% 的帧会把拇指读成伸直，加上 cross 条件后只剩 4%（2 帧误为 THUMBS_UP）。✋ 的拇指判定正确（腕距 1.84、cross −0.92），但中/无名/小指 PIP 只有 144–153°，低于 160°，整手被读成 POINT_UP。后者多半是表情画风所致；真人放松张开时 PIP 是否也低于 160°，由 M1 摄像头核对时记录。
- **dev-2 新增两个共享模块**：`liveness/frame.py`（帧契约、ROI、样本读写，属可移植核心）与 `config.py`（严格校验）；§6 目录树没有列出，属必要的契约层。
- **dev-3** t_switch 由 `score()` 从 frames 现算，不存进 Sequence/录制格式，保证离线可复现。
- **dev-4** `--record` 只能与 `--camera` 同用。
- **dev-5 伸直判定换特征（2026-11-03 真手 smoke 证据）**：四指伸直 ⇔ lm_px 上的 2D ∠(MCP,PIP,DIP) > pip_extended_deg **且** lm_world 的 3D 角 > pip3d_min_deg；拇指伸直 ⇔ ‖P4−P6‖/掌宽 > thumb_tip_ratio，不再用腕距比和 cross。证据：真手完全张开时 3D 中指 PIP 中位 137°（2D 177°），160° 在 3D 上过不了，整手读成 UNKNOWN；用户握拳时拇指贴食指外侧不横跨掌面，cross 不触发、腕距比 1.46–2.08，91 帧握拳全读成 THUMBS_UP，e_thumb 在 69/91 帧饱和到 1.0。只用 2D 的问题：蜷曲手指朝镜头时 2D 角被压扁成直（模板 FIST 绕 x 转 70° 读 [159,180,164,150]），所以保留 3D 下限。e 同步改为 e_thumb 由 thumb_tip_ratio 映射、e_finger = min(e2d, e3d)。四个阈值都是 calib，仅凭一人右手 13 s 定的初值，M4 标定；掌心朝镜头的握拳、拇指藏在四指里的握拳还没有真手样本。OPEN_PALM 模板拇指外展方向改为 (-0.85,-0.50,-0.10)。smoke 回放：proc_ms p50 8.7–9.3 / p95 9.0–11.4 ms（两次回放）；新规则逐帧准确率 THUMBS_UP 52/55、FIST 90/91、OPEN_PALM 28/28、V_SIGN 9/9（tests/test_smoke_replay.py，样本只在本机）。第二段录制（2026-11-04，6 手势 × 双手，含拳面/掌心朝镜头两种握拳）：5 个手势双手全对，只有 OK 双手都读成 UNKNOWN——真 OK 环的指尖在 lm_world 里相距 0.37–0.50 倍掌宽（lm_px 上只有 0.08–0.22），所以 OK 的 ‖P4−P8‖ 也改在 lm_px 上算、按像素掌宽归一，阈值仍 0.3。左手 V 时小指翘起（TIM.P）判 UNKNOWN 属姿势本身；该段里无名指 2D 角 173° 看似伸直、3D 角 105° 才判出蜷曲，是 3D 下限在真手上起作用的实例，但离 110° 只有 5°。

## ⑥ M1 分工（agent-team 编队 A）

| 角色 | 模型 | 负责 |
|---|---|---|
| 主会话 | opus | 契约层（liveness/frame.py、config.py、config.yaml）、本计划、规格书、审 diff、提交 |
| builder-gesture | sonnet | gesture_templates.py、gesture.py、tests/test_gesture.py |
| builder-io | sonnet | camera.py、hand_tracker.py、tests/test_camera.py、test_tracker.py、test_frame.py、test_config.py |
| builder-app（第二波） | sonnet | ui.py、main.py、tests/test_main.py |
| verifier | sonnet | pytest、纯度 grep、合成视频与表情手 headless 冒烟、bench |
