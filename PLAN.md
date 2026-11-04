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

**M3（计划）**：temporal.prerecorded_s 0.3 · match_frames 3 · window_s 1.0 · max_gap_s 0.15 · f_delta 0.5 · mid_band [0.2, 0.8] · max_ext_rate 15 → 41（dev-8，calib）· switch_jump 1.0（brief §5.0/§5.1）；nonrigid.c / s（synth→calib）；planarity.min_pair_dt_s 0.3 · min_normal_deg 12 · max_pairs 10 · min_pairs 3 · tau_hard 0.02 · c 0.04 · s 0.01 · ransac_thresh 0.02 · ransac_iters 200 · ransac_seed 0（brief §5.3，tau_hard/c/s calib）；texture.band [0.15, 0.45] · c / s（synth→calib）· min_blocks 5；fusion.w_planarity 0.65 · w_texture 0.20 · w_nonrigid 0.15 · planarity_floor 0.4 · texture_floor 0.3 · pass_total 0.6（brief §5.6，floors calib）。

**M4（计划）**：session.challenge_len 1 · seq_second_react_max_s 2.5（brief §3.4）。

## ③ 测试清单（测试名 → 一句话断言）

| 模块 | 测试 | 断言 |
|---|---|---|
| liveness/frame.py | test_frame.py::test_crop_clamps_to_border | 贴边中心仍返回 128×128，窗口内容等于平移后的原图块 |
| | ::test_sample_roundtrip | 含无手帧的序列写入再读回逐字段相等 |
| | ::test_palm_width_falls_back_to_p0_p9 | 转腕缩短 P5–P17 时取 P0–P9 |
| config.py | test_config.py::test_default_passes | config.yaml 通过校验 |
| | ::test_missing_leaf | 逐个删除任一叶子键都抛 ConfigError |
| | ::test_bad_type、::test_bad_pair、::test_non_finite、::test_non_finite_pair、::test_yaml_nan_rejected | bool/字符串冒充数字、PAIR 逆序或长度不对、NaN/inf 都抛错 |
| gesture.py | test_gesture.py::test_templates_classify | 6 模板 ×{原样, 镜像} ×10 随机变换（±30°、0.7–1.3 倍、σ=1% 掌宽）100% 正确 |
| | ::test_template_margins | 每个模板的判定量离阈值有余量（PIP、拇指两特征、OK 间距） |
| | ::test_foreshortened_curl_needs_3d | FIST/THUMBS_UP 模板绕 x 转 70° 后 2D 角被压扁成直，靠 3D 下限仍判对；关掉下限则误判（dev-5 回归） |
| | ::test_extension_degree | e ∈ [0,1]，伸直≈1、蜷曲≈0，lm_world 或 lm_px 含 NaN 返回 NONE/NaN |
| | test_smoke_replay.py::test_smoke_replay_accuracy | 本机 smoke 样本按人工真值分段逐帧准确率 ≥ 90%；样本不在本机时整文件 skip（dev-5） |
| | test_gesture.py::test_smoother_* | 0.2 s 窗内多数、并列 → UNKNOWN、丢检 > 50% → NONE、窗口按时间不按帧 |
| camera.py | test_camera.py::test_sample_dir_uses_npz_t_and_no_flip | 样本目录读出的 t 等于 frames.npz 的 t，帧不翻转 |
| | ::test_npz_row_mismatch | npz 行数 ≠ mp4 帧数抛错 |
| | ::test_bare_mp4 | 裸 mp4 的 t = idx / fps |
| hand_tracker.py | test_tracker.py::test_black_frame_no_hand | 全黑 1280×720 → present False、n_hands 0（模型缺失 skip） |
| | （无单独测试，在 ::test_black_frame_no_hand 末尾） | 相同 t 连续调用不报错 |
| | ::test_ensure_model_uses_local_file | 已有模型不下载，补写 .sha256 |
| main.py | test_main.py::test_headless_requires_video | --headless 无 --video → 退出码 2 |
| | ::test_headless_raw_mp4 | 合成 mp4 跑完，逐帧行数 = 帧数，--bench 打印 p50/p95 |
| | ::test_headless_emoji_hand | 表情 ✋ 视频至少一半帧 present（字体缺失 skip） |
| | ::test_save_frames | --save-frames 产出 PNG |
| session.py（M2） | test_session.py | 见 PROMPT.md §0.3 M2 |
| liveness/*（M3） | test_temporal / nonrigid / planarity / texture | 见 PROMPT.md §5.1–5.4 合成用例 |
| liveness.score（M3） | test_score.py | 合成整回合（针孔相机、关节角插值 FIST→V_SIGN、HOLD ±20° 转腕、类皮肤 ROI）：真手 PASS；HOLD 换平面照片 → PLANAR；单帧硬切 → INSTANT_SWITCH；不转腕 → NEED_ROTATION；质量/双手/HOLD_BROKEN 门控与顺序；nonrigid None → LOW_QUALITY；丢检里藏切换时 veto_instant_switch 仍为真；fuse 第 2/3 步；掌心正对 pitch 0 的 2D 阶跃刻画测试（已知局限，dev-8） |
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
- **dev-5 伸直判定换特征（2026-11-03 真手 smoke 证据）**：四指伸直 ⇔ lm_px 上的 2D ∠(MCP,PIP,DIP) > pip_extended_deg **且** lm_world 的 3D 角 > pip3d_min_deg；拇指伸直 ⇔ ‖P4−P6‖/掌宽 > thumb_tip_ratio，不再用腕距比和 cross。证据：真手完全张开时 3D 中指 PIP 中位 137°（2D 177°），160° 在 3D 上过不了，整手读成 UNKNOWN；用户握拳时拇指贴食指外侧不横跨掌面，cross 不触发、腕距比 1.46–2.08，91 帧握拳全读成 THUMBS_UP，e_thumb 在 69/91 帧饱和到 1.0。只用 2D 的问题：蜷曲手指朝镜头时 2D 角被压扁成直（模板 FIST 绕 x 转 70° 读 [159,180,164,150]），所以保留 3D 下限。e 同步改为 e_thumb 由 thumb_tip_ratio 映射、e_finger = min(e2d, e3d)。四个阈值都是 calib，仅凭一人右手 13 s 定的初值，M4 标定；掌心朝镜头的握拳、拇指藏在四指里的握拳还没有真手样本。OPEN_PALM 模板拇指外展方向改为 (-0.85,-0.50,-0.10)。smoke 回放：proc_ms p50 8.7–9.3 / p95 9.0–11.4 ms（两次回放）；新规则逐帧准确率 THUMBS_UP 52/55、FIST 90/91、OPEN_PALM 28/28、V_SIGN 9/9（tests/test_smoke_replay.py，样本只在本机）。第二段录制（2026-11-04，6 手势 × 双手，含拳面/掌心朝镜头两种握拳）：5 个手势双手全对，只有 OK 双手都读成 UNKNOWN——真 OK 环的指尖在 lm_world 里相距 0.37–0.50 倍掌宽（lm_px 上只有 0.08–0.22），所以 OK 的 ‖P4−P8‖ 也改在 lm_px 上算、除以 lm_px 上的 2D ‖P5−P17‖（不是 §5.0 按 max 规则取的像素掌宽），阈值仍 0.3。左手 V 时小指翘起（TIM.P）判 UNKNOWN 属姿势本身；该段里无名指 2D 角 173° 看似伸直、3D 角 105° 才判出蜷曲，是 3D 下限在真手上起作用的实例，但离 110° 只有 5°。
- **dev-6 平面性帧对夹角改用"摆角"，帧对不重复用帧**：§5.3 用掌面法向夹角筛帧对。改为对 lm_world 的刚性掌点 P0、P1、P5、P9、P13、P17 做 Kabsch 求相对旋转 R，摆角 = arccos(R[2,2])，即视线方向的变化；仍与 `planarity.min_normal_deg` 比较。理由：掌面倾斜时，绕光轴的面内转动也会改变法向夹角，但面内转动在图像上只是相似变换、没有视差，这样选出的帧对会让真手看起来是平面（test_image_plane_roll_is_insufficient）。帧对按摆角从大到小贪心选取，每帧最多用一次，避免一帧坏的世界坐标把 10 对都带偏（test_one_glitched_frame_does_not_decide）。待用户确认。
- **dev-7 `nonrigid.min_frames` = 5；nonrigid 为 None 时判 INSUFFICIENT(LOW_QUALITY)**：§5.2 没有最少帧数，也没有自己的原因码；HOLD 内合格帧 < 5 时 r 不可信，返回 None，score() 映射为 LOW_QUALITY，fuse() 不会拿到 None 分数。按现行配置，nonrigid 为 None（HOLD 合格帧 < 5）时 planarity 必然也是 NEED_ROTATION（3 对互不重叠的帧至少要 6 个合格帧，两者用同一掩码），INSUFFICIENT_ORDER 里 LOW_QUALITY 在前，所以显示 LOW_QUALITY：根因是合格帧太少，不是没转腕。待用户确认。
- **dev-8 `temporal.max_ext_rate` 15 → 41（smoke 证据；结果是 30 fps 下 5.1(b) 失效，要用户拍板）**：在 smoke 三段录像的 23 次自然手势过渡上逐个跑 §5.1 门控（假定揭示时刻在前一手势结束前 0.4 s）。默认 15/s 时 16/23 判 INSTANT_SWITCH，只有 4 次通过。每次过渡的 max_ext_rate_observed 分位数 p50 19.1、p99 31.4、最大 31.7 /s。smoke 帧率是 15–30 fps，不是 25–30（233858 一段 30/15 fps 交替；#02、#04、#05 约 15 fps）。按 §5.1 的 M4 规则（真手 99 分位 × 1.3，且不低于 100 分位）取 41；41 下 18/23 通过。
  - **(b) 由帧率决定，不由过渡快慢决定**：e ∈ [0,1]，所以速率 ≤ 1/dt。41 只在帧间隔 < 24.4 ms（约 41 fps 以上）时才可能触发；smoke 最短检测帧间隔 26.5 ms，30 fps 目标下单帧硬切读 ≤ 30 /s，(b) 等于关闭。帧率再高又会误拒真手的平滑过渡：合成 live_round 在 45 fps 下 11/28（10 次只有 (b) 触发）、60 fps 下 25/28（全部只有 (b) 触发）判 INSTANT_SWITCH。真手的 e 本身就是阶跃式变化：23 次过渡里 9 次的最大速率来自单帧 de ≥ 0.85 的跳变，所以 M4 按同一规则重标会落在约 1.3 × fps，重标救不回 (b)。
  - **攻击 C 的代价**：模拟 OPEN_PALM 照片 → 目标照片硬切（评审 attackC 代理），41 下 5.1 放行 468/1500（31%，无抖动）、674/1500（45%，1 px / 1 mm 抖动），15 下 0/1500 与 1/1500；目标 V_SIGN、POINT_UP、OK 最差。满足 (a) 的是 t_match 的目标姿态，不只是起始姿态：真 V_SIGN 帧 74% 有一根手指落在 mid_band（OPEN_PALM 起始帧 0%）。最终判定通常仍是 FAIL：照片在 HOLD 的 ±20° 转腕里会碰到 5.3 PLANAR 硬否决，漏判要靠 S5 ④ 那样的复合逃逸。
  - **41 下真手误拒（18/23 通过）**：① 3 次 OPEN_PALM→THUMBS_UP 判 NO_TRANSITION：手指 2D 角一低于 160° 标签就翻转，t_match 时 2D 角还有 132–154°、e 还在 0.53–0.80，|Δe| < f_delta，F 为空。② 2 次 INSTANT_SWITCH，都由 (a) 判出（速率 14.8、28.6，低于 41）：#02 V_SIGN→FIST（约 15 fps）与 #14 V_SIGN→POINT_UP。中指透视缩短后 2D 角只有 59–70°，e2d 归零，min(e2d, e3d) 把落在 mid_band 的 e3d（0.25–0.30）抹掉了；当时掌面法向偏离光轴 17–58°。不是"中间态在 25–30 fps 下被跳过"。
  - **2D 阶跃由偏航决定，不是"掌心正对 ±5°"**：手指屈曲平面包含视线时，2D PIP 角会在一帧内跳变（合成 FIST→V_SIGN pitch 0：39.9° → 163.0°），e 跳过 mid_band → INSTANT_SWITCH。"正对 ±5°"只是 FIST→V_SIGN、帧相位 0 这一个合成切片（-5.5..+4.0°）；盆地随帧相位、动作时长和起始姿态移动（相位 0.75：-2..+14°；move_s 0.45：+7..+11°；OPEN_PALM 起始、相位 0：→THUMBS_UP -6..0° 与 +11..+18°，→V_SIGN -13..-1° 与 +9..+17°，→POINT_UP -10..-1° 与 +10..+18°，另有零散单点），smoke 里 26 次单帧 2D PIP 跳变 > 100° 发生在掌面法向 16.7–89.3°。只用 e3d 有镜像问题：OPEN_PALM 起始的蜷曲在多数角度变成 NO_TRANSITION，真手 #21 THUMBS_UP→POINT_UP 在 e3d 下失败（食指 e3d 一帧 0.04 → 1.0），smoke 19/23 对 18/23。`tests/test_score.py::test_facing_palm_2d_step_known_limitation` 是断言当前结果的刻画测试（pitch 0、seed 0 → FAIL / INSTANT_SWITCH）。
  - **M4 选项（要用户决定）**：(1) 保持 41，接受 (b) 在 30 fps 下关闭，C 只靠 (a) 与 5.3；(2) (b) 改成按时间窗计速率，属改规格；(3) 采集固定 30 fps，并在 meta 记实测 fps；(4) e 用 min(e2d, e3d) 还是只用 e3d，两边各有一类误拒。回到 15 则 30 fps 下 3/3 真 OPEN_PALM→FIST 被拒。
- **dev-9 首次匹配只认在 t0 或之后结束的目标连续段**：§5.1 line 211 把 t_match 定义为原始标签连续 ≥ 3 个检测帧等于目标的首帧时间，没有限定 t0。liveness/temporal.first_match 忽略整段落在 t0 之前的连续段（倒计时里短暂摆出目标又放下）；跨过 t0 的连续段仍从起点算 t_match，所以揭示时已是目标姿态照样判 PRERECORDED。理由：§5.1 判 PRERECORDED 的依据是"挑战揭示前就已是目标姿态"，揭示时已离开目标姿态不属于这种情况。代价：轮播各手势的回放若在倒计时里碰巧放过目标又换掉，按字面规则判 PRERECORDED，现实现放过，之后仍要过 5.1 连续性、5.3、5.4。session.py 自己的首次匹配计数从揭示那一帧起才开始（倒计时的帧不计），所以整段在 t0 之前的连续段两边都不算；跨过 t0 的连续段 session 从揭示帧起计（只决定超时与能否进 HOLD），first_match 从起点算（→ PRERECORDED）。刻画测试：tests/test_temporal.py::test_target_run_across_reveal_absence_is_not_prerecorded。要用户确认：保持，或回到字面规则。

- **dev-10 真手关键点夹具不再默认入库**：§7.1 让 `tests/fixtures/gestures_real.npz`"提交入库、仅存本机、不推远端"，前提是仓库没有远端。2026-11-04 起 origin 是 GitHub 公开仓库，main 的提交都会被推送，所以把 `tests/fixtures/` 加进 .gitignore，防止 `git add -A` 把关键点推上去；`tests/test_gestures_real.py` 夹具缺失时 skip，行为不变。导出的夹具留在本机工作区、不提交（被忽略的未跟踪文件换分支不会丢，测试在任何分支都会跑）；不要 `git add -f` 到本地分支——切回 main 时 git 会把它从工作区删掉、main 上的真手回归随之 skip，且 `git push --all` 或误合并会把它推上公开远端。待用户确认。
- **dev-11 attack_replay 的 OPEN_PALM 片段从握拳开始**：§7.3 让 6 段回放视频都"张开手 1 s → 做目标手势"，对 OPEN_PALM 这段自相矛盾：视频首帧就是目标，揭示时已匹配，first_match 把跨过 t0 的连续段从起点算，只会判 NO_TRANSITION / PRERECORDED，这几条样本永远"被拦住"，让 B 的通过率虚低（attack_switch 排除 OPEN_PALM 就是同一原因）。docs/calibration.md 改为该段握拳 1 s → 张开手，倒计时时画面里是握拳。备选：回放目标干脆排除 OPEN_PALM、只录 5 段。待用户确认。

## ⑥ M1 分工（agent-team 编队 A）

| 角色 | 模型 | 负责 |
|---|---|---|
| 主会话 | opus | 契约层（liveness/frame.py、config.py、config.yaml）、本计划、规格书、审 diff、提交 |
| builder-gesture | sonnet | gesture_templates.py、gesture.py、tests/test_gesture.py |
| builder-io | sonnet | camera.py、hand_tracker.py、tests/test_camera.py、test_tracker.py、test_frame.py、test_config.py |
| builder-app（第二波） | sonnet | ui.py、main.py、tests/test_main.py |
| verifier | sonnet | pytest、纯度 grep、合成视频与表情手 headless 冒烟、bench |

## ⑦ M2 决策与分工（2026-11-04）

规格书全文在会话临时目录（SPEC_M2.md），要点与任务书外的决定：

- **D1 状态机输出即 UI 契约**：`session.update()` 每帧返回 `SessionOutput`，字段名与另一会话设计的 HUD 的 ui 字典一致（phase / t_in_phase / phase_len / prompt / target / stable_progress / first_match / hint / verdict / reasons / reason_text / scores / total / smooth_label / e / palm_ratio / quality_ok），HUD 以后可以直接替换 ui.py。
- **D2 首次匹配的"连续 ≥ 3 个检测帧"**：present=False 的帧打断连续段（保守解释）；t_match 取连续段首帧。
- **D3 稳定保持**：以 stable_start 起算的经过时间 ≥ 0.5 s 即完成，期间非目标/丢检的累计时长 ≤ 0.1 s 不清零，超过即清零重计。
- **D4 超时回合**：不构造 Sequence、不调 score()；录制仍落盘（t_hold_* 与 hold_match_ratio 为 null，roi 为空）。~~WRONG_GESTURE 只在 5.0 s 超时分支判定；4.0 s 首次匹配超时一律 TIMEOUT。~~（这句已被 D14 取代：两个超时分支都按同一条件判 WRONG_GESTURE——最后 1 s 内 ≥ 50% 的帧是同一个非目标已知手势——否则 TIMEOUT。）
- **D5 目标抽取**：`random.Random(seed).choice(sorted(候选))`；候选为空用全部 6 个；`--target` 固定且不排除。
- **D6 ROI 块由 session 收集**（纯逻辑，只看 ndarray 与时间戳）：[t0, HOLD 结束]、间隔 ≥ 0.1 s、最多 70 块丢最早；离线重放用样本里存的块（`external_roi`）。
- **D7 Sequence.frames 与 frames.npz 相同**：从 ROUND_START 帧到进入 RESULT 的那一帧（含），RESULT 页的帧不录。
- **D8 logs/sessions.jsonl 只在摄像头模式写**；`--video` 重放不写日志。
- **D9 `--video` 结束时 headless 总是打印一行结果 JSON**（没有完成的回合 verdict 为 null），`--bench` 的指标并入同一行。
- **D10 fusion 的第 2/3 步现在就按 §5.6 通用实现（liveness/fusion.py）**，M2 只是分数全为占位 1.0；M3 只替换分数来源。
- **D11 r 键在 COUNTDOWN/CHALLENGE/HOLD 删除本进程刚建的未完成样本目录**（shutil.rmtree），这是"丢弃尚未落盘的回合"的实现；RESULT 里的 x 只改 meta.discard。q / 关窗 / Ctrl-C / 出错 / --max-frames 中途结束时 Recorder.close() 同样删除未完成回合目录（最终整合评审）。

分工：builder A（契约层）→ builder B（session.py）→ builder C1（main/ui/README/Recorder）∥ builder C2（tools/evaluate + roundtrip）→ verifier → critic（fable：状态机语义；opus：录制格式与评测）→ 主会话审 diff、提交。

M2 评审后的补充决定（fable 审状态机、opus 审录制格式与评测，2026-11-04）：
- **D12 首次匹配截止按连续段首帧算**（§5.1 原文：t_match = 连续段首帧时间）：截止时若有进行中的连续段且其首帧 ≤ t0 + 4.0 s，不判超时；规格书 §0.3 第 4 条原写法与任务书冲突，已改。
- **D13 进 HOLD 必须已首次匹配**：稳定保持完成但没有 3 个连续原始匹配帧（标签抖动）时不进 HOLD，最终走超时；否则 M3 的 §5.1 拿不到 t_match。
- **D14 WRONG_GESTURE 两个超时分支都判**：4.0 s 首次匹配超时也按"最后 1 s ≥ 50% 同一非目标已知手势"改原因，不然从头做错手势的人只能看到 "Timeout"。evaluate 两者都归"未到达活体检查"，不影响指标。
- **D15 单列样本按 (类别, variant)**：real/still、attack_replay/preplay、attack_*/handheld 三种单列；attack_replay 的 still/moving 按 §7.3 "各半" 参与通过率与切分。任务书 §7.3 末段"variant ∈ {still,preplay,handheld} 不参与切分"与同节 B 类"still/moving 各半"相抵触，取后者。
- **D16 real 的 n 计入 TIMEOUT / WRONG_GESTURE / 无结果样本**（§7.5 "n = test 半区该类样本数"），都算未通过；攻击类维持排除并单列。
- **D17 split=test 任一判据 n = 0 视为未达标、没有样本退出 1**，避免零证据判"达标"。attack_replay 按 §7.5 只 WARN：未达标或 n = 0 仍退出 0，但照写 failure 报告。
- **D18 `--meta` 可重复给、`--record` 目录名限定六个类别**，防止样本落盘后才发现元数据缺失（data/ 下的样本不可改）。

## ⑧ M3 决策与分工（2026-11-04）

四个检查模块 temporal / nonrigid / planarity / texture 由主会话在独立 worktree 上实现（编队 A：每个模块 builder → 对抗评审 → 修复），接到 M2 的 score() 上；fusion.py 与 quality.py 沿用 M2。

- **S1 score() 自己从关键点重算标签与伸展度**（gesture.analyze），不用录制里的 raw_label，保证换阈值后离线重放可复现。
- **S2 掩码**：质量门控逐帧（quality.frame_ok）；HOLD = [t_hold_start, t_hold_end]；5.2 与 5.3 只用 HOLD ∩ 合格帧；5.4 只用时间戳对应合格帧的 ROI 块（roi_t 与帧 t 同为 float64，逐值相等）。
- **S3 原因码**：5.1 / 5.3 / 5.4 各自返回原因码，加上质量门控的 INSUFFICIENT 码一起交给 fuse()；nonrigid 为 None → LOW_QUALITY（dev-7）。debug 的 veto_prerecorded / veto_instant_switch 取 5.1 全部触发项（temporal 的 fired），veto_planar 取 5.3 的原因，所以 INSUFFICIENT 决定判定、或 5.1 先返回了别的码时，§7.4 的硬否决命中率仍然算得出来。
- **S4 [synth] 值**：nonrigid.c 0.010 / s 0.0015 与 texture.c 100 / s 25 来自合成测试的分离点，标 [synth][calib]，M4 重标。smoke 真手（非挑战录制，只做量级核对）：1.5 s 同标签滑窗的平面性 R 0.04–0.32（一个 V_SIGN 窗 5.17 为关键点抖动），没有 PLANAR；纹理块 m 的中位数 28–31、p99 73–130（c = 100；取自 video.mp4 解码帧，有损压缩，仅作量级参考）；非刚性 r 的 p5 / p50 / p95 ≈ 0.02 / 0.05–0.12 / 0.17–0.21，远高于 c，真手在 5.2 上饱和为 1。
- **S5 已知风险（交给 M4 评测确认）**：① OPEN_PALM 合成 R ≈ tau_hard（strict xfail），真手 3/3 回合 R 0.024–0.036 低于 c、planarity 分低于下限（S9），按 §5.3 规则可能要移出挑战集或单设阈值；② 5.1 (a) 可被"起始姿态本身就有一根手指落在 mid_band"满足，t_match 的目标姿态也能满足（真 V_SIGN 帧 74%）：模拟照片切换在 41 下 31–45% 过 5.1，15 下约 0%（dev-8）；③ 掌宽小时纹理块外沿会混进背景，真手 M 可达 136、已触发 SCREEN 下限（S9）；④ 照片在高噪声（相对掌宽）下 R 可能越过 tau_hard 逃过否决；⑤ dev-8 的两个 5.1 误拒来源：OPEN_PALM→THUMBS_UP 的 NO_TRANSITION（3/23）；min(e2d, e3d) 抹掉 mid_band 内 e3d 的 INSTANT_SWITCH（2/23，掌面法向 17–58°）；⑥ 5.1(b) 由帧率决定：41 在约 41 fps 以下不可能触发，约 45 fps 以上误拒真手平滑过渡（dev-8）。测试：`tests/test_score.py::test_facing_palm_2d_step_known_limitation` 与 `tests/test_temporal.py::test_static_start_pose_in_band_known_limitation` 是刻画测试，替换了原来的两条 strict xfail；`tests/test_planarity.py` 的 OPEN_PALM strict xfail 是 §7.2 之外剩下的唯一一条 xfail，待用户按 §0.2 决定。
- **S6 验证**：全量 pytest 通过；§6 纯度 grep 无输出；smoke 三段 `main.py --video … --headless --target X --bench` 全部退出 0，score_ms 15–61 ms（≤ 200），proc_ms p95 ≈ 9 ms。
- **S7 evaluate 用当前配置重算 raw_label 再回放**：session 状态机（首次匹配、HOLD、HOLD_BROKEN、TIMEOUT）读的是 raw_label，S1 只覆盖 score()。tools/evaluate 在 session.replay 前对每个检测帧用 gesture.classify 重算 raw_label，gestures 准确率也用重算标签；roundtrip 在配置 sha8 不同且标签有变化时把样本单列在 "Label drift (gesture config changed)"，不算 mismatch；sha8 相同而标签有变化时记为 mismatch（录制与回放的分类代码不一致）。局限：录制时用户的动作是对录制时反馈的反应，离线回放复现不了；重算后首次匹配或 HOLD 推迟时，录像在录制时的 RESULT 帧就结束，回放没有结果：roundtrip 记在 Label drift（no result），real 样本按 D16 计未通过，M4 的 [calib] 手势重标可能借这个回放假象压低真手 PASS 率（合成样本 pip_extended_deg 160→165 只改 1 帧标签就出现）。现行处理只求看得见：evaluate 的 WARN 节逐个列出"重标后回放没有 RESULT"的样本，失败报告列出无结果的 real 样本。待用户决定：(a) 重标后仍按录制时的 HOLD 时刻回放；(b) 修订 D16，这类样本不计入 n；(c) 保持现状、只报 WARN。
- **S8 最终整合评审后的决定**：(1) --split 默认 calib；all 与 test 一样，有 test 样本就计入 test_runs.json、满 2 次拒绝；(2) evaluate 报告逐样本打印 §5.6 debug 全集；(3) Recorder 中途退出丢弃未落盘回合；(4) config 校验拒绝 NaN/inf；(5) 恢复两条 §0.2 断言（test_low_quality_count 的精确原因断言；M2 的 RESULT 原因文案断言移植到 tests/test_hud.py）；(6) 测试里的 [calib] 现值改为从 config.yaml 读，冻结新值不再让测试变红；(7) tools.evaluate / tools.calibrate 遇配置错误也只打印 `error: …`（不出 traceback），以退出码 2 结束，与 main.py 一致；权重和先转 float，非 UTF-8 或超过 4300 位的整数也报配置错误；(8) 待用户决定：tests/test_main.py 用 `--video` 跑 main.py 的用例在 models/ 缺模型时会联网下载（c726dd7 起就有），与 PROMPT §2"pytest 不触发下载"冲突；按 §0.2 没有擅自加 skip。
- **S9 首批真摄像头回合（2026-11-04 上午，用户自测；不是 §7.3 样本，不进任何评测表）**：Mac 内置摄像头，检测帧间隔约 0.035 s（≈ 28 fps），`main.py --camera 0 --debug`，未录样本；3 次启动都正常退出，每个结果页写一行 logs/sessions.jsonl。18 回合：PASS 2（POINT_UP 0.977、OK 0.907）、FAIL 7（WRONG_GESTURE 2、PLANAR 2、INSTANT_SWITCH 2、SCREEN 1）、INSUFFICIENT 9（NEED_ROTATION 3、HAND_LOST 2、HOLD_BROKEN 2、NO_TRANSITION 3，一回合同时 HOLD_BROKEN 与 NO_TRANSITION）。① **OPEN_PALM 近平面在真手上成立**：3 个评分回合 R = 0.024 / 0.036 / 0.034，都在 tau_hard 0.02 之上、c 0.04 之下，planarity 分 0.17 / 0.39 / 0.35 全低于下限 0.4，转腕 30–42° 也没拉开；两回合 FAIL(PLANAR) 走的是下限而不是硬否决，一回合被 INSTANT_SWITCH 盖住。其他手势 R 0.09–0.67。② **5.1(a) 误拒真手 3 回合**（THUMBS_UP 被 NEED_ROTATION 盖住；V_SIGN、OPEN_PALM 直接 FAIL）：全部 mid_band_hit = False，观测速率 23.7–30 < 41，不是 (b)；OPEN_PALM 那回合 F = [0] 只有拇指参与过渡；V_SIGN 那回合匹配前 0.33 s 有一次 track switch（5.0），过渡窗被截到 0.33 s。③ **dev-8 的真手证据**：有速率值的 14 回合 max_ext_rate_observed 7.6–30，13/14 超过任务书默认 15，只有 POINT_UP（7.6）不超。④ **纹理把真手判成屏幕**：16 回合 M 17–136，中位数约 70；FIST 81 / 83 / 105 / 121、POINT_UP 122、V_SIGN 136 偏高，OPEN_PALM / THUMBS_UP 17–70。score = 1 − sigmoid((M − 100)/25)，M > 121.2 即低于 texture_floor 0.3：V_SIGN 136 → FAIL(SCREEN)（其余三项都好，total 0.839），POINT_UP 122 → 0.292 被 NO_TRANSITION 盖住，FIST 121 → 0.305。texture.c 100 / s 25 是 [synth] 值，对本机摄像头偏低；指节、指缝直边推高 m 是 5.4 已记的局限，按 M4 规则用 calib 分布重标。⑤ NO_TRANSITION 3 回合：两回合过渡窗内检测帧 < 2，一回合有帧但 F = []；没有录像定不了因，最像"换手势时手出画、摆好再入画"，要用 `--record` 回放确认。⑥ 其他：第一轮 3 回合 NEED_ROTATION 是用户没转腕（摆角 3–8°），之后 24–130°；HAND_LOST 断检 0.235 / 0.336 s；一回合 two_hands_ratio 0.19 逼近 0.2。**阈值一个都没动**：一人、一种光照、18 回合不是标定依据，这些数字只把 S5 ①②、dev-8 和 texture.c 从合成预测变成真手证据，决定权在用户。

分工：每个模块一个 builder（opus），按模块对抗评审（opus，effort max）后修复；score() 接线、config、端到端合成测试由主会话完成，评审另派。


## ⑨ HUD 设计与集成（与 M3 并行）

- **设计过程**：先用 ui-ux-pro-max 定设计系统，再做三个方向（A BIOMETRIC SCANNER、B PRECISION GLASS、C INSTRUMENT TELEMETRY），每个方向出全部状态，由三位评审打分。最终以 C 为底座，借 A 的氛围、B 的文案。之后改了四轮，每轮都有 first-user / 摄像头条件鲁棒性（合成抖动、缩放、分辨率；第 4 轮另用 smoke 录制的真手关键点回放；都没有开实时摄像头）/ code 三路评审和对抗式证伪。设计稿、状态图与验收数字见设计 artifact，工作区存档在 reports/hud_design/（gitignored）。
- **集成方式**：只新增 hud.py。上半部分是纯函数 render(img, hand, ui)，底部是适配器 build_ui / draw / HudState。ui.draw() 改成一行转发，用模块级 HudState（一个进程一个窗口）；main.py 往 stats 里加 recording。SessionOutput 的字段就是 HUD ui 字典的 session 部分（D1），session.py 没有为 HUD 做任何改动。
- **按键**：RESULT 页按 PROMPT §3.1 不显示 "SPACE new round"，显示 X discard sample（录制时）/ R back to start / D / Q；SPACE 只在 IDLE 出现。
- **转腕环只用于显示**：tilt = 掌法线 yaw 相对 HOLD 起点的角度，按 mod 180 取，避免侧掌时翻转 180°；目标 = 1.5 × planarity.min_normal_deg。活体判定只看 5.3 的 Kabsch 摆角，与这个显示无关。
- **显示层去抖（只影响画面，不进判定）**：手丢失不超过 0.15 s 时沿用上一帧的手；提示横幅要持续 0.2 s 才出现，消失时延迟 0.6 s；首次匹配后手势丢了，显示 MATCH LOST，倒计时照走。
- **字体**：正文用 OpenCV 内置的可变字重 sans；数字和标签用系统的 SF Mono / SF Pro，通过 cv2.FontFace 加载，找不到时退回内置 sans。不引入 Pillow / Qt，屏幕文案全部是 ASCII（PROMPT §2、§0.2）。
- **性能**：720p 下 15 个状态各 30 帧，render p50 1.59 ms、p95 2.07 ms（预算 4 ms）。
- **待你验证 [ME]**：真摄像头端到端没有跑过（评审与测试只用录像回放、合成帧和表情手），检查清单见 README「测试」。
- **已知局限**：COUNTDOWN 阶段手的颜色仍随逐帧质量变化；侧掌时掌法线噪声大，转腕范围可能偏宽；箭头跟着原始关键点抖动；个别姿态下左箭头会碰到小指准星。
