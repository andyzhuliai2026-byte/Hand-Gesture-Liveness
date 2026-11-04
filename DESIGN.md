# DESIGN — 手势挑战-应答式活体检测

规格见 [PROMPT.md](PROMPT.md)，实施计划与偏差见 [PLAN.md](PLAN.md)，安装、运行、录制见 [README.md](README.md)。本文的数字都引自这三份文件、config.yaml 与代码；还没有实测的结果标（待 M4 填写）。

## 1. 目标与范围

- v1 是 macOS 笔记本摄像头上的 Python demo，用来验证算法；最终要跑在手机上（PROMPT §1）。界面随机给一个目标手势，系统判断 (a) 手势对不对、(b) 是不是真人的手。
- **可移植核心** = gesture.py + gesture_templates.py（纯 numpy 的手指拓扑 / 模板常量）+ liveness/：只 import numpy、标准库与彼此，不碰摄像头、UI、cv2、mediapipe（PROMPT §1，§6 纯度 grep 必须无输出）。gesture.py 依赖 gesture_templates（`import gesture_templates as gt`）超出了 PROMPT §1 对核心的定义（gesture.py + liveness/），§6 纯度 grep 也不覆盖它。输入是带时间戳的关键点序列 + 128×128 掌心灰度块，输出 `LivenessResult(verdict, total, scores, reasons, debug)`。
- 不训练模型：除 MediaPipe 自带的手部模型外不用任何学习得到的模型（含 sklearn）；不做身份识别、rPPG、深度/ToF（PROMPT §1）。手势分类与五项检查全是规则 + 几何。
- 移植手机要改的只有帧源（camera.py）、UI（hud.py / ui.py）、模型加载（hand_tracker.py）；gesture.py、gesture_templates.py 与 liveness/（含 dev-2 的契约层 liveness/frame.py）原样搬。config.py 依赖 pyyaml，手机端换成同样严格的校验（缺键、类型错、非有限数、权重和 ≠ 1 都报错）。

## 2. 威胁模型与已知盲区

| 代号 | 攻击 | v1 起作用的检查 | 无效的检查 |
|---|---|---|---|
| A | 打印件 / 屏幕上的静态手势照片 | 5.3 平面性（硬否决 + 权重 0.65）；5.2 辅助 | — |
| B | 手机/平板屏幕回放预录手势视频 | 5.4 纹理；5.1 只在预判播放（PRERECORDED）或序列模式下有效 | 5.3（视频里的手本身在做 3D 运动）、5.2（天然非刚性） |
| C | 挑战出现后实时切换手势照片 | 5.1 时序（INSTANT_SWITCH）；5.3 对其中每张静态照片 | 5.2（切换本身抬高残差） |
| D | 手套 / 假手模型 | 无，v1 不防 | 未评估（§7.3 不录 D） |

- **已知盲区**：圆柱弯曲的打印件（r ≈ 5 cm 时深度弧高约 1.9 cm，接近 OPEN_PALM 的非平面量），5.3 可能漏判。合成测试 `tests/test_planarity.py::test_bent_photo_cylinder_known_limitation` 对应 §7.2 唯一允许的 xfail，但它目前通过：合成的弯曲 FIST 照片在 tau_hard 0.02 下仍全部判 PLANAR，没有余量，tau_hard 下调后才会变成 xfail。5.3 漏判弯曲打印件仍是 PROMPT 的假设（待 M4 填写）。PROMPT 要求评测单列，但 §7.3 规程没有弯曲打印件这一类，现有样本不覆盖（待 M4 填写）。
- **结论的适用范围**：评测结论只对 PROMPT §7.3 定义的攻击做法有效（攻击者不针对算法调整手法）；真手样本若只来自一人，则只代表单人（§7.3 要求条件允许时 real 至少 1/3 来自第二个人）。

## 3. 流程：挑战-应答状态机与 HUD

```
IDLE → COUNTDOWN(1.0 s) → CHALLENGE(≤ 5.0 s) → HOLD(1.5 s) → VERIFYING → RESULT(3.0 s) → IDLE
                               └─ TIMEOUT / WRONG_GESTURE ──────────────→ RESULT（不经 VERIFYING）
```

- session.py 是纯逻辑状态机，所有计时由帧时间戳驱动，`--video --headless` 与 evaluate.py 用同一状态机确定性重放（PROMPT §3、§3.3）。`session.update()` 的逐帧输出 SessionOutput 的字段与 HUD ui 字典的 session 部分一一对应（D1）；帧率、录制状态、转腕角度等显示字段由 hud.py 自己补上。
- 目标抽取排除用户当前已近似摆出的手势：揭示前 0.5 s 内出现过的标签，以及伸展度与模板在所有参与手指上都差 < 0.5 的手势（§3.1，D5）。
- 首次匹配 = 原始标签连续 3 个检测帧等于目标，须在 t0 + 4.0 s 内开始（D2、D12）；稳定保持 = 平滑标签等于目标 0.5 s（累计 ≤ 0.1 s 的中断不清零），须在 t0 + 5.0 s 内完成；进 HOLD 前必须已首次匹配（D13）。超时回合不调 score()（D4）。
- HOLD 总跑满 1.5 s，提示转腕；它是 5.2/5.3 的证据窗口。hold_match_ratio < 0.7 → INSUFFICIENT(HOLD_BROKEN)。VERIFYING 对 [COUNTDOWN 开始, HOLD 结束] 调一次 score()，预算 ≤ 200 ms（smoke 实测 15–61 ms，S6）。
- HUD（hud.py，详见 README "运行"）：顶栏四步 01 READY / 02 MATCH / 03 TURN / 04 RESULT；HOLD 期间的转腕环显示已转过的左右范围，左右总范围达到 18°（1.5 × `planarity.min_normal_deg` 12°）时环上出现青色对勾、卡片显示 KEEP HOLDING；README 要求左右各转 15–20°，是在这之上留的余量；平面性要找到至少 3 对视线方向相差 ≥ 12° 的帧；结果页显示 verdict、原因文案、总分与各项分数条。

## 4. 各项检查

**5.1 挑战-应答时序**（门控项，防 C）。原理：真手从起始姿态到目标姿态要连续地走过中间态，照片切换是硬切。输入：原始逐帧标签与伸展度 e ∈ [0,1]^5（不用平滑标签）；e_finger = min(e2d, e3d)，2D PIP 角 90°→170°、3D 角 100°→140° 线性映射，拇指按 ‖P4−P6‖/掌宽 0.4→0.9 映射（dev-5）。判据：t_r = t_match − t0 < 0.3 s → FAIL(PRERECORDED)；t_match 只认在 t0 或之后结束的目标连续段，整段落在倒计时里的连续段忽略，跨过 t0 的连续段仍从起点算（揭示时已是目标姿态照样 PRERECORDED，dev-9，PLAN ⑤）；过渡窗口 W = [max(t0, t_match − 1.0 s, 最后一次 t_switch), t_match]，t_switch 由 score() 从 lm_px 现算（掌心跳变 > 1.0 × 像素掌宽，dev-3）；W 内相邻检测帧间隔 > 0.15 s → INSUFFICIENT(HAND_LOST)；变化手指 F（|Δe| ≥ 0.5）为空 → INSUFFICIENT(NO_TRANSITION)；连续性要 (a) F 中某指在 W 内出现 e ∈ [0.2, 0.8] 且 (b) 最大速率 ≤ `temporal.max_ext_rate`（现值 41 /s，dev-8；任务书默认 15），否则 FAIL(INSTANT_SWITCH)。局限：(b) 由帧率决定，不由过渡快慢决定：e ∈ [0,1]，速率 ≤ 1/dt，41 只在帧间隔 < 24.4 ms（约 41 fps 以上）时可能触发，30 fps 目标下 (b) 等于关闭；约 45 fps 以上反而误拒真手平滑过渡（合成 45 fps 11/28、60 fps 25/28）；M4 按同一规则重标会落在约 1.3 × fps，救不回 (b)（dev-8，S5 ⑥）。(a) 可被起始姿态满足，t_match 的目标姿态也能满足（真 V_SIGN 帧 74% 有手指在 mid_band）：模拟 OPEN_PALM → 目标照片硬切在 41 下 31–45% 过 5.1，15 下约 0%，V_SIGN、POINT_UP、OK 最差，C 实际靠 (a) 与 5.3 的 PLANAR（S5 ②，`tests/test_temporal.py::test_static_start_pose_in_band_known_limitation` 刻画测试）；对 B 只在预判播放或序列模式下有效。

**5.2 非刚性运动**（弱辅助，权重 0.15，不进任何否决）。原理：照片在面内运动下只做相似变换。输入：HOLD ∩ 质量合格帧的 lm_px 2D 点，少于 5 帧时分数为 None → INSUFFICIENT(LOW_QUALITY)（dev-7）。判据：广义 Procrustes（平移 + 旋转 + 等比缩放）对齐后 r = sqrt(mean_i Var_t) / 像素掌宽中位数，score = sigmoid((r − c)/s)，c = 0.010、s = 0.0015 来自合成数据（S4）。局限：真手生理微颤 0.1–0.3 mm 低于关键点噪声 2–5 px，本项只能区分"刚性静置的照片"与"有自主微调的真手"，对 B、C 无效；smoke 真手 r 的 p5 / p50 / p95 ≈ 0.02 / 0.05–0.12 / 0.17–0.21，远高于 c，真手在 5.2 上饱和为 1（S4）。规则：evaluate 显示真手 vs 照片 AUC < 0.7 时权重置 0、把 0.15 加到 5.3 上并记在本文（待 M4 填写）。

**5.3 平面性**（核心，硬否决 + 权重 0.65，防 A 及 C 中每张照片）。原理：平面上的点两帧之间满足单应性，真实 3D 手只在出平面转动（视差）时不满足；所以帧对必须有出平面转动、且同手势。输入：HOLD 内质量合格、原始标签 == 目标的帧。帧对：t_b − t_a ≥ 0.3 s，摆角 ≥ 12°（dev-6：对 lm_world 的 P0、P1、P5、P9、P13、P17 做 Kabsch，摆角 = arccos(R[2,2])，排除面内转动），按摆角从大到小、每帧最多用一次取至多 10 对；不足 3 对 → INSUFFICIENT(NEED_ROTATION)。判据：每对用全部 21 个 lm_px 点做 Hartley 归一化 + DLT 求 H，e = 重投影 RMS / 帧 a 像素掌宽；R = 各对 e 的中位数；R < `tau_hard` 0.02 → FAIL(PLANAR)；score = sigmoid((R − 0.04)/0.01)。RANSAC 内点比例 ρ（阈值 0.02 × 掌宽、200 次、seed 0）只记录不计分；z 不打分，只报告 z_spread。局限：对 B 无效；OPEN_PALM 近平面，合成数据上真手 R 最小 0.012、中位数 0.016，低于 tau_hard（照片最大 0.009），对应 strict xfail（S5 ①）；弯曲打印件盲区；照片在高噪声下 R 可能越过 tau_hard（S5 ④）；MediaPipe 2D 关键点带手形先验，小角度转腕时视差可能接近噪声（PLAN ④ 2）。smoke 真手 1.5 s 滑窗 R 0.04–0.32（S4）。首批真摄像头回合（PLAN ⑧ S9）：OPEN_PALM 3 回合 R 0.024 / 0.036 / 0.034，全在 tau_hard 之上、c 之下，planarity 分 0.17–0.39 低于下限 0.4；其他手势 R 0.09–0.67。

**5.4 屏幕/翻拍纹理**（权重 0.20，下限 0.3，B 的唯一防线）。原理：屏幕像素栅格与相机采样产生摩尔纹，在中高频环带出现尖峰。输入：[t0, HOLD 结束] 约 10 Hz 采的 128×128 原分辨率灰度块（最多 70 块），只用质量合格帧的块（S2）。判据：去均值 → 2D Hann 窗 → |FFT|²，在 [0.15, 0.45] 周期/像素环带取 m = 最大值 / 均值，M = 各块 m 中位数，score = 1 − sigmoid((M − 100)/25)（c、s 来自合成数据，S4）；合格块 < 5 → INSUFFICIENT(LOW_QUALITY)。局限：手机屏像素间距约 0.055 mm，摄像头在 30 cm 处约 0.27 mm/像素，3–5 个屏幕像素落进 1 个相机像素，手机屏的摩尔纹多半检不到，预期只对平板/显示器或贴近的手机有效；掌宽小时块的外沿混进背景（S5 ③）；score 对 M 是线性 sigmoid，强直边（指缝、块内的手轮廓）会让 m 达到栅格量级，无论 c 取多少分数都约为 0（liveness/texture.py 文档串），重标 texture.c 解决不了，改成 log-M sigmoid 属于改规格。smoke 真手块 m 中位数 28–31、p99 73–130（S4；取自 video.mp4 解码帧，有损压缩，仅作量级参考）。首批真摄像头回合（PLAN ⑧ S9，原分辨率块）：16 回合 M 17–136、中位数约 70，FIST / POINT_UP / V_SIGN 偏高；M > 121.2 即低于 texture_floor，V_SIGN 136 的一回合被判 SCREEN，POINT_UP 122 低于下限被其他原因盖住；texture.c 100 对本机摄像头偏低，等 calib 重标。

**5.5 质量门控**。原理：低质量帧上的关键点和纹理不可信，宁可判 INSUFFICIENT 让用户重试，也不在坏数据上判 PASS 或 FAIL。输入：逐帧像素掌宽、掌心块亮度与饱和度、hand_score、n_hands。逐帧不合格：像素掌宽 < 0.08 × 画面宽、roi_mean < 40 或 > 220、饱和像素占比 > 10%、hand_score < 0.7；不合格帧不进 5.2–5.4。[t0, HOLD 结束] 内合格帧占比 < 60% 或合格帧 < 15 → INSUFFICIENT(LOW_QUALITY)；n_hands ≥ 2 的帧占比 ≥ 20% → INSUFFICIENT(TWO_HANDS)（防"真手举着照片"）。局限：MediaPipe 低于 min_*_confidence（0.5）的帧直接不返回手，只能当无手帧；可用的置信度只有 handedness score。

**5.6 融合与判定**。total = 0.65 × planarity + 0.20 × texture + 0.15 × nonrigid。顺序：① 任一 INSUFFICIENT → INSUFFICIENT；② 任一硬否决（PRERECORDED、INSTANT_SWITCH、PLANAR）→ FAIL；③ planarity ≥ 0.4 且 texture ≥ 0.3 且 total ≥ 0.6 → PASS，否则 FAIL（原因 = 触发下限的项或加权损失最大的项）。**INSUFFICIENT 优先的代价**：按 §7.3 规程录的攻击样本大多在第 ① 步被截住（attack_photo 预期就是 NO_TRANSITION），PLANAR / PRERECORDED 很少成为 verdict；所以 5.1 / 5.3 本身的效果看 evaluate 按 debug 的 veto_* 另算的**硬否决命中率**（§7.4，S3），不受判定顺序影响。INSUFFICIENT 对攻击类算未通过，对真手算误拒的一部分（§7.5 要求 real 的 INSUFFICIENT ≤ ⌊0.1 n⌋）。

## 5. 阈值来源与标定

- config.yaml 每个阈值键带来源标签（模型路径与下载地址除外）：[brief] 任务书给定、[synth] 合成推导、[calib] M4 用 calib 半区校准、[impl] 实现参数、[dev-N] PLAN.md 偏差。config.py 启动时严格校验（缺键、未知键、类型错、非有限数（NaN/inf）、三项权重和 ≠ 1 及少数取值约束都报错；main.py 与两个 tools 都以退出码 2 结束），代码里不许 `cfg.get(key, 默认值)`。任何阈值都不得依据合成数据确定（§7.1），[synth] 值只是 M4 前的占位。
- **"所有活体阈值与此 tracker 配置绑定，改配置需重新标定"**（PROMPT §2；tracker：running_mode VIDEO、num_hands 2、三个 min_*_confidence 0.5，见 config.yaml `tracker:`）。
- M4 规则由 `tools/calibrate.py` 在 calib 半区上计算（只出报告 `reports/calib_<sha8>.md`，不覆盖旧报告、依次加 -2、-3…，从不改 config.yaml）：
  - `temporal.max_ext_rate` = max(1.3 × 真手 99 分位, 真手最大值)，向上取到 0.1（§5.1：不得低于真手 100 分位）。
  - `fusion.planarity_floor` / `texture_floor` = min(到达第 ③ 步的真手最低分, 初值 + 0.2)，向下取 2 位小数（§5.6：calib 真手 FRR = 0 的最大值，初值 0.4 / 0.3）。
  - `planarity.tau_hard` 约束：calib 真手 R 的 5 分位 ≥ 2 × tau_hard；不满足时 §5.3 给的做法是把 OPEN_PALM 移出挑战集或单设阈值；把 tau_hard 降到 ≤ p5/2 是 tools/calibrate.py 额外给的 calib 选项，全局下调会削弱对 A 唯一的否决和对弯曲打印件的余量，采用前先对照 calib 报告第 4 节 attack_photo 的 R 与 `test_bent_photo_cylinder_known_limitation`。都只报告不自动改。
  - `nonrigid.c/s`、`texture.c/s`、`planarity.c/s`：任务书没给公式（planarity 只给了默认值 0.04 / 0.01），六个值都按 calib 分布与 AUC 决定，理由写进 config.yaml 注释。
  - 手势阈值（pip_extended_deg 160、pip3d_min_deg 110、thumb_tip_ratio 0.75、ok_tip_ratio 0.3）在 gestures calib 半区 t ≥ t0 + 1.0 s 的帧上核对余量；`--export-fixture` 导出 §7.1 回归夹具 tests/fixtures/gestures_real.npz（只含关键点）。
- evaluate 回放前用冻结配置对每个检测帧重算 raw_label（PLAN ⑧ S7），session 状态机与 gestures 准确率都跟当前配置走。局限：录制时用户的动作是对录制时反馈的反应，离线回放复现不了；重算后首次匹配或 HOLD 推迟时，录像在录制时的 RESULT 帧就结束，回放没有结果：roundtrip 记在 Label drift（no result），real 样本按 D16 计未通过，M4 的 [calib] 手势重标可能借这个回放假象压低真手 PASS 率（合成样本 pip_extended_deg 160→165 只改 1 帧标签就出现）。WARN 节逐个列出这类样本，怎么计入 n 待用户决定（PLAN ⑧ S7）。
- 冻结：人把值和 calib 依据写进 config.yaml 注释 → commit config.yaml → `.venv/bin/python -m tools.evaluate --split test`。**test 半区最多运行 2 次**，第 2 次前必须在下表记录改了什么及其 calib 依据，两次报告都保留；evaluate 用 `<--out>/test_runs.json` 记账：含 test 样本的 `--split test` 与 `--split all` 都计入，默认的 `--split calib` 从不计数，第 3 次直接拒绝（exit 2）；账本按输出目录记、放在 gitignored 的 reports/ 下，只防误操作，删掉账本就能绕过，真正的约束是这条规程本身。

| 次 | config sha8 | 相对上一次改了什么 | calib 依据 |
|---|---|---|---|
| 1 | （待 M4 填写） | — | （待 M4 填写） |

## 6. 与任务书的偏差（详见 PLAN.md ⑤）

- **dev-1** 拇指判定加横跨方向条件 cos(P4−P2, P17−P5)；已被 dev-5 取代，cross 只保留为调试量。
- **dev-2** 新增契约层 liveness/frame.py（可移植核心）与 config.py（严格校验）。
- **dev-3** t_switch 由 score() 从 frames 现算，不存进 Sequence 与录制格式。
- **dev-4** `--record` 只能与 `--camera` 同用。
- **dev-5** 四指伸直 = 2D PIP 角 > 160° 且 3D 角 > 110°；拇指 = ‖P4−P6‖/掌宽 > 0.75；OK 指尖间距改在 lm_px 上算；e 同步改为 min(e2d, e3d)。
- **dev-6** 平面性帧对用 Kabsch 摆角代替掌面法向夹角，帧对不重复用帧；待用户确认。
- **dev-7** `nonrigid.min_frames` = 5；5.2 分数为 None 时判 INSUFFICIENT(LOW_QUALITY)；待用户确认。
- **dev-8** `temporal.max_ext_rate` 15 → 41（smoke 23 次真手过渡，按 §5.1 规则）。e ∈ [0,1] 使 41 在约 41 fps 以下不可能触发，30 fps 下 (b) 关闭；M4 按同一规则重标会落在约 1.3 × fps。选项（保持 41 且 (b) 关闭；(b) 按时间窗计速率 = 改规格；采集固定 30 fps；min(e2d, e3d) 还是只用 e3d）待用户决定（PLAN ⑤ dev-8）。
- **dev-9** t_match 只认在 t0 或之后结束的目标连续段：倒计时里短暂摆出目标又放下不判 PRERECORDED，跨过 t0 的连续段仍从起点算；待用户确认保持还是回到 §5.1 字面规则（PLAN ⑤ dev-9）。

## 7. 已知风险与局限

- **PLAN ④ 三大风险**：① 真手误拒来自手势/时序规则（拇指互判、快速比划丢检触发 0.15 s 的 HAND_LOST）；② 5.3 对真手的分辨余量（小角度转腕视差接近噪声、OPEN_PALM 近平面）；③ 回放 B ≤ 20% 大概率达不到。
- **PLAN ⑧ S5**：① OPEN_PALM 合成 R ≈ tau_hard；② 5.1 (a) 可被起始姿态和 t_match 的目标姿态满足（41 下模拟照片切换 31–45% 过 5.1）；③ 掌宽小时纹理块混进背景；④ 照片高噪声下 R 可能越过 tau_hard；⑤ 下面 dev-8 的两个 5.1 误拒来源；⑥ 5.1(b) 由帧率决定（见 5.1）。
- **dev-8 的两个 5.1 误拒来源**（smoke 23 次过渡，15–30 fps，41 /s 下 18/23 通过）：OPEN_PALM→THUMBS_UP 时手指 2D 角一低于 160° 标签就翻转，t_match 时 2D 角 132–154°、e 还在 0.53–0.80，|Δe| < f_delta → NO_TRANSITION（3 次）；V_SIGN→FIST（#02，约 15 fps）与 V_SIGN→POINT_UP（#14）中指透视缩短后 2D 角 59–70°，e2d 归零，min(e2d, e3d) 抹掉了 mid_band 内的 e3d（0.25–0.30），掌面法向 17–58°，(a) 不满足 → INSTANT_SWITCH（2 次）。
- **2D PIP 阶跃由偏航决定**：手指屈曲平面包含视线时 2D PIP 角一帧内跳变（合成 pitch 0：39.9° → 163.0°），e 跳过 mid_band → INSTANT_SWITCH。不是"掌心正对 ±5°"：那只是 FIST→V_SIGN、帧相位 0 的一个合成切片，盆地随帧相位、动作时长和起始姿态移动；smoke 里 26 次单帧 2D PIP 跳变 > 100° 发生在掌面法向 16.7–89.3°。只用 e3d 有镜像问题：OPEN_PALM 起始的蜷曲多数角度变成 NO_TRANSITION，真手 #21 THUMBS_UP→POINT_UP 在 e3d 下失败，smoke 19/23 对 18/23。`tests/test_score.py::test_facing_palm_2d_step_known_limitation` 是刻画测试。5.1 是否改用 e3d 要用户决定（PLAN ⑤ dev-8 M4 选项 (4)）。
- **手机屏摩尔纹多半不可见**（见 5.4）：对手机回放 B，5.4 很可能起不了作用，以 evaluate 的 phone 行 AUC 为准（待 M4 填写）。
- **OPEN_PALM 近平面**：按 §5.3 规则，若 calib 上与照片 R 重叠，移出挑战集或单设阈值，并记在本文（待 M4 填写）。真手证据：首批真摄像头 3/3 评分回合 planarity 分低于下限（下一条）。
- **首批真摄像头回合（PLAN ⑧ S9，2026-11-04，18 回合，一人一种光照，不是 §7.3 样本）**：PASS 2 / FAIL 7 / INSUFFICIENT 9。真手被拒的三条路径：OPEN_PALM 平面性分 0.17–0.39 低于下限（R 0.024–0.036）；5.1(a) mid_band_hit = False 判 INSTANT_SWITCH 3 回合（速率 23.7–30 都低于 41，不是 (b)；一回合只有拇指参与过渡，一回合过渡窗被 track switch 截到 0.33 s）；纹理 M 136 判 SCREEN 1 回合、122 低于下限被盖住 1 回合。有速率值的 14 回合 13 回合 max_ext_rate_observed > 15（dev-8）。阈值未动，等 §7.3 calib 样本按规则重标。
- **§7.2 之外剩一个 strict xfail**：`tests/test_planarity.py` 的 OPEN_PALM 用例（§5.3 FAR = FRR = 0 断言本身），§7.2 只允许弯曲照片一条，待用户按 §0.2 决定。原来另外两条已改为断言当前结果的刻画测试，行为一变就失败：`tests/test_score.py::test_facing_palm_2d_step_known_limitation`、`tests/test_temporal.py::test_static_start_pose_in_band_known_limitation`。
- 其余：D 手套/假手不防；多手入镜靠 TWO_HANDS 门控（20%）；按 r 可无限重试，生产环境需要尝试次数锁定；手势阈值初值仅凭一人右手 13 s smoke 定（dev-5）；MediaPipe 的 z 是单目回归值，翻拍照片很可能仍输出"正常"深度。

## 8. v2 候选

- 高光/眩光占比（屏幕玻璃、光面打印件的镜面反射）、主动光一致性（窗口白闪 200 ms 看手部漫反射响应）、HSV 肤色偏移（PROMPT §5.4）。
- z 的去留：v1 不用 z 打分，evaluate 报告 z_spread 分布，由实测决定（待 M4 填写）。
- 序列模式 `challenge_len: 2`（§3.4）：两个互不相同的目标、第二个反应窗 [0.3, 2.5] s、5.1 对两段分别判，30 种有序对迫使回放攻击者现场找到正确的过渡片段，是 v1 约束下对 B 最有效的手段。目前 config.py 只接受 1。

## 9. 评测结果（待 M4 填写）

首批真摄像头自测（2026-11-04 上午，18 回合，不是 §7.3 样本，不进下表）：PASS 2 / FAIL 7 / INSUFFICIENT 9，细节见 §7 与 PLAN.md ⑧ S9。

test 半区（§7.5）。n 的口径：real 的 n = test 半区该类全部样本，TIMEOUT / WRONG_GESTURE / 无结果都算未通过（D16）；攻击类的 n = 到达活体检查的样本，TIMEOUT / WRONG_GESTURE 作为"未到达活体检查"单列、不计入 n（§3、D16），与 tools/evaluate.py 的判定一致。

| 判据 | 目标 | 实际 |
|---|---|---|
| real（variant ≠ still）PASS | ≥ ⌈0.9 n⌉ | （待 M4 填写） |
| real（variant ≠ still）INSUFFICIENT | ≤ ⌊0.1 n⌋ | （待 M4 填写） |
| attack_photo PASS | ≤ ⌊0.1 n⌋ | （待 M4 填写） |
| attack_switch PASS | ≤ ⌊0.1 n⌋ | （待 M4 填写） |
| attack_replay PASS（目标，不达标不算交付失败） | ≤ ⌊0.2 n⌋ | （待 M4 填写） |
| gestures 平滑后逐帧准确率 / 每个手势 | ≥ 95% / ≥ 90% | （待 M4 填写） |
| 只报告：real/still NEED_ROTATION、replay/preplay PRERECORDED、attack_*/handheld TWO_HANDS | 各 ≥ 75% | （待 M4 填写） |

- 各检查项对各类攻击的 AUC、硬否决命中率、attack_replay 按 phone / tablet+monitor 分列的纹理 AUC（手机 < 0.7 如实写这里）：（待 M4 填写）。
- 5.2 与 z 的实测结论：（待 M4 填写）。
- 报告：`reports/eval_test_<sha8>.md`；未达标时 `reports/failure_<sha8>.md`；attack_replay 按 §7.5 只 WARN，未达标或 n = 0 时仍退出 0，但照写失败报告（待 M4 填写）。
- 若 attack_replay 仍 > 20%：按 §5.6 / §7.5 如实写明 **v1 对 B 的防线只有 5.4**，不回调融合、不放宽 5.1。（待 M4 填写）
