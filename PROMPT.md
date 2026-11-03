# Claude Code 任务书：手势挑战-应答式活体检测算法 + macOS Demo

> 用法：在本目录（HandGesture/）启动 `claude`，把本文件全文作为第一条消息发送。项目根 = 本目录。

---

## 0. 工作方式

### 0.1 流程

1. 先进入 plan mode，输出四样东西，我确认后再动手：① §5.0 数据结构的代码草案；② config.yaml 全部键的默认值、单位、来源（任务书给定 / 合成推导 / 待 calib 校准）；③ 每个模块的测试清单（测试名 + 一句话断言）；④ 你认为最可能达不到 §7.5 验收的 3 个风险。里程碑范围已在 §0.3 固定，plan mode 只细化步骤，要调整范围先说明理由等我确认。
2. 每一步只改一个模块及其测试；改完跑 `python -m pytest -q`，全绿再动下一个。gesture.py、session.py、liveness/* 先写测试（至少一条能失败的断言），再实现到通过。
3. 开工先 `git init`；每个里程碑结束 commit 一次。`.gitignore` 包含 data/、models/、reports/、logs/、.venv/。
4. 卡住规则：同一问题连续 2 次尝试未解决（mediapipe 装不上、视频读不了、某项检查在合成数据上分不开真假等），停下汇报现象、已试方案、可选方案，等我决定。
5. 需要我做的事（录样本、给摄像头权限、目测 UI）明确停下来告诉我。

### 0.2 硬约束

- **你不得在自己的 Bash 里打开摄像头或 cv2 窗口**（会卡死会话且触发系统权限弹窗），不得运行 tccutil 或修改任何隐私/系统设置。凡需摄像头或目测的验收，给我一张逐项勾选清单，由我执行并回报；你自己能验证的一律走 `python -m pytest -q` 和 `python main.py --video <样本目录> --headless`。我没有回报的目测项，汇报里写"待你验证"，不得写"已验证"。
- 禁止：修改 §7.5 的验收数字；删除、放宽或 skip/xfail 已有断言（§7.2 明确允许的 xfail 除外）；在 test 半区上调参；用 mock、假数据或硬编码结果让流程"跑通"。确需改动，先停下说明理由。
- 依赖只限 §2 所列。新增任何依赖先问我。模型和依赖只从 pypi.org 和 storage.googleapis.com/mediapipe-models 获取；下载被拦截就停下告诉我，不用第三方镜像，不用别处找来的 .task 文件。
- 不得修改、删除、移动 data/ 下的样本和本任务书。data/ 下的手部视频与关键点属于生物特征数据，只保存在本机，不提交、不上传；唯一例外是 §7.1 的回归夹具（只含 calib 半区关键点，仓库仅存本机、不推远端）。
- 代码中不出现任何 UI 文案以外的中文；屏幕文案一律英文 ASCII（cv2.putText 不支持中文和 emoji）。手势一律用 ID：OPEN_PALM / FIST / THUMBS_UP / V_SIGN / POINT_UP / OK。

### 0.3 里程碑（范围固定）

验收命令前的标签：**[CC]** = 你自己跑；**[ME]** = 我在自己的终端跑，你给清单。

**M1 帧源 + 关键点 + 手势分类**
- 做：环境（§2）、模型下载、camera.py、hand_tracker.py、gesture.py、gesture_templates.py、最小 ui.py（镜像预览、骨架、逐帧/平滑标签、proc_ms、camera fps）、main.py 的 `--camera / --video / --headless / --bench / --debug / --record`（M1 的 --record 从首帧录到按 q 或 `--max-frames`，目录名同 §7.3，只写 frames.npz + video.mp4 + 最小 meta.json（无 targets/t0），挑战字段 M2 补）。
- [CC] `python -m pytest -q` 全绿（手势合成测试 + tracker 冒烟测试，模型缺失时冒烟测试 skip）。
- [ME] `python main.py --camera 0 --debug`：6 手势 × 左右手标签正确（目测；thumb_ratio 的数值核对放 M4）；`python main.py --camera 0 --record data/smoke/` 录一条约 10 s 的真手片段（张开手 → 做几个手势 → 转腕）。
- [CC] `python main.py --video data/smoke/<目录> --headless --bench`：正常退出，逐帧打印 t、present、平滑标签，最后打印 proc_ms 的 p50/p95（目标见 §6：p50 ≤ 35、p95 ≤ 60）。

**M2 状态机 + 质量门控 + 录制格式 + 评测框架（活体分暂用占位，全部 1.0）**
- 做：session.py（§3 全部规则，纯逻辑）、liveness/quality.py、结果画面、完整 `--record`（§7.3 格式）、`--target / --seed / --meta`、tools/evaluate.py（读样本、重放挑战、重算、出表）、logs/sessions.jsonl。M2 的占位版 `score()` 已实现 §5.5 质量门控与 HOLD_BROKEN 判定，只有 5.1–5.4 用占位值（temporal = 1.0，其余分数 = 1.0）。
- [CC] `python -m pytest -q` 全绿，含 tests/test_session.py（用带 raw_label 的合成 Frame 序列驱动：超时、WRONG_GESTURE、保持中 ≤100 ms 中断不清零、目标排除当前手势、hold_match_ratio 事件值、r 重置）与 tests/test_roundtrip.py（录制格式写入 → 读回重算，与阈值无关的 debug 量 allclose(rtol=1e-5)，同一 config 下 verdict/reasons 相等）。
- [CC] `python main.py --video data/smoke/<目录> --headless --target OPEN_PALM`：按 §6 的"meta 缺 targets/t0"规则跑完状态机并打印结果 JSON；任意 verdict 均可，预期 INSUFFICIENT(HOLD_BROKEN)（smoke 片段里 HOLD 期间在换手势）。
- 结束后停下，把 §7.3 的录制命令清单发给我，等我回复"样本录完"。

**M3 五项活体检查 + 融合**
- 做：liveness/temporal.py、nonrigid.py、planarity.py、texture.py、`score()`；每个模块先写合成测试再实现。
- [CC] `python -m pytest -q` 全绿；§6 纯度 grep 无输出；`python -m tools.evaluate --split calib` 打印各类样本各项分数 mean ± std 与 AUC。

**M4 校准、冻结、测试集评测、文档**
- 做：只用 calib 半区按 config 注释里的规则定阈值 → 冻结 config.yaml → 跑 test 半区；README.md、DESIGN.md；可选：`challenge_len: 2` 序列模式（§3.4）。
- [CC] `python -m tools.evaluate --split test`：退出码 0 = 达标；1 = 未达标并已生成 §7.5 失败报告。

### 0.4 里程碑汇报格式（固定 5 段）

① 完成了什么 ② 我要执行的命令/勾选清单 ③ 你实际执行过的命令及原始输出（没执行就写"未执行"） ④ 已知问题 ⑤ 需要我做的事。

---

## 1. 背景与目标

做一个"手势挑战-应答式活体检测"算法。最终要跑在手机上，本阶段只在 macOS 上用笔记本摄像头做 Python demo 验证算法。

两个核心功能：
1. 界面随机显示一个目标手势，用户对着摄像头做出该手势。
2. 系统在用户做手势的全过程中判断：(a) 手势是否正确；(b) 这是否是一只真人的手，而不是照片、屏幕回放、打印件。

**v1 不做**：身份识别；手机移植；深度模型训练（除 MediaPipe 自带模型外不用任何学习得到的模型，含 sklearn）；rPPG；深度/ToF 相机。

**可移植核心** = gesture.py + liveness/：只允许 import numpy 和标准库，不依赖摄像头、UI、cv2、mediapipe。输入是带时间戳的关键点序列 + ROI 灰度块，输出活体结果。这是离线评测和日后移植的边界，不要破坏。

---

## 2. 环境与技术栈（以下事实已在本机核实，照此执行）

- 本机：Apple M4，arm64，macOS 27.2。已有 uv 0.12.3 和 arm64 的 Python 3.12.13（uv 管理）；**没有 3.11**；另有一个 x86_64 的 /usr/local/bin/python3.9，不要碰它。
- 环境：`uv venv --python 3.12 .venv && uv pip install -r requirements.txt`。安装前执行 `.venv/bin/python -c 'import platform; print(platform.machine())'`，必须输出 arm64。所有命令用 `.venv/bin/python`（或激活后的 python）在项目根执行。
- requirements.txt：`mediapipe==1.2.0`、`numpy`、`pyyaml`、`pytest`。**不要再装 opencv-python / opencv-python-headless**：mediapipe 1.2.0 的依赖自带 opencv-contrib-python（实测装入 5.0.0.93）。装完 `uv pip list | grep -i opencv` 只能出现一行。装好后 `uv pip freeze > requirements.lock` 一并交付。遇到 No matching distribution 先查解释器架构；不要换 x86 解释器，不要把 mediapipe 降到 0.10.x。
- MediaPipe（1.2.0 实测）：**`mp.solutions` 已不存在，禁止使用，也不许为了用它降级。** 只用 Tasks API：
  ```python
  import mediapipe as mp
  from mediapipe.tasks import python as mpp
  from mediapipe.tasks.python import vision
  opts = vision.HandLandmarkerOptions(
      base_options=mpp.BaseOptions(model_asset_path="models/hand_landmarker.task"),
      running_mode=vision.RunningMode.VIDEO,
      num_hands=2,                                  # 见 §5.5
      min_hand_detection_confidence=cfg[...], min_hand_presence_confidence=cfg[...], min_tracking_confidence=cfg[...])
  lm = vision.HandLandmarker.create_from_options(opts)
  res = lm.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts_ms)
  # res.handedness / res.hand_landmarks / res.hand_world_landmarks（每只手一个列表）
  ```
  - 摄像头和 `--video` 都走 VIDEO 模式、同一代码路径，不用 LIVE_STREAM（异步回调，不可复现）。
  - `rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)` 得到连续数组；不要用 `frame[:, ::-1]` 之类的视图。
  - `ts_ms` 为 int 且严格递增：`ts_ms = max(int(t * 1000), last_ts + 1)`；按 r 重开不重置；每个 `--video` 输入新建一个 landmarker。
  - 21 点连线按拓扑常量用 cv2.line 自己画（没有 drawing_utils 了）。
  - 启动时的 'Created TensorFlow Lite XNNPACK delegate'、'NORM_RECT without IMAGE_DIMENSIONS' 等日志可忽略。
- 模型文件：pip 包不含模型，API 也不会自动下载。hand_tracker.py 在 `models/hand_landmarker.task` 不存在时用 urllib 从 `https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task` 下载一次（已核实可访问，7,819,105 字节），计算 sha256 写到 `models/hand_landmarker.task.sha256`；下载失败打印等价 curl 命令并以退出码 2 结束。pytest 不触发下载。
- tracker 的 running_mode、num_hands、三个 min_*_confidence 全部写进 config.yaml。DESIGN.md 注明"所有活体阈值与此 tracker 配置绑定，改配置需重新标定"。
- UI 只用 OpenCV 窗口（cv2.imshow），不引入 Qt/Tk/Web/Pillow。
- 摄像头与镜像：`cap = cv2.VideoCapture(idx, cv2.CAP_AVFOUNDATION)`，然后 set 1280×720；读到首帧核对 shape，不是 720p 就打印实际分辨率继续运行。每帧时间戳 `t = time.monotonic()` 在 read() 返回后立刻取。**镜像只做一次**：read() 后立刻 `frame = cv2.flip(frame, 1)`，检测、录制、显示都用这份镜像帧；`--video` 回放时不再翻转。手势规则不依赖 handedness 标签（镜像会反转它），handedness 只取置信度。read() 返回 False 跳过该帧，连续失败 30 次报错退出。

---

## 3. 交互流程（状态机）

```
IDLE → COUNTDOWN(1.0 s) → CHALLENGE(≤ 5.0 s) → HOLD(1.5 s) → VERIFYING → RESULT(3.0 s) → IDLE
                               └─ TIMEOUT / WRONG_GESTURE ──────────────→ RESULT（不经 VERIFYING）
```

所有计时（倒计时、超时、保持、HOLD、结果页）**由帧时间戳驱动**，不用 time.time()/sleep，这样 `--video --headless` 能以任意速度确定性复现。唯一例外是摄像头模式的 t0 取法，见 §3.1。

TIMEOUT / WRONG_GESTURE 回合：verdict = FAIL，reasons = [TIMEOUT 或 WRONG_GESTURE]，scores = {}，total = None，debug = {}；`--record` 仍落盘；evaluate.py 把这类攻击样本单列为"未到达活体检查"，不作为攻击被挡的证据。

### 3.1 各状态

- **IDLE**：显示 "Press SPACE to start"。空格 → COUNTDOWN。RESULT 3 s 后回到 IDLE，不自动开始下一轮。
- **COUNTDOWN**：开始记录帧、重置平滑器；提示 "Show one hand"。
- **CHALLENGE**：
  - 揭示时刻 t0：session.py 在进入 CHALLENGE 的帧输出 REVEAL 事件，t0 暂取该帧的 t；摄像头模式下 main.py 在该帧 `cv2.waitKey` 返回后调用 `session.set_t0(time.monotonic())` 覆盖（macOS 上画面在 waitKey 里刷新）；headless/video 模式不覆盖。
  - 目标抽取：从 6 个手势均匀随机抽，**排除用户当前已近似持有的手势**：揭示前 0.5 s 内出现过的任何已知原始/平滑标签，以及最后一个检测帧的伸展度 e（§5.1）与其模板 e* 在所有参与手指上都满足 |e_i − e*_i| < 0.5 的手势；全被排除则从 6 个里抽。`--target ID` 固定目标（录样本用，不排除）；`--seed N` 固定随机序列。
  - 画面上方大字显示目标 ID + 骨架图标（用 gesture_templates.py 的模板投影画线框）+ 倒计时。
  - 首次匹配（定义见 §5.1）必须在 t0 + 4.0 s 内，否则 FAIL(TIMEOUT)。
  - 稳定保持：平滑标签 == 目标持续 ≥ 0.5 s（期间累计 ≤ 100 ms 的非目标/丢检不清零），且必须在 t0 + 5.0 s 内完成，否则 FAIL(TIMEOUT)；若超时且最后 1.0 s 内 ≥ 50% 的帧平滑标签为同一个非目标已知手势，原因改为 WRONG_GESTURE。
  - 稳定保持完成 → HOLD。
- **HOLD（1.5 s）**：提示 "Hold it. Slowly tilt your hand left and right"。HOLD 总是跑满 1.5 s；session.py 计算并输出 hold_match_ratio（检测到手的帧里原始标签 == 目标的占比），< 70% 时由 `score()` 在 §5.6 第 1 步判 INSUFFICIENT(HOLD_BROKEN)。HOLD 是 5.2/5.3 的证据采集窗口，这个设计是为了让平面性检测有出平面转动可用。
- **VERIFYING**：对 [COUNTDOWN 开始, HOLD 结束] 的序列同步调用 `score()`，不单独显示画面；耗时 ≤ 200 ms。
- **RESULT**：绿色 PASS / 红色 FAIL / 黄色 INSUFFICIENT，第二行显示原因码对应文案（§5.6）。每次进入 RESULT 向 logs/sessions.jsonl 追加一行 {time, targets, verdict, reasons, scores, total, debug, sample_dir}。
- 按键：空格开始；任何状态按 r → 丢弃尚未落盘的回合回 IDLE（RESULT 中已落盘的样本不受影响）；x（仅 `--record` 且在 RESULT 页）→ 当前样本 meta.json 写 discard:true；q 退出；d 切换调试层。README 写明：先点一下预览窗口，按键才会生效。

### 3.2 调试层（d）

实时只显示逐帧可得的量：21 点骨架；原始标签与平滑标签；5 指伸展度 e（§5.1）；掌宽占画面宽比例；质量门控状态；camera fps；proc_ms。各检查项结果在 RESULT 页显示，另附 debug 中间量（§5.6）。CHALLENGE 期间**不**反复调用 score()。

### 3.3 session.py 纯逻辑

状态机写成纯函数/类：输入 Frame（§5.0，含 raw_label）与按键事件，输出状态、提示文案、事件（REVEAL/t0、t_hold_start、t_hold_end、hold_match_ratio、verdict）。不调用摄像头和 UI。evaluate.py 用同一状态机在录制帧上重放挑战；tests/test_session.py 直接构造带 raw_label 的 Frame 序列驱动它。

### 3.4 序列模式（M4 可选）

config `challenge_len ∈ {1, 2}`，默认 1。为 2 时：两个目标互不相同，第一个按 §3.1 排除当前手势；第一个只需稳定保持 0.5 s（无 HOLD），其完成时刻即第二个目标的 t0；第二个目标反应时间窗 [0.3, 2.5] s，完成稳定保持后进 HOLD；5.1 对两段分别计算、任一不过即不过；5.2/5.3/5.4 只用 HOLD 窗口。这是 v1 约束下对回放攻击 B 最有效的手段（30 种有序对，攻击者必须现场找到正确过渡片段），所以 evaluate 支持按 challenge_len 分别出表。

---

## 4. 手势集合与分类（规则法，gesture.py）

全部用 `lm_world`（米）计算。掌宽（世界坐标）= ‖P5 − P17‖。

- 四指（食指–小指）伸直 ⇔ ∠(MCP, PIP, DIP) > 160°。
- 拇指伸直 ⇔ ‖P4 − P0‖ / 掌宽 > `gesture.thumb_ratio`（默认 1.3；M4 用 data/gestures 的 calib 半区核对，改值须在 config 注释写明依据）。
- 判定表（1 = 伸直，0 = 弯曲，* = 不关心；自上而下第一个匹配的胜出，都不匹配 → UNKNOWN）：

| ID | 拇指 | 食指 | 中指 | 无名指 | 小指 | 额外条件 |
|---|---|---|---|---|---|---|
| OK | * | * | 1 | 1 | 1 | ‖P4 − P8‖ < 0.3 × 掌宽 |
| OPEN_PALM | 1 | 1 | 1 | 1 | 1 | — |
| FIST | 0 | 0 | 0 | 0 | 0 | — |
| THUMBS_UP | 1 | 0 | 0 | 0 | 0 | 不判方向 |
| V_SIGN | * | 1 | 1 | 0 | 0 | — |
| POINT_UP | * | 1 | 0 | 0 | 0 | 不判方向 |

- 平滑（只用于 UI 显示和"稳定保持"判定，**不进入 §5.1 的时序判据**）：取最近 0.2 s 内所有帧的原始标签投票，丢检帧投 NONE；某标签票数 > 50% 则输出它，否则 UNKNOWN。
- "左右手都要正确识别" = 同一手势左手、右手都分到同一 ID；不输出、不使用左右手标签。测试：每个模板水平镜像后分类结果不变。
- gesture_templates.py（项目根，UI 图标与测试共用）：6 个手势的 21×3 3D 模板，单位米，按真实手比例（掌宽约 8 cm，手指相对掌面在 z 方向错开 1–4 cm）。

---

## 5. 活体检测算法（核心，liveness/）

**原则：活体必须被正面证明，证据不足 ≠ 通过。** 任一核心项（5.1、5.3）数据不足时结果为 INSUFFICIENT（黄色，提示重试），不是 PASS，也不记为攻击 FAIL。`score()` 总是把能算的分项全算完并写进 debug，verdict 最后统一判定（§5.6），这样 evaluate.py 能看到每个攻击样本在每一项上的分布。

### 5.0 数据约定（录制格式、在线计算、离线评测共用）

Frame（每个摄像头帧一个，不论是否检测到手）：
- `t: float` 秒。摄像头模式 = read() 返回后的 time.monotonic()；视频模式 = 同目录 frames.npz 的 t，没有则 = 帧号 / fps。
- `n_hands: int` 本帧检测到的手数；`present: bool` = 选中的手是否存在（选择规则见下）。
- `lm_px: (21,3) float32` 镜像帧上的像素坐标：x = 归一化 x × W，y = 归一化 y × H，z = MediaPipe z × W。**禁止在归一化坐标上做任何几何计算**（16:9 下各向异性，会把面内旋转算成非刚性）。
- `lm_world: (21,3) float32` MediaPipe world landmarks，米。
- `hand_score: float` = handedness[i][0].score，作为检测置信度。
- `raw_label: str` 本帧原始手势标签（main.py / evaluate.py 用 gesture.py 填；`score()` 不信任它，自行重算）。
- `roi_mean: float`、`roi_sat_ratio: float` 本帧 128×128 掌心灰度块（定义见下）的灰度均值与 ≥ 250 像素占比，**每帧都算**（块本身只按 10 Hz 存），供 §5.5 逐帧门控离线可复现。
- `frame_wh: (int, int)`。
- 无手帧：present=False，lm_* 填 NaN，hand_score / roi_mean / roi_sat_ratio = NaN，raw_label = NONE。
- 多手选择：选掌心离上一帧最近的那只；无上一帧时取 hand_score 最高者。**跟踪切换** = 相邻检测帧掌心位移 > 1.0 × 像素掌宽，或 present 由 False 变 True 时新手掌心与最后已知位置差 > 1.0 × 像素掌宽；切换时刻记为 t_switch，供 §5.1 使用。

ROI 块（约 10 Hz 采样，不逐帧）：从 [t0, HOLD 结束] 每隔 ≥ 100 ms 取一次，最多 70 块，超过丢最早的。在**原始分辨率、不缩放**的镜像帧上，以掌心（P0、P5、P9、P13、P17 像素均值）为中心裁 128×128 灰度块（numpy：0.299R + 0.587G + 0.114B），贴边时平移窗口不填充。存为 `roi: (M,128,128) uint8` + `roi_t: (M,)`。缩放会抹掉摩尔纹，所以不缩放。

Sequence：frames（COUNTDOWN 开始 → HOLD 结束，或超时）、roi、roi_t、targets、t0、t_hold_start、t_hold_end、hold_match_ratio、t_switch 列表、frame_wh。

```python
score(seq: Sequence, cfg: dict) -> LivenessResult(
    verdict: "PASS" | "FAIL" | "INSUFFICIENT",
    total: float | None,            # INSUFFICIENT 或硬否决时为 None
    scores: dict,                   # temporal（1.0 / 0.0 / None = PASS / FAIL / INSUFFICIENT）、nonrigid、planarity、texture ∈ [0,1]，缺数据为 None
    reasons: list[str],             # §5.6 原因码
    debug: dict)                    # §5.6 中间量
```

- 逐帧手势标签与伸展度由 score() 内部调用 gesture.py 重算，不作为输入。
- **像素掌宽**（§5.1–5.5 所有像素量的归一化基准）= max(‖P5 − P17‖, ‖P0 − P9‖)，用 lm_px 的 x、y；取 max 是为了抵消转腕时的透视缩短。
- 掌面法向：lm_world 中 P0、P5、P17 三点平面的法向。

### 威胁模型

- **A** 打印件 / 屏幕上的静态手势照片。
- **B** 手机/平板屏幕回放预录手势视频（动态平面）。**v1 只有部分防护**：5.3 对 B 无效（见下），防线只有 5.4 纹理与序列模式的 5.1 时序。
- **C** 攻击者在挑战出现后实时切换多张手势照片（手机相册滑动）。
- **D** 戴手套 / 假手模型。v1 不防，记为已知局限。
- **已知盲区**：圆柱弯曲的打印件（r ≈ 5 cm 时深度弧高约 1.9 cm，接近 OPEN_PALM 的非平面量），5.3 可能漏判；写入 DESIGN.md，评测单列。

### 5.1 挑战-应答时序（门控项，防 C；对 B 仅在预判播放或序列模式下有效）

输出 ∈ {PASS, FAIL(PRERECORDED), FAIL(INSTANT_SWITCH), INSUFFICIENT(NO_TRANSITION), INSUFFICIENT(HAND_LOST)}。全部基于**原始逐帧输出**，不用平滑标签。

- **首次匹配** t_match = 原始标签连续 ≥ 3 个检测帧等于目标的首帧时间。反应时间 t_r = t_match − t0。t_r < 0.3 s → FAIL(PRERECORDED)（挑战揭示前就已是目标姿态，说明预录/预判）。t_r > 4.0 s 由状态机按 TIMEOUT 处理，不到这里。
- **伸展度** e ∈ [0,1]^5（lm_world）：食指–小指 = PIP 夹角在 90°→170° 间线性映射并截断；拇指 = ‖P4−P0‖/掌宽 在 1.1→1.5 间线性映射并截断（端点写 config）。目标模板 e* 来自 §4 判定表（* 的手指不参与）。
- **过渡窗口** W = [max(t0, t_match − 1.0 s, 最后一次 t_switch), t_match]：§5.0 的跟踪切换时刻之前的检测帧不进入 W。
- **检测空档**：只在 W 内相邻两个检测帧之间计算（不含 W 起点到首个检测帧），间隔 > `temporal.max_gap_s`（默认 0.15）→ INSUFFICIENT(HAND_LOST)，文案 "Keep your hand in view"。
- **变化手指** F = { i ∈ 全部 5 指（含目标模板中为 * 的手指）: |e_i(W 首个检测帧) − e_i(t_match)| ≥ 0.5 }；e* 只用于说明目标，不用于筛 F。F 为空（手在揭示后才入画且已是目标姿态，或揭示时已近似目标）→ INSUFFICIENT(NO_TRANSITION)，文案 "Show your hand before the gesture appears"。
- **连续性**（两条都满足才 PASS，否则 FAIL(INSTANT_SWITCH)）：
  (a) 中间态：至少一个 i ∈ F 在 W 内存在 e_i ∈ [0.2, 0.8] 的检测帧（照片硬切没有中间态）；
  (b) 速率：W 内相邻检测帧 max_i |Δe_i| / Δt ≤ `temporal.max_ext_rate`（默认 15 /s；真人最快的手指伸展约 100–150 ms 走完 0→1，照片硬切在 30 fps 下 ≥ 27 /s）。M4 用 calib 真手样本的 99 分位 × 1.3 校准，不得低于真手 100 分位。
- 以上阈值一律按时间（秒）定义，不按帧数，暗光降帧时含义不变。
- 合成测试：200 ms 内平滑 0→1 的过渡 → PASS；OPEN_PALM→OK 的 200 ms 平滑过渡 → PASS；单帧硬切 → INSTANT_SWITCH；中间丢检 300 ms → HAND_LOST；全程已是目标 → NO_TRANSITION；t_r = 0.1 s → PRERECORDED。

### 5.2 非刚性运动（弱辅助项，权重 0.15）

- 只在 HOLD 窗口内、通过质量门控的帧上算。lm_px 的 2D 像素坐标做广义 Procrustes（相似变换：平移 + 旋转 + 等比缩放）对齐到窗口均值形状；r = sqrt(mean_i Var_t(对齐后 P_i)) / 窗口内合格帧像素掌宽的中位数；score = sigmoid((r − `nonrigid.c`) / `nonrigid.s`)。
- 定位要诚实：真手的生理微颤（0.1–0.3 mm）低于关键点噪声（2–5 px），本项只能区分"刚性静置的照片"与"有自主微调的真手"。对 C 无效（切换本身抬高残差方差），对 B 无效（视频里的手天然非刚性）。倾斜平面产生的是射影变换而非相似变换，其残差由 5.3 负责。
- 不进入任何否决。evaluate 显示真手 vs 照片的 AUC < 0.7 时，权重置 0、把 0.15 加到 5.3 上，并在 DESIGN.md 记录。
- 合成测试：负样本 = 面内运动（静止/平移/面内旋转/缩放）+ 噪声 σ = 0.5% 掌宽；正样本 = 同样运动 + 每根手指独立的 1–3% 掌宽低频摆动；断言正样本 r 中位数 > 负样本 r 的 95 分位。

### 5.3 平面性检测（核心，门控 + 加权 0.65；防 A，及 C 中的每张静态照片）

- 原理：平面上的点在任何相机/平面运动下，两帧之间满足单应性（homography）；真实 3D 手只有在**出平面转动**（视差）时才不满足。纯平移、面内旋转、远近缩放都是相似变换 ⊂ 单应，分不出平面与立体，所以帧对必须有出平面转动——这正是 HOLD 阶段要求转腕的原因。手势切换是非刚性，会抬高残差、让照片看起来像 3D，所以帧对必须同手势标签。
- **注意**：回放视频里的手本身在做 3D 运动，两帧关键点不满足单一单应性，**本项对 B 无效**。
- **合格帧对** (a, b)：都在 HOLD 窗口内；都通过质量门控；两帧原始标签都 == 目标；t_b − t_a ≥ 0.3 s；掌面法向夹角 ≥ `planarity.min_normal_deg`（默认 12°；法向只用来**筛选**帧对，不做证据——即使 MediaPipe 的 z 是幻觉值，它与 2D 外观一致，筛选方向不会错）。按夹角从大到小取至多 10 对。
- 合格帧对 < 3 → INSUFFICIENT(NEED_ROTATION)，文案 "Tilt your hand more and retry"。
- 每对：对**全部 21 个** lm_px 2D 点用 numpy 实现 Hartley 归一化 + DLT 最小二乘估计 H；残差 e = 全部 21 点重投影误差 RMS / 帧 a 的像素掌宽。**禁止用 RANSAC 内点子集上的残差做平面性指标**：掌部 6 点本就近共面，OPEN_PALM 时约 16 点近共面，RANSAC 会把拇指和弯曲指当外点剔掉，让真手看起来是平面。另以阈值 0.02 × 帧 a 的像素掌宽、200 次迭代、固定 seed 跑一次 RANSAC，取内点比例 ρ 作为**只记录不计分**的第二特征（平面 ≈ 1.0，真手通常 ≤ 0.7）。
- R = 各对 e 的中位数。**硬否决**：R < `planarity.tau_hard`（默认 0.02）→ FAIL(PLANAR)。score = sigmoid((R − `planarity.c`) / `planarity.s`)，默认 c = 0.04、s = 0.01。
- 标定约束（M4）：calib 真手 R 的 5 分位 ≥ 2 × tau_hard。evaluate 按目标手势分别报告 R 分布；若 OPEN_PALM 与照片重叠，把 OPEN_PALM 从挑战集移除或单独设阈值，并记录。
- MediaPipe 的 z（图像 z 与 world z）都是单目回归值不是测量值，对翻拍照片很可能仍输出"正常"的深度分布。**v1 不用 z 打分**，只在 evaluate 报表里输出其分布，供 v2 决定去留。
- 合成测试：正样本 = gesture_templates 的 3D 手模板（不用随机点云），针孔相机 f = 900 px、距离 40 cm、绕掌面内轴出平面旋转 15–25°、噪声 σ = 1 px；负样本 = 同模板压平到 z = 0 后做同样运动；各 20 对，断言正样本 R 中位数 > 负样本 R 的 95 分位，且 tau_hard 下 FAR = FRR = 0。另加：(a) 无运动帧对 → 断言输出 INSUFFICIENT 而非低残差；(b) 圆柱面 r = 5 cm 的弯曲照片 → `pytest.xfail`，写入已知局限；(c) 平面样本 + 1 个 20% 掌宽的粗大外点 → 全点残差 e 相对无外点时升高 ≥ 3×，且 ρ = 20/21（±1/21），验证两个特征互补。

### 5.4 屏幕/翻拍纹理（加权 0.20，下限 0.3；B 的唯一防线）

- 只在 VERIFYING 阶段算，输入是 §5.0 的 roi 块。每块：去均值 → 乘 2D Hann 窗 → |FFT|²；在 `texture.band`（默认 [0.15, 0.45] 周期/像素）的环带内取 m = 环带最大值 / 环带均值。M = 各块 m 的中位数；score = 1 − sigmoid((M − `texture.c`) / `texture.s`)。
- **【假设项，M3 用样本验证】** 手机屏像素间距约 0.055 mm，MacBook 摄像头在 30 cm 处约 0.27 mm/像素，3–5 个屏幕像素落进 1 个相机像素，摩尔纹很可能检不到；预期只对平板/显示器或贴近的手机有效。所以 §7.3 的回放样本必须含平板或显示器。evaluate 若显示手机回放 AUC < 0.7，如实写入 DESIGN.md，不要调别的项去"补"。
- 下限见 §5.6 `fusion.texture_floor`。合格 ROI 块 < 5 → INSUFFICIENT(LOW_QUALITY)。
- v2 候选（v1 不实现，列入 DESIGN.md）：高光/眩光占比（屏幕玻璃、光面打印件的镜面反射）、主动光一致性（窗口白闪 200 ms 看手部漫反射响应）、HSV 肤色偏移。
- 合成测试：同一 ROI 叠加周期 4 px、幅度 10 灰度级的正弦光栅后 score 必须下降。

### 5.5 质量门控（quality.py）

逐帧不合格条件（任一即不合格；阈值全在 config.yaml）：像素掌宽 < 0.08 × 画面宽；本帧 roi_mean < 40 或 > 220；本帧 roi_sat_ratio > 10%；hand_score < 0.7。两项 ROI 量用 §5.0 的逐帧字段，不用 10 Hz 的块，保证离线可复现。CHALLENGE/HOLD 期间实时提示 "Move closer" / "More light" / "Too bright"。不合格帧不进入 5.2–5.4。

- [t0, HOLD 结束] 内合格帧占比 < 60% 或合格帧 < 15 → INSUFFICIENT(LOW_QUALITY)。
- **多手**：num_hands = 2。[t0, HOLD 结束] 内 n_hands ≥ 2 的帧占比 ≥ 20% → INSUFFICIENT(TWO_HANDS)，文案 "Show only one hand"（防"真手举着照片"的组合，也避免跟踪在两只手之间跳变）。活体计算只用位置连续的同一只手；跟踪切换的定义与对 §5.1 窗口的影响见 §5.0。
- MediaPipe 的三个 min_*_confidence 是 options 阈值，低于阈值的帧不返回手，按无手帧处理；结果里没有单独的"检测置信度"，只有 handedness score。

### 5.6 融合与判定

权重（config）：planarity 0.65、texture 0.20、nonrigid 0.15；total = Σ w_k × score_k。下限（config）：`fusion.planarity_floor` 初值 0.4、`fusion.texture_floor` 初值 0.3；M4 标定规则：取 calib 真手上 FRR = 0 的最大值，且不高于初值 + 0.2。

判定顺序：
1. 任一 INSUFFICIENT（LOW_QUALITY、TWO_HANDS、HOLD_BROKEN、HAND_LOST、NO_TRANSITION、NEED_ROTATION）→ verdict INSUFFICIENT，total = None。
2. 任一硬否决（PRERECORDED、INSTANT_SWITCH、PLANAR）→ FAIL，total = None。
3. score_planarity ≥ planarity_floor 且 score_texture ≥ texture_floor 且 total ≥ 0.6 → PASS；否则 FAIL：因下限触发时 reasons = 触发下限的那一项（planarity → PLANAR，texture → SCREEN）；因 total 不足时 reasons = 加权损失 w_k × (1 − score_k) 最大的一项（planarity → PLANAR，texture → SCREEN，nonrigid → STATIC）。

注意：INSUFFICIENT 优先于硬否决是刻意的（证据不足既不能当攻击证据，也不能放过）。代价是按 §7.3 规程录的攻击样本大多在第 1 步就被截住，PLANAR / PRERECORDED 很少成为 verdict；所以 evaluate.py 必须另按 debug 重算"硬否决命中率"（§7.4）来评估 5.1 / 5.3 本身。

**若评测后回放 B 仍 > 20%，在 DESIGN.md 写明 v1 对 B 的防线只有 5.4，不得为达标回调融合或放宽 5.1。**

原因码与屏幕文案：TIMEOUT "Timeout" · WRONG_GESTURE "Wrong gesture" · PRERECORDED "Too fast - pre-recorded?" · INSTANT_SWITCH "Suspected photo switch" · PLANAR "Suspected flat photo/screen" · SCREEN "Suspected screen replay" · STATIC "Suspected static image" · LOW_QUALITY "Retry: low quality" · TWO_HANDS "Show only one hand" · HOLD_BROKEN "Hold the gesture while tilting" · HAND_LOST "Keep your hand in view" · NO_TRANSITION "Show your hand before the gesture appears" · NEED_ROTATION "Tilt your hand more and retry"。

`debug` 至少包含：t_r、F、max_ext_rate_observed、gap_max_s、t_switch、r、R、rho、n_pairs、normal_angles、M、n_roi、qualified_ratio、two_hands_ratio、hold_match_ratio。`--headless` 与 evaluate.py 都要打印。

配置：启动时校验 config.yaml 键齐全、类型正确、三项权重之和为 1；代码里禁止 `cfg.get(key, 默认值)` 这类隐藏默认值；测试：删掉任一键启动即报错。每个阈值在 config.yaml 注释里写明来源（任务书给定 / 合成推导 / calib 分位规则）。

---

## 6. 代码结构与命令行

项目根 = HandGesture/，不再嵌套一层。所有命令在项目根执行。

```
main.py              # 入口
camera.py            # 摄像头 / 视频目录 / 裸 mp4 统一帧源，带时间戳，镜像一次
hand_tracker.py      # HandLandmarker 封装 → Frame；模型下载
gesture.py           # 伸展度、规则分类、平滑          （可移植核心）
gesture_templates.py # 6 个手势 3D 模板（UI 图标 + 测试共用）
liveness/            # 可移植核心
  __init__.py        # score(seq, cfg) -> LivenessResult
  temporal.py  nonrigid.py  planarity.py  texture.py  quality.py  fusion.py
session.py           # 状态机（纯逻辑）
ui.py                # OpenCV 绘制
config.yaml
tools/
  __init__.py
  evaluate.py        # python -m tools.evaluate
tests/               # 合成数据 + 一个真实关键点夹具
data/{smoke,real,attack_photo,attack_switch,attack_replay,gestures}/<id>/   # 我录
models/  reports/  logs/   # gitignore
README.md  DESIGN.md  requirements.txt  requirements.lock
```

main.py 参数：`--camera N`、`--video PATH`（样本目录或裸 mp4）、`--record DIR`、`--target ID`、`--seed N`、`--meta k=v ...`、`--headless`（不建窗口，按帧时间戳尽快处理，只能与 --video 一起用；结束打印结果 JSON 一行）、`--bench`（结束打印 proc_ms p50/p95 与 score() 耗时）、`--max-frames N`、`--save-frames DIR`（把带叠加层的帧存 PNG，供你用 Read 检查 UI）、`--debug`、`--config PATH`。

`--video PATH`：视频模式在首帧自动触发 SPACE，逐帧处理、不丢帧，并**对视频重跑 MediaPipe**（用于复现问题）。PATH 是样本目录时：t 取 frames.npz（行数必须等于 video.mp4 帧数，不等则 exit 2）；meta.json 有 targets/t0 时只强制这两项（t_hold_* 由状态机重放得出，与 meta 不一致时打印警告），同时给了 `--target` 则以 `--target` 为准并打印警告；meta 缺 targets/t0（M1 录的 smoke 片段）时按裸 mp4 规则处理。PATH 是裸 mp4 时：必须给 `--target`，t = 帧号 / fps，t0 = 首个 t ≥ 首帧 t + 1.0 s 的帧的 t。

纯度验收（M3 起，在项目根执行，必须无输出）：
```
grep -rnE '^[[:space:]]*(import|from)[[:space:]].*(^|[^A-Za-z0-9_])(cv2|mediapipe|yaml|camera|hand_tracker|ui|session)([^A-Za-z0-9_]|$)' gesture.py liveness/
```

性能目标：`--video <样本目录> --headless --bench` 下 proc_ms（从拿到帧到本帧 UI 数据准备完，含颜色转换、检测、分类、逐帧特征；不含 waitKey、绘制与录制写盘）全程 p50 ≤ 35 ms、p95 ≤ 60 ms（30 帧滑动中位数只用于 UI 显示）；score() ≤ 200 ms；[ME] 正常室内光下窗口 fps ≥ 20。摄像头实际 fps 只显示不考核（暗光自动降帧与代码无关）。v1 不做 GPU/多线程。

---

## 7. 测试、样本、评测与验收

### 7.1 单元测试（pytest；全部 `np.random.default_rng(0)` 固定种子；不依赖摄像头、网络）

- gesture：每个模板 × {原样, 水平镜像} × 10 次随机变换（绕任意轴 ±30°、尺度 0.7–1.3、噪声 σ = 1% 掌宽），分类 100% 正确；平滑序列测试（0.2 s 窗内多数、并列 → UNKNOWN、丢检 → NONE）。
- 每个 liveness 子模块：≥ 1 条方向性断言（真样本分数 > 假样本分数，带固定 margin）+ ≥ 1 条退化输入（空序列、全丢检、合格帧对不足）不抛异常且返回约定值。具体用例见 §5.1–5.4。
- tracker 冒烟：加载模型，对 1280×720 全黑图 detect_for_video 返回 0 只手；模型文件缺失时 skip。
- test_session.py、test_roundtrip.py 见 M2；config 校验测试见 §5.6。
- 真实关键点回归夹具：M2 录完后，把 data/gestures **calib 半区**的关键点（不含 roi/video，约几百 KB）导出为 tests/fixtures/gestures_real.npz 提交入库（§0.2 的唯一例外：仓库仅存本机、不推远端），断言原始标签逐帧准确率 ≥ 95%；夹具缺失时 skip。只靠自造合成数据的测试容易变成同义反复。
- 合成测试只证明实现方向正确；**任何阈值都不得依据合成数据确定**。

### 7.2 允许的 xfail

仅 §5.3 (b) 弯曲照片一条。

### 7.3 样本录制规程（M2 前固定；evaluate.py 只认这个格式）

入口：`python main.py --camera 0 --record data/<类别>/ [--target ID] [--meta k=v ...]`。每次从 COUNTDOWN 到 RESULT 自动落盘 `data/<类别>/<YYYYmmdd-HHMMSS>/`；RESULT 画面按 x 标记作废（meta.json 写 "discard": true，不删文件）。

- `frames.npz`：t、n_hands、present、lm_px、lm_world、hand_score、raw_label、roi_mean、roi_sat_ratio（§5.0），roi、roi_t。
- `meta.json`：{"schema":1, "class", "medium":"none|phone|tablet|monitor|print|na", "variant":"still|moving|preplay|handheld|na", "lighting":"normal|dim|backlit|na", "hand":"left|right|na", "targets":[ID], "t0", "t_hold_start", "t_hold_end", "challenge_len", "frame_wh", "mediapipe_version", "model_sha256", "config_sha8", "live":{verdict,total,scores,reasons,debug}, "discard":false, "notes"}。`--meta` 未给的枚举字段写 "na"。
- `video.mp4`：镜像帧，只供人工回看与 `--video` 重跑；evaluate.py 不读它（有损压缩会改变 ROI 纹理）。
- 录制前（README 与清单都要写）：在摄像头使用中打开 控制中心 → 视频效果，为运行程序的 App 关闭"手势反应 Reactions"（👍 ✌️ 正是它的触发手势，会往画面里加气球烟花）、"人物居中 Center Stage"、"影棚灯光"、"人像"、"背景"。介质上手的像素掌宽 ≥ 画面宽 10%（debug 层显示占比），否则作废重录。**持介质的手不得入画**：用支架/夹子或从画面外持握；debug 层 two_hands_ratio ≥ 0.2 的攻击样本作废重录。磁盘：每条约 1–3 MB，总计 < 300 MB。

类别与最低数量（每类 ≥ 的条数；由我录，条件允许时 real 至少 1/3 来自第二个人）：
- `real` ≥ 20：左右手各半；normal/dim/backlit 都要有；随机挑战；COUNTDOWN 时把张开的手放在画面里，按提示自然转腕。另录 ≥ 4 条"HOLD 期间几乎不动"的（`--meta variant=still`），只用来确认 NEED_ROTATION 路径，不计入 FRR。
- `attack_photo`（A）≥ 8：print 4、phone 4；`--target` 固定；挑战出现前介质不在画面内，出现后 1–3 s 移入并在 HOLD 提示时倾斜 ±20°。预期 verdict 为 INSUFFICIENT(NO_TRANSITION)（§5.6 INSUFFICIENT 优先）；5.3 的有效性看该类 debug.R 分布与硬否决命中率（§7.4）。
- `attack_switch`（C）≥ 12：手机相册 8、平板 4；相册里张开手照片的下一张就是目标手势照片；`--target` 固定且 ≠ OPEN_PALM；COUNTDOWN 时展示张开手照片，挑战出现后滑到目标照片，HOLD 时倾斜介质 ±20°。预期 verdict 为 INSTANT_SWITCH / NO_TRANSITION / HAND_LOST 之一，evaluate 报告三者分布；5.1 (a)(b) 的有效性以合成测试与 debug.max_ext_rate_observed 分布为准，5.3 看硬否决命中率。
- `attack_replay`（B）≥ 16：phone 8、tablet 或 monitor 8；先用手机录 6 段真手视频（张开手 1 s → 做目标手势 → 保持 3 s 并转腕）；`--target` 固定；视频开头的张开手在 COUNTDOWN 时已在画面里，挑战出现后 0.5–2.0 s 内由攻击者**现场**开始播放/寻址对应片段，屏幕亮度最高；still（架住）/ moving（手持晃动，持机的手不入画）各半。另录 ≥ 4 条 phone 在 COUNTDOWN 期间就已开始播放的（预判目标，`--meta variant=preplay`），只用于确认 PRERECORDED 路径，不计入 B 通过率。另录 ≥ 4 条持介质的手故意入画的（`--meta variant=handheld`），只用于确认 TWO_HANDS 路径，不计入通过率。
- `gestures`：6 手势 × 左右手 × 1 条 = 12 条，`--target ID`，挑战出现后 1 s 内做出并保持 ≥ 3 s；t0 + 1.0 s 之后的帧视为该手势，用于分类准确率与 thumb_ratio 核对（M4）。

切分：evaluate.py 先排除 discard:true 的样本，再对每个类别按目录名排序，奇数位 → calib、偶数位 → test，不手工指定。variant ∈ {still, preplay, handheld} 的样本不参与通过率统计与切分计数，各自单列一行并核对其预期原因码。

### 7.4 evaluate.py

- 只读 frames.npz + meta.json，不重跑 MediaPipe；用 session.py 的状态机按 meta 的 targets/t0 重放，再调 score()。roundtrip 核对：meta.config_sha8 == 当前 config sha8 时比较 verdict/reasons/scores；否则只比较与阈值无关的 debug 量（t_r、gap_max_s、R、rho、M、qualified_ratio、two_hands_ratio、hold_match_ratio），allclose(rtol=1e-5)；不一致的样本单独列出。
- `--split calib|test|all`，`--challenge-len 1|2`，`--gestures`（输出 6×6 混淆矩阵与逐手势准确率）。
- 输出表：每类一行，列为 n、PASS/FAIL/INSUFFICIENT 计数、通过率、各分项 mean ± std、各分项对 real 的 AUC（攻击行）、INSUFFICIENT 原因分布、**硬否决命中率**（按 debug 重算 PRERECORDED / INSTANT_SWITCH / PLANAR 各自会触发的样本占比，不受 §5.6 判定顺序影响）、TIMEOUT/WRONG_GESTURE 计数（单列为"未到达活体检查"）；另按目标手势输出 5.3 的 R 分布；表头打印 config.yaml sha256 前 8 位；写入 `reports/eval_<split>_<sha8>.md`。
- 口径：通过率 = PASS / 该类有效样本数（INSUFFICIENT 计为未通过并单列）；FAR(类) = 该类通过率；FRR = 1 − real 通过率。

### 7.5 验收（只在 test 半区判定；n = test 半区该类样本数；退出码 0 = 全部达标，1 = 未达标）

- real（variant ≠ still）：PASS ≥ ⌈0.9 n⌉，INSUFFICIENT ≤ ⌊0.1 n⌋。
- 只报告、不作为达标判据的三行：real(variant=still) 的 INSUFFICIENT(NEED_ROTATION) 占比（预期 ≥ 75%）；attack_replay(variant=preplay) 的 PRERECORDED 命中（预期 ≥ 75%）；attack_*(variant=handheld) 的 TWO_HANDS 占比（预期 ≥ 75%）。低于预期在报告里标 WARN。
- attack_photo：PASS ≤ ⌊0.1 n⌋。
- attack_switch：PASS ≤ ⌊0.1 n⌋。
- attack_replay：PASS ≤ ⌊0.2 n⌋ —— 这是目标，不达标不算交付失败，但必须出失败报告并在 DESIGN.md 如实写明，禁止为达标调融合。
- gestures：平滑后逐帧准确率 ≥ 95% 且每个手势 ≥ 90%。
- 防过拟合：阈值和权重只能用 calib 半区确定；进入 test 前冻结 config.yaml；test 半区最多运行 2 次，第 2 次前必须在 DESIGN.md 记录改了什么及其 calib 依据，两次结果都保留。
- 未达标时生成 `reports/failure_<sha8>.md`，必须包含：① 每个未达标判据的实际值、目标、差距；② 每个误通过的攻击样本和误拒的真手样本：目录名、各分项分数、损失最大的一项；③ 各检查项在 calib 半区 real 对每类攻击的 AUC（< 0.7 视为该特征在当前数据上无效）；④ 不超过 3 条下一步建议，每条写清改什么、预期改善哪个判据、需要我补录什么。只写结论不附证据的，视为未交付。
- 评测结论只对 §7.3 定义的攻击做法有效（攻击者不针对算法调整手法），且真手若只来自我一人则只代表单人；DESIGN.md 写明。

---

## 8. macOS 运行注意（写进 README）

- 摄像头权限归运行 python 的那个 App（Terminal / iTerm / VS Code / Cursor）。首次运行 `cap.isOpened()` 为 False 是预期行为：OpenCV 发出授权请求后立即返回失败。程序打印"Allow camera for this terminal app in System Settings → Privacy & Security → Camera, then rerun"并以退出码 2 结束，不要循环重试。权限由我处理。
- 本机同时有内置摄像头和 iPhone 连续互通相机，index 不保证对应关系；画面来自 iPhone 时改用 `--camera 1` 或在 iPhone 上关闭连续互通相机。窗口标题显示 camera index 与实际分辨率。'AVCaptureDeviceTypeExternal is deprecated' 警告可忽略。
- v1 不使用线程：打开 VideoCapture、读帧、检测、绘制、imshow、waitKey 全部在主线程串行。先 `cv2.namedWindow(WIN)`；每帧 `key = cv2.waitKey(1) & 0xFF`（macOS 上一次约 10 ms，计入帧预算）；首帧之后若 `cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1` 视为窗口已关闭并退出；退出时依次 `cap.release()`、`landmarker.close()`、`cv2.destroyAllWindows()`。
- 异常路径：摄像头打不开、视频读完、模型加载失败 → 打印明确错误，非 0 退出码，不得静默重试或吞异常（这是 --headless 冒烟能判定成败的前提）。
- 视频效果（Reactions / Center Stage 等）必须关闭，见 §7.3。

---

## 9. 交付物

1. 可运行代码 + README（安装、运行、按键、配置、录制规程、权限与视频效果说明）；我在新终端按 README 逐条复制执行能跑通。
2. `python -m pytest -q` 的完整输出贴在 M4 汇报里。
3. `reports/eval_test_<sha8>.md`；未达标时另附 `reports/failure_<sha8>.md`。
4. DESIGN.md ≤ 150 行：算法原理；各检查项防哪类攻击、对哪类无效；已知局限（B 部分防护、弯曲打印件、D 手套/假手、单人数据、多手入镜、按 r 无限重试——生产环境需尝试次数锁定）；z 特征与 5.2 的实测结论；所有阈值与 tracker 配置绑定；移植手机要改的地方（帧源、UI、模型加载；gesture.py 与 liveness/ 原样搬）。
5. requirements.lock、config.yaml（每个阈值带来源注释）。
