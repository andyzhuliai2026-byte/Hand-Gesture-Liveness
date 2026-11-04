# HandLiveness — 手势挑战-应答式活体检测 demo（macOS）

当前进度：**M3 代码完成**（挑战状态机、质量门控、按回合录制、评测框架，活体分项 5.1 时序、5.2 非刚性、5.3 平面性、5.4 纹理全部实现），M4 中不依赖样本的部分（tools/calibrate.py、评测报告补充、DESIGN.md 中不依赖实测的章节）也已完成，可选的 `challenge_len: 2` 序列模式没有做。M3 验收的 calib 评测要等 §7.3 样本。待你决定：OPEN_PALM 的 strict xfail、dev-6～dev-9（PLAN.md ⑤；dev-10、dev-11 已确认）、重标后回放没有结果的样本怎么计（PLAN.md ⑧ S7）、测试缺模型时是否联网下载（PLAN.md ⑧ S8 (8)）。阈值仍是任务书默认值或由合成数据/smoke 录像推出的初值，标定在 M4 做。规格见 [PROMPT.md](PROMPT.md)，实施计划与偏差记录见 [PLAN.md](PLAN.md)。

## 安装

需要 Apple Silicon（arm64）与 [uv](https://docs.astral.sh/uv/)。在项目根执行：

```bash
uv venv --python 3.12 .venv
```

```bash
uv pip install --python .venv/bin/python -r requirements.lock
```

模型 `models/hand_landmarker.task` 在首次运行时自动下载（约 7.8 MB，来自 storage.googleapis.com/mediapipe-models）。

## 运行

实时摄像头，带调试层：

```bash
.venv/bin/python main.py --camera 0 --debug
```

按键（先用鼠标点一下预览窗口，按键才会生效）：

- `空格`：开始一回合（倒计时 1 s → 揭示目标手势 → 稳定保持 0.5 s → 保持并转腕 1.5 s → 结果页 3 s → 回到待机）
- `r`：丢弃尚未落盘的回合，回到待机（结果页里按 r 只是提前回到待机，已落盘的样本不受影响）
- `x`：仅 `--record` 且在结果页：把刚落盘的样本标记作废（meta.json 写 `"discard": true`，不删文件）
- `q` 退出，`d` 开关调试层

转腕要够幅度：HOLD 的 1.5 s 里手掌左右各转 15–20°（像拧门把手），平面性检测要找到至少 3 对视线方向相差 ≥ 12° 的帧。左右总范围达到 18°（1.5 × `planarity.min_normal_deg` 12°）时，HUD 的转腕环出现青色对勾、上方卡片显示 KEEP HOLDING，左右各转 15–20° 是留了余量。不转或只在画面内旋转会得到 INSUFFICIENT（Tilt your hand more and retry）。揭示目标后再开始换手势，太早（< 0.3 s）会被判预录。

界面（hud.py，只用 OpenCV 绘制）：顶栏是四步进度 01 READY / 02 MATCH / 03 TURN / 04 RESULT 与摄像头帧率，录制时亮红色 REC。挑战期间上方卡片给出目标手势的线框图标和名称、挑战剩余秒数的环形倒计时；首次匹配前显示 MATCH IN x.x s；匹配后手势仍在时显示 MATCHED，手势丢了显示 MATCH LOST，照提示重新摆出即可（倒计时不停）。HOLD 期间手下方的转腕环显示已转过的左右范围：没转够时两端有琥珀色箭头，卡片显示 TURN LEFT / RIGHT 和 RANGE n / 18 deg（左右总范围）；总范围到 18° 后显示 KEEP HOLDING。结果页右侧显示 PASS / FAIL / INSUFFICIENT、原因文案、总分和各项分数条（3D shape、Skin texture、Natural motion、Response timing）。底栏是当前提示与按键说明。琥珀色只用来提示需要你做什么。

调试层（`d`）：挑战阶段显示原始/平滑标签、每根手指的伸展度 e（T I M R P）、2D/3D 近节夹角、拇指与 OK 指尖特征、手部置信度、ROI 亮度与饱和比例、质量门控提示、处理耗时；结果页改为显示 score() 的 debug 中间量（t_r、F、max_ext_rate_observed、R、rho、n_pairs、M、r、veto_* 等）。

参数：

- `--target ID`：固定目标手势（`OPEN_PALM FIST THUMBS_UP V_SIGN POINT_UP OK` 之一），录攻击样本时用；不给则随机抽取，并排除你当前已近似摆出的手势
- `--seed N`：固定随机目标序列
- `--meta k=v ...`：写进样本 meta.json 的标注，只接受 `medium=none|phone|tablet|monitor|print|na`、`variant=still|moving|preplay|handheld|na`、`lighting=normal|dim|backlit|na`、`hand=left|right|na`、`notes=<文字>`，未给的写 `na`（notes 写空串），其他键或值直接报错退出

每次进入结果页，都会向 `logs/sessions.jsonl` 追加一行 JSON：时间、目标、verdict、原因码、分项分数、总分、debug、样本目录（未录制为 null）。只有摄像头模式写这个文件，`--video` 不写。

离线回放样本（不开窗口，逐帧打印 `t present raw smooth phase target`，最后一行打印 JSON：最后一个完成回合的结果、帧数、最终状态；加 `--bench` 再附处理耗时与 score 耗时）：

```bash
.venv/bin/python main.py --video data/real/<时间戳> --headless --bench
```

回放带 `targets`/`t0` 的样本目录时，自动按 meta 的目标与 t0 重放；裸 mp4 或 M1 的 smoke 录像必须给 `--target`：

```bash
.venv/bin/python main.py --video data/smoke/<时间戳> --headless --target OPEN_PALM
```

## 配置

所有阈值与可调参数都在 `config.yaml`，每个阈值键的注释带来源标签：[brief N] 任务书给定（N 是 PROMPT 节号）、[synth] 由合成数据推出、[calib] 待 M4 用 calib 半区标定、[impl] 实现参数、[dev-N] PLAN.md 的偏差。活体阈值与 `tracker:` 配置绑定，改 tracker 要重新标定。config.py 启动时严格校验：缺键、未知键、类型错、非有限数（NaN / inf）、三项融合权重和 ≠ 1，以及少数取值约束（各时长与 sigmoid 尺度 s > 0、min_pairs ≤ max_pairs、num_hands ≥ 1 等；不是每个键都查范围）都直接报错。main.py、tools.evaluate、tools.calibrate 遇到配置错误都只打印 `error: …`（不出 traceback），以退出码 2 结束。`--config PATH` 改用另一份配置文件（录制时 meta.json 记它的 sha8）。

另有两个调试参数，多用于回放：`--max-frames N` 处理 N 帧后停止；`--save-frames DIR` 把每帧画好界面的画面存成 `DIR/000000.png`、`000001.png`…（摄像头模式同样会存，真手画面属于生物特征数据，只留在本机，不要提交）。

## 录制样本

规程见 PROMPT.md §7.3，逐步操作清单见 [docs/calibration.md](docs/calibration.md)。每回合从倒计时到结果页自动落盘到 `data/<类别>/<YYYYmmdd-HHMMSS>/`（frames.npz、meta.json、video.mp4）；中途按 `r` 丢弃、或按 `q` / 关窗 / Ctrl-C 退出的回合不落盘（出错退出、`--max-frames` 用完时也一样），未完成的回合目录直接删除。

录制前先关闭 macOS 视频效果（摄像头使用中打开 控制中心 → 视频效果，为终端 App 关闭"手势反应"、"人物居中"、"影棚灯光"、"人像"、"背景"）。

类别与最低条数：

| 类别 | 条数 | 要点 |
|---|---|---|
| `real` | ≥ 20 | 左右手各半，normal/dim/backlit 都要有，随机挑战；COUNTDOWN 时张开的手已在画面里，HOLD 按提示自然转腕；条件允许时至少 1/3 来自第二个人；另录 ≥ 4 条 HOLD 几乎不动的（`variant=still`），不计入 FRR |
| `attack_photo` | ≥ 8 | print 4、phone 4；`--target` 固定；挑战出现后 1–3 s 移入介质，HOLD 时倾斜 ±20° |
| `attack_switch` | ≥ 12 | 手机相册 8、平板 4；`--target` 固定且不是 OPEN_PALM；相册里张开手照片的下一张就是目标照片；COUNTDOWN 时展示张开手照片，挑战出现后滑到目标照片，HOLD 时倾斜介质 ±20° |
| `attack_replay` | ≥ 16 | phone 8、tablet/monitor 8，still/moving 各半；`--target` 固定。先用手机录 6 段真手视频：张开手 1 s → 目标手势 → 保持 3 s 并转腕（OPEN_PALM 那段从握拳开始，PLAN.md dev-11）。视频首帧的手在 COUNTDOWN 时已在画面里，挑战出现后 0.5–2.0 s 内现场开始播放对应片段，屏幕亮度最高。另录 ≥ 4 条 phone 的 `variant=preplay`（预判目标：按空格前约 1 s 就开始播放，揭示时视频里已是目标手势）、≥ 4 条 `variant=handheld`（持介质的手故意入画） |
| `gestures` | 12 | 6 手势 × 左右手，`--target ID`，挑战出现后 1 s 内做出并保持 ≥ 3 s；同一手势的左右手**相邻**录，逐个手势轮换先后（第 1、3、5 个先左后右，第 2、4、6 个先右后左） |

`--record` 的目录名必须是 `real`、`attack_photo`、`attack_switch`、`attack_replay`、`gestures`、`smoke` 之一，拼错直接报错退出。

切分规则（evaluate 自动做，不手工指定）：每个类别按目录名（即录制时间）排序，奇数位 → calib、偶数位 → test。所以：

- `gestures` 同一手势的左右手相邻录，两条正好一条进 calib、一条进 test；逐个手势轮换先后，否则 calib 半区只有一只手；
- `real` 按同一只手两条一组录（左、左、右、右），不要左、右交替：奇数位全是左手，calib 半区会只剩左手；
- 某条按 `x` 作废后**立即重录同一条**（同一手势、同一只手、同一介质），不要跳到下一条再回头补，否则后面所有样本的奇偶位都会错开、配对被打乱。

只有这几种 (类别, variant) 单列、不参与切分和通过率：`real` 的 `still`、`attack_replay` 的 `preplay`、`attack_*` 的 `handheld`。`attack_replay` 的 `still` 和 `moving` 都正常参与切分与通过率，`real` 的 `moving` 也一样。

示例（录一条手机照片攻击）：

```bash
.venv/bin/python main.py --camera 0 --record data/attack_photo/ --target V_SIGN --meta medium=phone variant=moving lighting=normal hand=right
```

介质上手的像素掌宽要 ≥ 画面宽 10%（调试层显示占比），否则按 `x` 作废重录。持介质的手不得入画（用支架/夹子或从画面外持握）；调试层结果页 two_hands_ratio ≥ 0.2 的攻击样本按 `x` 作废重录，`variant=handheld` 除外（这类样本就是要让持介质的手入画）。磁盘：每条约 1–3 MB，总计 < 300 MB。

## 评测

在录好的样本上重放状态机并出表（只读 frames.npz + meta.json，不重跑 MediaPipe），报告写到 `reports/eval_<split>_<sha8>.md`；同一 split 与 sha8 已有报告时不覆盖，依次写 `-2`、`-3`…，实际路径打印在 stderr（`report -> …`）：

```bash
.venv/bin/python -m tools.evaluate --split calib
```

`--split calib|test|all` 选半区（默认 calib；目录名排序后奇数位 calib、偶数位 test，all 两半都含），`--gestures` 另出 6×6 手势混淆矩阵；`--split test` 按 PROMPT.md §7.5 判定，未达标时退出码 1 并写 `reports/failure_<sha8>.md`（同样不覆盖旧报告，路径打印在 stderr）。例外是 attack_replay：§7.5 说它只是目标、不达标不算交付失败，所以只记 WARN，未达标或 n = 0 仍退出 0，但照写 failure_<sha8>.md。

报告除各类别的判定表外，还有 FAIL 原因分布、attack_replay 按 phone / tablet+monitor 分列的行（只报告，不进 §7.5 判据）、按目标手势分列的平面性残差 R，max_ext_rate_observed、R、M、r、z_spread 的分布，以及 `## Debug per sample (5.6)`：所选半区里 gestures 以外的每个样本一行（含 still / preplay / handheld 单列样本），列出 §5.6 的 15 个 debug 量（t_r、F、max_ext_rate_observed、gap_max_s、t_switch、r、R、rho、n_pairs、normal_angles、M、n_roi、qualified_ratio、two_hands_ratio、hold_match_ratio）。

**test 半区最多跑 2 次**（§7.5）：`--split test` 与 `--split all` 只要含 test 样本，都记进 `reports/test_runs.json`，已有 2 条时第 3 次直接拒绝（退出码 2，不写报告）；`--split calib` 从不计数。所以冻结 config.yaml 之前不要跑 `--split all`。第 2 次之前先在 DESIGN.md 的表里写清改了什么、依据哪份 calib 报告。

## 标定（M4）

样本录完后，在 calib 半区上出标定建议（只写报告 `reports/calib_<sha8>.md`，同样不覆盖旧报告、依次写 `-2`、`-3`…，以 stderr 的 `report -> …` 为准；不改 config.yaml）：

```bash
.venv/bin/python -m tools.calibrate
```

报告给出 `temporal.max_ext_rate` 与两个融合下限的建议值、`planarity.tau_hard` 约束是否满足、按手势分列的 R、各检查项的分布与 AUC，以及手势阈值在真手上的余量。冻结由人来做：把采用的值和报告依据写进 config.yaml 注释并提交，然后才跑 `--split test`。`--export-fixture` 另把 gestures calib 半区的关键点（不含视频与掌心块）导出到 `tests/fixtures/gestures_real.npz`，供 `tests/test_gestures_real.py` 做真手回归；这个文件是生物特征关键点，不得推到远端：仓库现在有公开远端，`tests/fixtures/` 已加进 .gitignore，导出后留在本机工作区、不提交（被忽略的未跟踪文件换分支不会丢，测试在任何分支都会跑；丢了重跑 `--export-fixture`）。不要 `git add -f` 到本地分支：切回 main 时文件会被删掉，而且一次 `git push --all` 就会上公开远端（PLAN.md dev-10，已确认）。

## macOS 注意事项

- **摄像头权限**归启动 python 的那个 App（Terminal、iTerm、VS Code 等）。第一次运行会弹授权框并以退出码 2 结束，这是预期行为：点"允许"后重新运行。被拒绝过的话，到 系统设置 → 隐私与安全性 → 摄像头 里打开。
- **连续互通相机**：本机同时有内置摄像头和 iPhone，编号不保证对应。画面来自 iPhone 时改用 `--camera 1`，或在 iPhone 上关闭连续互通相机。窗口标题会显示摄像头编号与实际分辨率。
- **视频效果必须关闭**：摄像头使用中打开 控制中心 → 视频效果，为终端 App 关闭"手势反应"（👍 ✌️ 会触发气球烟花特效）、"人物居中"、"影棚灯光"、"人像"、"背景"。不关会污染关键点与评测样本。
- 终端里的 'AVCaptureDeviceTypeExternal is deprecated' 警告可忽略。
- `data/` 下的录像与关键点属于生物特征数据，只保存在本机，已被 `.gitignore` 排除。

## 测试

```bash
.venv/bin/python -m pytest -q
```

测试只用合成数据和渲染的表情手，不需要摄像头。`tests/test_main.py` 里用 `--video` 跑 main.py 的用例会经过 `ensure_model()`：`models/` 里没有模型时会联网下载（PROMPT §2 要求 pytest 不触发下载；要不要给这些用例加"模型缺失时 skip"按 §0.2 待你决定）。本机有 smoke 录像或 `tests/fixtures/gestures_real.npz` 时，对应的真手回放测试会自动参与，没有就跳过。

自动测试和夜间整合都没有打开过摄像头（只用录像回放、合成帧和表情手），以下真摄像头项**待你验证**：

- [ ] `.venv/bin/python main.py --camera 0 --debug` 跑完一回合，再加 `--record data/smoke` 跑一回合（结果页按 `x` 可作废）
- [ ] 窗口标题为 `HandLiveness  cam N  WxH`，编号与分辨率和实际摄像头一致
- [ ] 正常室内光下顶栏 `CAM n FPS` ≥ 20（PROMPT §6 的 [ME] 验收项）
- [ ] HUD 依次出现 待机 → 倒计时 → 揭示目标 → 保持 → 结果页；挑战中匹配后手势丢了显示 MATCH LOST；HOLD 转腕总范围到 18° 时环上出现青色对勾、卡片显示 KEEP HOLDING
- [ ] `空格` / `r` / `x` / `d` / `q` 都生效；点窗口关闭按钮能退出
- [ ] 每次进入结果页，`logs/sessions.jsonl` 追加一行（按 `r` 中途丢弃的回合不写）
