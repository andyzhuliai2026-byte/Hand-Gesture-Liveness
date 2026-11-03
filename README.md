# HandLiveness — 手势挑战-应答式活体检测 demo（macOS）

当前进度：**M2**（挑战状态机、质量门控、按回合录制、评测框架；活体分项 5.1–5.4 仍是占位值 1.0，M3 实现）。规格见 [PROMPT.md](PROMPT.md)，实施计划与偏差记录见 [PLAN.md](PLAN.md)。

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

调试层显示：原始/平滑标签、每根手指的伸展度 e（T I M R P）、近节夹角、拇指两项特征（腕距比 wrist、横跨方向 cross）、掌宽占画面比例、ROI 亮度、质量门控状态（`quality ok/bad/-` 与当前提示）、处理耗时。

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

## 录制样本

规程见 PROMPT.md §7.3。每回合从倒计时到结果页自动落盘到 `data/<类别>/<YYYYmmdd-HHMMSS>/`（frames.npz、meta.json、video.mp4）；中途按 `r` 丢弃的回合不落盘。

录制前先关闭 macOS 视频效果（摄像头使用中打开 控制中心 → 视频效果，为终端 App 关闭"手势反应"、"人物居中"、"影棚灯光"、"人像"、"背景"）。

类别与最低条数：

| 类别 | 条数 | 要点 |
|---|---|---|
| `real` | ≥ 20 | 左右手各半，normal/dim/backlit 都要有，随机挑战；另录 ≥ 4 条 HOLD 几乎不动的（`variant=still`），不计入 FRR |
| `attack_photo` | ≥ 8 | print 4、phone 4；`--target` 固定；挑战出现后 1–3 s 移入介质，HOLD 时倾斜 ±20° |
| `attack_switch` | ≥ 12 | 手机相册 8、平板 4；`--target` 固定且不是 OPEN_PALM；挑战出现后从张开手照片滑到目标照片 |
| `attack_replay` | ≥ 16 | phone 8、tablet/monitor 8，still/moving 各半；另录 ≥ 4 条 `variant=preplay`、≥ 4 条 `variant=handheld` |
| `gestures` | 12 | 6 手势 × 左右手，`--target ID`，挑战出现后 1 s 内做出并保持 ≥ 3 s；同一手势的左右手**相邻**录（先左后右或先右后左都行） |

`--record` 的目录名必须是 `real`、`attack_photo`、`attack_switch`、`attack_replay`、`gestures`、`smoke` 之一，拼错直接报错退出。

切分规则（evaluate 自动做，不手工指定）：每个类别按目录名（即录制时间）排序，奇数位 → calib、偶数位 → test。所以：

- `gestures` 同一手势的左右手相邻录，两条正好一条进 calib、一条进 test；
- 某条按 `x` 作废后**立即重录同一条**（同一手势、同一只手、同一介质），不要跳到下一条再回头补，否则后面所有样本的奇偶位都会错开、配对被打乱。

只有这几种 (类别, variant) 单列、不参与切分和通过率：`real` 的 `still`、`attack_replay` 的 `preplay`、`attack_*` 的 `handheld`。`attack_replay` 的 `still` 和 `moving` 都正常参与切分与通过率，`real` 的 `moving` 也一样。

示例（录一条手机照片攻击）：

```bash
.venv/bin/python main.py --camera 0 --record data/attack_photo/ --target V_SIGN --meta medium=phone variant=moving lighting=normal hand=right
```

介质上手的像素掌宽要 ≥ 画面宽 10%（调试层显示占比），持介质的手不得入画，否则按 `x` 作废重录。

## 评测

在录好的样本上重放状态机并出表（只读 frames.npz + meta.json，不重跑 MediaPipe），报告写到 `reports/eval_<split>_<sha8>.md`：

```bash
.venv/bin/python -m tools.evaluate --split calib
```

`--split calib|test|all` 选半区（目录名排序后奇数位 calib、偶数位 test），`--gestures` 另出 6×6 手势混淆矩阵；`--split test` 按 PROMPT.md §7.5 判定，未达标时退出码 1 并写 `reports/failure_<sha8>.md`。

## macOS 注意事项

- **摄像头权限**归启动 python 的那个 App（Terminal、iTerm、VS Code 等）。第一次运行会弹授权框并以退出码 2 结束，这是预期行为：点"允许"后重新运行。被拒绝过的话，到 系统设置 → 隐私与安全性 → 摄像头 里打开。
- **连续互通相机**：本机同时有内置摄像头和 iPhone，编号不保证对应。画面来自 iPhone 时改用 `--camera 1`。窗口标题会显示摄像头编号与实际分辨率。
- **视频效果必须关闭**：摄像头使用中打开 控制中心 → 视频效果，为终端 App 关闭"手势反应"（👍 ✌️ 会触发气球烟花特效）、"人物居中"、"影棚灯光"、"人像"、"背景"。不关会污染关键点与评测样本。
- `data/` 下的录像与关键点属于生物特征数据，只保存在本机，已被 `.gitignore` 排除。

## 测试

```bash
.venv/bin/python -m pytest -q
```

测试只用合成数据和渲染的表情手，不需要摄像头。
