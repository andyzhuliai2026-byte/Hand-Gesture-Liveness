# HandLiveness — 手势挑战-应答式活体检测 demo（macOS）

当前进度：**M1**（帧源、关键点、手势分类、最小界面、录制）。规格见 [PROMPT.md](PROMPT.md)，实施计划与偏差记录见 [PLAN.md](PLAN.md)。

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

按键（先用鼠标点一下预览窗口，按键才会生效）：`q` 退出，`d` 开关调试层。

调试层显示：每根手指的伸展度 e（T I M R P）、近节夹角、拇指两项特征（腕距比 wrist、横跨方向 cross）、掌宽占画面比例、ROI 亮度、处理耗时。

录一段样本（按 `q` 结束，自动存到 `data/smoke/<时间戳>/`）：

```bash
.venv/bin/python main.py --camera 0 --record data/smoke/
```

离线回放样本（不开窗口，逐帧打印 `t present raw smooth`，最后打印耗时统计）：

```bash
.venv/bin/python main.py --video data/smoke/<时间戳> --headless --bench
```

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
