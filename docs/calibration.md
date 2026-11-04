# 样本录制与标定步骤（PROMPT.md §7.3、§7.5）

本文是操作清单。规程原文在 PROMPT.md §7.3（录制）、§7.5（判据与 test 半区两次上限），工具说明在 README「评测」「标定（M4）」。所有命令在仓库根目录执行；`data/<类别>/` 目录不用先建，第一回合落盘时自动创建。

## 0. 每次开机录制前

1. 关闭 macOS 视频效果：摄像头开着时打开 控制中心 → 视频效果，给运行程序的终端 App 关掉"手势反应"、"人物居中"、"影棚灯光"、"人像"、"背景"。不关会往关键点里掺假（👍 ✌️ 正是"手势反应"的触发手势）。
2. 一律带 `--debug` 录：回合进行中右侧 SIGNAL 面板有 `PALM WIDTH x%`（手或介质上手的像素掌宽 / 画面宽）。这一行的圆点按质量门控 8%（`quality.min_palm_frac`）变琥珀色，不是按录制要求的 10%：8%–10% 之间圆点仍是青色、底栏也没有 Move closer，攻击样本要自己看数字，必须 ≥ 10%。结果页左侧 RESULT VALUES 面板第 4 项是 `two_hands_ratio`（攻击样本 ≥ 0.2 作废，`variant=handheld` 除外）。
3. 窗口弹出后先用鼠标点一下，按键才生效。顶栏红点 + REC 表示这次运行带了 `--record`（待机时也亮）；回合在进入结果页时才落盘。`d` 开关调试层，误按后 PALM WIDTH / RESULT VALUES 消失，再按一次。
4. 画面来自 iPhone（连续互通相机）时改用 `--camera 1`，或在 iPhone 上关闭连续互通相机；窗口标题显示摄像头编号与实际分辨率。

命令模板：

```bash
.venv/bin/python main.py --camera 0 --debug --record data/<类别> [--target ID] [--meta k=v ...]
```

`--record` 的目录名只能是 `real`、`attack_photo`、`attack_switch`、`attack_replay`、`gestures`、`smoke`。`--meta` 只认这几个键和值：`medium=none|phone|tablet|monitor|print`、`variant=still|moving|preplay|handheld`、`lighting=normal|dim|backlit`、`hand=left|right`、`notes=任意文字`；每个枚举键也接受 `na`；没给的枚举键写 `na`，notes 写空串；写错直接报错退出。`hand` 和 `lighting` 评测不用，但不写就分不出哪些是左手、哪些是暗光，凑"左右各半、三种光照都有"时要靠它。

## 1. 一回合怎么录

空格开始 → 倒计时 1 s → 揭示目标 → 稳定保持 0.5 s → 保持并转腕 1.5 s → 结果页 3 s → 回到待机，再按空格录下一条。每回合在结果页自动落盘 `data/<类别>/<YYYYmmdd-HHMMSS>/`（frames.npz、meta.json、video.mp4，约 1–3 MB，全部 < 300 MB）。

- 结果页按 `x`：这条标记作废（meta.json 写 `"discard": true`，文件不删，评测跳过），终端打印 `marked discard -> <目录>` 才算成功。结果页只停 3 s（按 `r` 立即离开），之后按 `x` 静默无效；错过了就在下次按空格之前，按 `logs/sessions.jsonl` 最后一行的 `debug.two_hands_ratio` 判断，手动把该目录 meta.json 里的 `"discard": false` 改成 `true`。
- 结果页是 Timeout / Wrong gesture（介质没被认成目标）时没有 RESULT VALUES 面板。只要回合中 PALM WIDTH ≥ 10%，这类回合照常保留、不按 `x`：evaluate 把它单列为"未到达活体检查"，攻击类的 n 不计它；某类到达活体检查的条数明显不足时，在该类最后成对补录。
- 真手样本不要因为判定结果（FAIL / INSUFFICIENT / Timeout）按 `x`：real 的 n 含全部样本，这些就是 FRR 的证据。只有录制条件与 meta 不符（换错手、光照与 meta 写的不一致）或没按规程做（倒计时时手不在画面里）才作废。
- 中途按 `r`：丢弃、不落盘。`q`、关窗、Ctrl-C 退出时未完成的回合同样不落盘。
- 作废后**立刻重录同一条**（攻击与 gestures：同手势、同手、同介质；real：同手、同光照）。evaluate 按目录名排序、奇数位 calib / 偶数位 test 切分，跳过去再回头补会把后面所有样本的配对打乱。
- 同一类里按"同条件两条一组"连着录，每组两条正好一条进 calib、一条进 test，每种条件在两个半区各占一半：real 左、左、右、右；attack_photo print、print、phone、phone。**不要**左、右、左、右——奇数位全是左手，calib 半区就只剩左手、test 半区只剩右手（gestures 例外：每组是同一手势的左右手，先后按 §3 轮换）。

## 2. real（真手）≥ 20 条，另加 ≥ 4 条 still

不给 `--target`，随机挑战（程序会排除你倒计时时已近似摆出的手势）。倒计时时张开的手已在画面里，揭示后做手势，保持阶段按提示自然左右转腕；条件允许时至少 1/3 由第二个人录（`notes=person2`）。

揭示后再开始换手势，太快（< 0.3 s）会被判预录；保持阶段转到卡片显示 KEEP HOLDING（18°）。dim / backlit 要留在质量门控之内：底栏出现 More light（块均值 < 40）或 Too bright（块均值 > 220 或饱和像素 > 10%）的帧算不合格，LOW_QUALITY 会计入 real 的 INSUFFICIENT ≤ ⌊0.1 n⌋ 判据。

建议 24 条 = 2 只手 × 3 种光照 × 4 条。`--meta` 对整次运行生效，所以每种光照先用 hand=left 的命令连录 4 条、按 `q` 退出，再用 hand=right 的命令连录 4 条（4 是偶数，每只手每种光照在两个半区各 2 条）。第二个人同样按 4 条一组录：

```bash
.venv/bin/python main.py --camera 0 --debug --record data/real --meta hand=left lighting=normal
```

```bash
.venv/bin/python main.py --camera 0 --debug --record data/real --meta hand=right lighting=normal
```

再换 `lighting=dim`（关灯只留弱光）和 `lighting=backlit`（背对窗或灯）各 8 条。

另录 ≥ 4 条保持阶段几乎不动的（预期 INSUFFICIENT NEED_ROTATION，只用来确认这条路径，不计入通过率）：

```bash
.venv/bin/python main.py --camera 0 --debug --record data/real --meta variant=still hand=right lighting=normal
```

## 3. gestures（分类准确率）12 条

6 手势 × 左右手各 1 条，`--target` 固定。倒计时张开手，揭示后 1 s 内做出目标并保持到结果页；t0 + 1.0 s 之后的帧算该手势，判定结果无所谓，只用关键点。同一手势的左右手**相邻**录，并且逐个手势轮换先后（第 1、3、5 个手势先左后右，第 2、4、6 个先右后左），这样每个手势在两个半区各 1 条、calib 半区左右手各 3 条；全部先左后右会让 calib 半区和导出的回归夹具只有左手：

```bash
.venv/bin/python main.py --camera 0 --debug --record data/gestures --target OPEN_PALM --meta hand=left
```

```bash
.venv/bin/python main.py --camera 0 --debug --record data/gestures --target OPEN_PALM --meta hand=right
```

依次 FIST（先右后左）、THUMBS_UP（先左后右）、V_SIGN（先右后左）、POINT_UP（先左后右）、OK（先右后左）。

## 4. attack_photo（照片，A）≥ 8 条（按下面的顺序录 10 条）

道具：打印和手机里各一套 FIST、THUMBS_UP、V_SIGN、POINT_UP 照片，另加一张打印的 OPEN_PALM（OPEN_PALM 时真手最接近平面，是 §5.3 要核对的真手/照片重叠点）。calibrate 只读 calib 半区，哪个目标进 calib 由录制顺序决定：print 按 FIST、THUMBS_UP、V_SIGN、POINT_UP 录，phone 按 THUMBS_UP、FIST、POINT_UP、V_SIGN 录（每组两条的先后与 print 相反），最后同一张 OPEN_PALM 连录两条，共 10 条，每个目标在 calib、test 各一条（两种介质用同一顺序的话，THUMBS_UP、POINT_UP 永远进不了 calib）。但注意随机挑战在倒计时见到张开手时会排除 OPEN_PALM 和 OK，real 样本里不会有这两个目标，calibrate 的按手势 R 对照在 OPEN_PALM 一行真手侧会是空的，怎么补真手 OPEN_PALM 的 R 要另外决定。介质用支架或夹子固定，或从画面外持握，**持介质的手不能入画**。

每条：`--target` = 照片的手势；倒计时时介质不在画面内；揭示后 1–3 s 内移入；保持阶段把介质倾斜 ±20°。预期 INSUFFICIENT NO_TRANSITION（§5.6 证据不足优先），5.3 的效果看报告里该类的 R 分布与硬否决命中率。回合中看 `PALM WIDTH ≥ 10%`、结果页看 `two_hands_ratio < 0.2`，不满足按 `x` 重录。

```bash
.venv/bin/python main.py --camera 0 --debug --record data/attack_photo --target FIST --meta medium=print lighting=normal
```

```bash
.venv/bin/python main.py --camera 0 --debug --record data/attack_photo --target FIST --meta medium=phone lighting=normal
```

## 5. attack_switch（相册切换，C）≥ 12 条

道具：手机相册和平板相册里，一张张开手照片，下一张就是目标手势照片。`--target` 固定且**不能是 OPEN_PALM**。

每条：倒计时时展示张开手照片；揭示后滑到目标照片；保持阶段倾斜介质 ±20°。手机 8 条、平板 4 条。预期 INSTANT_SWITCH / NO_TRANSITION / HAND_LOST 之一，报告给出三者分布。

```bash
.venv/bin/python main.py --camera 0 --debug --record data/attack_switch --target V_SIGN --meta medium=phone
```

```bash
.venv/bin/python main.py --camera 0 --debug --record data/attack_switch --target V_SIGN --meta medium=tablet
```

## 6. attack_replay（屏幕回放，B）≥ 16 条，另加 ≥ 4 preplay、≥ 4 handheld

先用手机录 6 段真手视频，每段一个手势：张开手 1 s → 做目标手势 → 保持 3 s 并转腕。**OPEN_PALM 那段改成握拳 1 s → 张开手 → 保持 3 s 并转腕**：视频开头就是目标的话，揭示时已是目标，只会判 NO_TRANSITION / PRERECORDED，测不到回放防线（attack_switch 排除 OPEN_PALM 也是这个原因；PLAN.md dev-11，待确认）。播放设备屏幕亮度调到最高。

正式样本 16 条：phone 8，tablet 或 monitor 8；每种里 still（架住）/ moving（手持晃动，持机的手不入画）各半。每条 `--target` = 视频里的手势；倒计时时视频首帧的手（张开手；OPEN_PALM 段是握拳）已在画面里（停在首帧）；揭示后 0.5–2.0 s 内**现场**按播放或寻址到对应片段。

```bash
.venv/bin/python main.py --camera 0 --debug --record data/attack_replay --target FIST --meta medium=phone variant=still
```

```bash
.venv/bin/python main.py --camera 0 --debug --record data/attack_replay --target FIST --meta medium=tablet variant=moving
```

另两组只用来确认原因码，不计入通过率，也不参与切分：

- ≥ 4 条 `variant=preplay`（`--meta medium=phone variant=preplay`）：先在手机上按播放，约 1 s 后再按空格，让揭示时视频里已经是目标手势。视频开头有 1 s 起始手势（张开手；OPEN_PALM 段是握拳）而倒计时只有 1 s，倒计时中途才开始播放的话 t_match − t0 通常 ≥ 0.3 s，判不出 PRERECORDED。预期 PRERECORDED。
- ≥ 4 条 `variant=handheld`：持介质的手故意入画，预期 TWO_HANDS。这组 `two_hands_ratio ≥ 0.2` **不要**作废。

## 7. 录完之后：calib 半区评测与标定建议

```bash
.venv/bin/python -m tools.evaluate
```

默认只跑 calib 半区（等价于 `--split calib`），报告写到 `reports/eval_calib_<sha8>.md`（已有同名报告时写 `-2`、`-3`…，实际路径在 stderr 的 `report -> …`）：各类判定表与 FAIL 原因分布、attack_replay 按 phone / tablet+monitor 分列、按目标手势分列的平面性残差 R、各 debug 量的分布、逐样本的 §5.6 debug 全集。加 `--gestures` 另出 6×6 手势混淆矩阵。

```bash
.venv/bin/python -m tools.calibrate
```

写 `reports/calib_<sha8>.md`（同样不覆盖旧报告，依次 `-2`、`-3`…），不改 config.yaml：`temporal.max_ext_rate`（真手 p99 × 1.3 且 ≥ 最大值）、`fusion.planarity_floor` / `texture_floor`（calib 真手 FRR = 0 的最大值，且 ≤ 初值 + 0.2）的建议值；`planarity.tau_hard` 约束（真手 R 的 p5 ≥ 2 × tau_hard）是否满足；按手势分列的 R 与真手/照片是否重叠；nonrigid / texture / planarity 的 c、s 所需的分布与 AUC（任务书没有公式，由人按分布定并把理由写进 config.yaml 注释）；手势阈值在真手上的余量。

`--export-fixture` 另把 gestures calib 半区的关键点（不含视频与掌心块）导出到 `tests/fixtures/gestures_real.npz`，供 `tests/test_gestures_real.py` 做真手回归。**这个文件是生物特征数据**：PROMPT §0.2 / §7.1 允许它入库的前提是"仓库仅存本机、不推远端"，现在 origin 是 GitHub 公开仓库，`tests/fixtures/` 已加进 .gitignore。导出后留在本机工作区、不提交：被忽略的未跟踪文件换分支不会丢，`tests/test_gestures_real.py` 在任何分支上都会跑；丢了就重跑 `--export-fixture`。不要用 `git add -f` 提交到本地分支：切回 main 时 git 会把它从工作区删掉，main 上的真手回归随之 skip，而且那条分支一次 `git push --all` 或误合并就会上公开远端（PLAN.md dev-10，待确认）。

## 8. 冻结阈值与 test 半区

1. 把采用的值和依据（哪份 calib 报告、哪个分布）写进 config.yaml 对应键的注释（来源标签留 `[calib]`，去掉已不是来源的 `[synth]`，`[dev-N]` 保留），并把 config.yaml 第 2–3 行与 README「配置」里 `[calib]` 的说明改成"已按 calib 半区标定"，提交 config.yaml。活体阈值与 `tracker:` 配置绑定，改 tracker 要重新标定。
2. 跑 test 半区：

```bash
.venv/bin/python -m tools.evaluate --split test
```

退出码 0 = §7.5 全部非 WARN 判据达标；1 = 未达标，已写 `reports/failure_<sha8>.md`（同一 sha8 已有时写 `-2`、`-3`…，实际路径见 stderr 的 `failure report -> …`；完全没有样本也退出 1，但不写失败报告）；2 = 配置错误或被拒绝（不写报告）。attack_replay 只是目标：未达标或 n = 0 仍退出 0，但照写失败报告。

3. **test 半区最多跑 2 次**：`--split test` 与 `--split all` 只要含 test 样本就记进 `reports/test_runs.json`，已有 2 条时第 3 次直接拒绝（退出码 2）。第 2 次之前先在 DESIGN.md §5 的表里写清改了什么、依据哪份 calib 报告；两次报告都保留。冻结之前不要跑 `--split all`。
4. 结果填进 DESIGN.md §9；attack_replay 若仍 > 20%，按 §5.6 / §7.5 如实写明 v1 对 B 的防线只有 5.4，不回调融合、不放宽 5.1。

## 9. 量级与顺序

real 28 + gestures 12 + attack_photo 10 + attack_switch 12 + attack_replay 24 = 86 条，每条连重置约 15 s，纯录制约半小时，道具准备另算。建议顺序：real → gestures（不要道具）→ attack_photo → attack_switch → attack_replay。
