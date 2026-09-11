# 清洗管线重构：按公开方案鲁棒性报告重排攻击族

> 2026-09-09。设计已与用户确认（扩散净化按 A 方案收口：运行时与 sd-turbo
> 权重都随安装包内置、安装后零下载），本文件记录已实施的架构决策与验证口径。
> 研究边界见 `research/README.md` 与 `docs/研究用途与合规声明.md`。

## 1. 背景与依据

`research/PUBLIC_SCHEME_REPORT.md` 对五个公开深度水印方案给出完整证据链：

1. 经典失真攻击（滤波/重量化/几何/频域/普通转码）全部 BA≈1.0，处于对抗训练
   分布内，基本无效；
2. 扩散净化是最实用的黑盒攻击：TrustMark s0.30~0.35、WAM s0.15、
   MBRS s0.65 即到随机水平；视频逐帧净化 s0.15 把 VideoSeal 打到
   BA 0.53~0.58（8 段真实视频 FNR 1.00）；
3. 最强白盒攻击是 EOT-PGD + 正均值时变调制（ε=4/255、50 步、16 帧分块梯度
   累积，可穿真实 H.264/H.265 CRF23/28）；零均值时变无效；
4. 分割式水印的同步/定位模块最先失效（WAM mask 定向 PGD、扩散净化都能把
   IoU 打到 0）；
5. 嵌入域画像：VideoSeal = Y 低频 + 跨帧一致；WAM = Cb/Cr 色度低频 + 局部
   mask。这决定了「打哪里」比「打多狠」重要；
6. 帧序/帧插值/录屏重摄基本无效；转码效果高度依赖内容复杂度，评测必须分层。

现有管线的问题：70+ 选项火力压在报告证伪的经典原语上，`regenerate.py` 的
「再生」实为 temporal_subtract/FFT/DWT 重估计，没有生成式净化；验收矩阵只用
本地经典水印 + 合成切场视频做靶子，恰好是报告警告的简单内容假阳性场景。

## 2. 架构边界（正式代码 vs 研究侧）

核心原则：生产包只放黑盒原语；需要目标方案 decoder 的白盒能力留在研究侧。

| 能力 | 归属 | 理由 |
| --- | --- | --- |
| 扩散净化 `transform/purify.py` | 生产 | 黑盒、已接入清洗管线，是报告里唯一实用的通用攻击 |
| 嵌入域定向 `transform/embedding_domain.py` | 生产 | 黑盒（luma/chroma 低频重写），纯 numpy |
| EOT-PGD `research/scripts/eot_core.py` | 研究 | 白盒，必须拿到目标方案 decoder；生产侧不存在该条件 |
| 深度方案验收矩阵 `research/scripts/cleanse_matrix_deep.py` | 研究 | 依赖 vendor 仓库与公开权重，不进后端 CI |
| 经典原语（滤波/几何/频域/哈希等） | 生产（降级保留） | 对判重指纹仍有价值；不再作为水印清除主路径 |

## 3. 实现清单

### 3.1 生产原语

- `transform/purify.py`：sd-turbo img2img 逐帧净化（空 prompt、guidance=0，
  默认 strength 0.25 / 20 步）。A 方案运行时：
  - `backend[purify]` 声明 torch/diffusers/transformers/accelerate/safetensors/
    huggingface-hub/tqdm；`pnpm dev`、Electron dev、`pnpm package:backend`
    都带 `--extra purify`；
  - sd-turbo fp16 权重（约 2.4GB）由 `pnpm package:model` 准备到
    `packaging/models`，随 PyInstaller sidecar 内置；MPS/CUDA 直接 fp16，
    CPU 以 fp32 加载，一份权重覆盖全平台；安装后零下载；
  - 全局推理锁串行、MPS 逐批清缓存、`torch.inference_mode()`；
  - 取消/暂停/进度走线程本地控制面（`set_control`），不进入
    `TransformContext`，因此多进程武器池 pickle 不受影响；
  - 模型状态机 `missing/downloading/ready/error`；内置权重直接 ready，
    API/CLI 的下载入口仅作为开发/自托管兜底。
- `transform/embedding_domain.py`：分块 DCT 低频带（zigzag 1..15，剔除 DC）
  重写，按画像选 luma/chroma/both；纯 numpy、确定性。

### 3.2 管线接线

- `schemas.DesensitizeOptions` 新增 `purify_strength/purify_steps/
  purify_guidance/embedding_attack/embedding_strength`，全有界约束；
  `TemplatePayload` 同步补 camelCase 字段，模板 round-trip 不再丢配置。
- `strategies.TransformOptions` 同步扩展；thorough/fast 两策略在伪水印注入前
  执行「净化 → 嵌入域攻击 → 伪水印」，保证 spoof 载荷存活。
- `pipeline._prepare_desensitize` 预检净化能力：不可用时有效强度置零并记
  `purify_note="skipped: <reason>"`；首次下载记 `will_download`；
  净化/嵌入域攻击开启时禁用 YUV 快路径（RGB 域原语）。
- 净化任务强制单进程武器池（模型不跨进程回流失败原因且显存不可叠加）。
- 画质门控的参考帧在策略层之后取得，因此 `quality_protect` 只约束后续
  旋转/底层原语/哈希层，不会把净化结果拉回原图。
- `services` 结果模板带 `purify_note`；异步指标回写同步携带；任务开始清
  线程本地失败标记，避免跨任务污染。

### 3.3 开箱即用（A 方案）的对外接口

- API：`GET /api/purify/status`（依赖/内置模型/进度/设备/模型目录）、
  `POST /api/purify/install`（仅无内置权重时的开发兜底，202 后台下载）。
- CLI：`cthulhu purify-status`；`cthulhu purify-install` 仅开发/自托管兜底。
- 环境变量：`CTHULHU_PURIFY_ALLOW_DOWNLOAD=0`（离线禁用下载）、
  `CTHULHU_PURIFY_MODEL_DIR`（权重目录）、`CTHULHU_PURIFY_MODEL`（替换模型）、
  `HF_ENDPOINT`（镜像）。

### 3.4 验收矩阵

- `evaluate/harness.run_cleanse_matrix` 增加每档 PSNR/SSIM 与 FNR（BER<0.6
  判定失效）分层汇总；返回结构向后兼容。
- `research/scripts/cleanse_matrix_deep.py`：TrustMark/MBRS/WAM/VideoSeal 四
  方案 × simple/natural 内容 × 净化三档，视频额外验收「净化 + 真实 H.264
  CRF23」与 `--eot` 白盒档；输出 BA/FNR/PSNR/SSIM 到 CSV/MD。
- `research/scripts/purify_stack_matrix.py`：对 baseline / detail / embedding /
  full 四类黑盒增强栈做公开方案对比，统一输出 BA/FNR/PSNR/SSIM；用于校准
  “细节回注是否伤攻击”“嵌入域 v2 是否值得默认开启”等问题。

### 3.5 研究模式 · 白盒定向（已知公开方案）

> 2026-09-10 产品侧已移除：白盒只支持公开方案、600 帧短片上限，5 分钟
> 视频即使分段仍需数小时，不具备产品可用性；UI/API/schemas/打包与 sidecar
> 均已删除，研究脚本与权重保留在 `research/`。

- 主后端不引入 vendor 依赖：`transform/targeted.py` 只做参数校验、worker
  子进程调度、进度/取消/暂停与结果回传；`research/scripts/targeted_worker.py`
  在独立进程里加载公开 decoder 并执行 PGD。
- 支持方案：**VideoSeal / PixelSeal**（EOT-PGD，正均值时变调制 + EOT
  编码代理）与 **WAM**（mask 定向 / message 定向 / 两者同时）；用户只选
  方案、传视频，decoder 自动输出 mask/logits，不需要人工标注水印位置。
- 清洗选项新增 `target_scheme/target_attack/target_epsilon/target_steps/
  target_chunk/target_eot`；白盒预处理在常规清洗之前执行，指标仍以原始
  视频为参考。仅支持 ≤600 帧片段（研究模式）。
- 600 帧是白盒 worker 的硬上限（30fps 约 20 秒、60fps 约 10 秒），
  因为 worker 会把整段视频解码进内存并做逐窗口 PGD；5–10 分钟长视频
  当前不能直接跑白盒，请使用黑盒净化/嵌入域重写。若未来要支持长视频，
  需改为“分段解码 → 分段 PGD → FFV1 拼接”的流式 worker，耗时随时长线性增长。
- 打包：`pnpm package:targeted-worker` 把 VideoSeal/PixelSeal/WAM vendor
  代码与公开权重（MIT，约 1.7GB）复制到 `packaging/targeted`，用
  `cthulhu-targeted-worker.spec` 打成独立 sidecar；Electron 通过
  `CTHULHU_TARGETED_WORKER` 指向它。开发环境自动使用 `research/.venv`。
- DWSF 没有公开权重，白盒定向仍被“模型获取”卡住；需要作者权重或云 GPU
  复现训练后才能接入。

### 3.6 自动画像与共谋工作流

- `transform/profile.py`：黑盒内容复杂度画像（纹理/运动）在净化甜点区内
  建议强度/步数；新增跨帧一致性代理与细节回注建议。嵌入域按已知方案证据
  映射（VideoSeal→luma、WAM→chroma），未知方案保守 both。纯黑盒无法可靠
  识别嵌入域（静态水印与静态内容不可分），报告中 §2.8 的画像是干净/带水印
  白盒差分，已在模块注释中明确。
- `auto_profile` 选项接入清洗管线，结果返回 `profile_note/profile_metrics`；
  UI 在嵌入域卡片下新增“自动画像”开关与 `auto` 目标域选项。
- `services.run_collusion` + `POST /api/collusion` + 批量选择弹窗：同一内容
  多副本（2~32）对齐后平均/中位数，返回画质与 1/√N 预计削弱；报告证据
  8/16/32 副本 → BA 0.599/0.575/0.542，PSNR 65~67dB。

### 3.7 黑盒净化增强栈（2026-09-09 追加）

- `transform/purify.py`：新增 `purify_detail`（扩散后回注原帧高频细节，
  sigma=0.6 保守口径）与 `purify_temporal`（扩散前用跨帧一致性估计削弱
  低频稳定分量）。细节回注按帧内高频能量自适应，默认只保细纹理，不碰
  中低频；时序减法只在自动画像检测到高一致性/低运动时建议，强力档显式开启。
- `transform/temporal.py`：零均值低通 + 运动软掩码 + 时间中值估计，输出
  coherence/motion；不依赖干净参考帧，也不接触任何水印 decoder。
- `transform/embedding_domain.py`：新增 v2 变体（默认）：legacy 1..15 频带
  与报告 Top-10 频带并集、随机量化/抖动；可选随机分块相位、16×16 多尺度
  与 4:2:0 色度对齐。`embedding_variant=legacy` 可一键回退旧行为。
- UI：清洗面板新增「画质优先 / 强力清除」两个黑盒预设、细节回注/时序减法
  滑块、v2 与增强重写开关；模板编辑器同步全部字段。默认“画质优先”开净化
  + 细节回注、关嵌入域重写；“强力清除”开高 strength + 时序减法 + 嵌入域
  增强重写。
- 实测校准（research venv，MPS，4 帧/1 图小样本）：
  - VideoSeal simple：baseline BA 0.395 / PSNR 38.08 / SSIM 0.9921；
    detail BA 0.395 / PSNR 38.27 / SSIM 0.9924（同 BA 下画质更好）；
  - VideoSeal natural：baseline BA 0.535 / PSNR 30.39 / SSIM 0.9050；
    detail BA 0.523 / PSNR 30.48 / SSIM 0.9064；
  - WAM simple：baseline/detail BA 均 0.656，detail PSNR 32.09→32.21、
    SSIM 0.8728→0.8735；
  - WAM natural：baseline/detail BA 均 0.656，detail PSNR 22.76→22.87、
    SSIM 0.8278→0.8334；
  - MBRS natural s0.45：baseline/detail BA 均 0.578，detail PSNR
    21.69→21.71；说明细节回注对 MBRS 类高频水印基本中性，强力档仍默认
    关细节以留出最大攻击预算；
  - 嵌入域 v2 在当前小样本上 BA 增益有限且降低 SSIM，因此“画质优先”默认
    关闭，仅作为强力档/手动武器保留。
  - 结果文件：`research/output/purify_stack_*.{csv,md}`。

### 3.8 镜头级内容自适应与质量预算（2026-09-09 追加）

- `transform/profile.py` 新增 `profile_shots`：把 `decode_sampled` 的窗口帧
  映射回全局帧号，再按镜头切分抽样帧逐镜头画像；短镜头/无样本镜头回退到
  全片画像。
- `auto_profile=True` 时管线自动开启镜头检测（灰度 400 帧，缓存优先），
  另取彩色 60 帧做画像；`_build_segments` 返回 `seg_shot_indices`，
  处理循环按镜头选择 `TransformOptions`，多进程任务载荷携带该镜头的
  选项，不再全局一刀切。
- 质量预算：净化强度滑块是**上限**，自动画像只在预算内按镜头下调，
  不反向加码；细节回注/时序减法同样取 `min(用户值, 画像建议)`。
  强力档 `auto_profile=False`，仍按显式全局参数执行。
- UI 的「画质优先 / 强力清除」定位为**快捷预设**（不是互斥模式），
  使用与「清洗方式」一致的分段控件样式：选中项为白色胶囊 + 阴影。
  store 显式记录 `presetBase` 与 `presetModified`；手动改参后保留预设高亮，
  显示「已调整」标记和「恢复预设」，不再出现两个选项都不选中的断裂状态。
- 真实管线冒烟（3 段 18 帧合成硬切，sd-turbo 4 步）：检测到 3 个镜头，
  按镜头下发强度 0.178/0.188/0.177（用户上限 0.2），输出 MP4 正常。
- **删除「清洗方式 / 重新封装」**：重新封装对暗水印/内容水印的对抗能力为
  零，产品统一为重新编码。UI 选择器、store 的 `outputMode`、前端参数组装、
  后端 `output_mode`/`_run_remux` 分支与相关测试全部移除，openapi 与
  api-types 同步。
- **移除「检测参考」Tab**：没有干净同源基准与平台训练数据时，码流/盲检测
  分数会把“启发式可疑”误导成“水印检测结论”。UI 删除该 Tab、风险标签与
  检测入口，把编码/分辨率/帧率/时长/大小等客观文件信息移到清洗面板顶部；
  后端 `/api/detect` 与 CLI 检测保留为内部/研究工具，不再面向普通用户。
- **移除「水印区域 / 可见水印修复」**：该功能使用率低，且与暗水印清洗目标
  不同。UI 删除「水印区域」Tab、预览选框、区域 store、修复动作；后端删除
  `/api/repair`、`run_repair`、`jobs` 的 repair 分支与 `ffmpeg.repair_delogo`。
  历史 `_repaired` 产物仍按只读方式在产物列表中显示，避免旧记录丢失。

**方向修正**：研究侧 `eot_core.run` 早期误用 `+grad.sign()`（最大化
logits²，旧 ascend 口径）；已改为 `-grad.sign()`（最小化 logits²，报告
2.3 的 dual-tail descend 口径）。报告 §2.3 的独立 EOT 脚本本来就使用
descend，结论不变；deep matrix 的 `--eot` 结果如需引用应重跑。

## 4. 验证口径

- 后端：全量 pytest 通过；ruff 全过；版本升至 0.6.0，openapi.json 与
  api-types.ts 同步。
- 关键回归：
  - 多进程武器池在无净化任务下可 pickle（`CTHULHU_PROCESS_WORKERS=2`）；
  - 净化缺失依赖、内置模型、禁用下载与兜底下载路径都有测试覆盖；
  - 模板保存/加载保留净化与嵌入域字段；
  - 任务入队拒绝非法净化参数（422）。
- 研究侧实跑（research venv，MPS）：
  - TrustMark s0.25：BA≈0.53、PSNR≈33 dB / SSIM≈0.85；
  - VideoSeal 净化 0.15：BA≈0.47，净化+CRF23≈0.48，EOT+CRF23≈0.49~0.53；
  - 结果文件：`research/output/cleanse_matrix_deep.{csv,md}`。
- 以下白盒实跑为 2026-09-09 研究记录；2026-09-10 产品侧已移除，仅研究侧保留：
  - 白盒 worker 实跑（4 帧 512×512，MPS）：
  - VideoSeal EOT-PGD（ε=4/255、20 步、EOT）：self-BA 0.5205、
    |logits| 0.4034→0.0406、PSNR 45.26 / SSIM 0.9738；
  - PixelSeal EOT-PGD（ε=4/255、20 步、EOT）：self-BA 0.5225、
    |logits| 0.4597→0.0948、PSNR 46.41 / SSIM 0.9836；
  - WAM mask PGD（ε=4/255、4 步）：mask score 0.0495→0.0031、
    PSNR 47.10 / SSIM 0.9857；
  - 完整管线（`services.run_desensitize`）两种方案均返回
    `targeted_note=applied` 并产出最终 MP4；
  - 打包 worker 的 vendor 复制布局已用 `CTHULHU_TARGETED_VENDOR` 实测可跑；
    完整 PyInstaller sidecar 构建需在 CI/磁盘充裕机器上验证。
- 自动画像/共谋实跑：flat 复杂度 0.0 → 强度 0.15/20 步，textured 0.949 →
  0.245/29 步；3 副本平均 PSNR>25dB、预计残余 57.7%。
- 黑盒增强栈：新增 `test_temporal.py`、细节回注/时序接线回归与 v2 频带/
  确定性测试；UI typecheck + 52 tests + 生产构建通过。
- 镜头级自适应：新增 `test_profile_shots_splits_sampled_frames_by_shot` 与
  `test_pipeline_applies_per_shot_purify_budget`，验证按镜头下发不同强度且
  不超过用户滑块上限；UI 模式卡交互测试同步更新。

### 3.9 净化性能、进度与重启语义（2026-09-10 追加）

- **净化性能档**：新增 `purify_max_edge`（长边限制，默认 512）与
  `purify_batch`（批处理，默认 4）。扩散在低分辨率上执行，完成后放大回原
  尺寸，再做细节回注；水印能量集中在低频，降采样不会先丢水印。UI 暴露
  「快速 / 均衡 / 原始」三档，默认快速。balanced 预设强度从 0.25 降到
  0.15、steps 20，有效去噪步数约 3 步。
- **实测**：720×1280 合成视频，DCT-QIM 低频水印，8 帧：
  - optimized（512 长边、batch 4、s0.15、20 步）：9.42s，BA 0.4688，
    PSNR 31.7dB；
  - original（原分辨率、逐帧）：37.3s，BA 0.5234，PSNR 30.14dB；
  - 4 帧真实 sd-turbo 基准：optimized 4.77s vs original 14.73s。
- **公开深度方案复核（2026-09-10）**：重新下载 VideoSeal / PixelSeal / WAM
  官方权重（`research/vendor`，仅研究侧，不进包）后，对三档位做了同源对比
  （16 帧，BA 用跨帧 logits 平均；BA@h264 为净化后再过 H.264 CRF23）：

  | 方案 / 内容 | 原分辨率逐帧 | 快速 512/batch4 | 极速 256/batch8/s0.10 |
  | --- | --- | --- | --- |
  | VideoSeal 512 | BA 0.566 · 0.97s/帧 | BA 0.578 · 0.94s/帧 | BA 0.539 · 0.23s/帧 |
  | VideoSeal 720×1280 | BA 0.547 · 3.94s/帧 | BA 0.559 · 0.55s/帧 | BA 0.516 · 0.18s/帧 |
  | PixelSeal 720×1280 | BA 0.531 · 3.99s/帧 | BA 0.481 · 0.55s/帧 | BA 0.441 · 0.17s/帧 |
  | WAM 720×1280 | BA 0.707 · 4.00s/帧 | BA 0.469 · 0.55s/帧 | BA 0.475 · 0.17s/帧 |

  结论：极速档的清除效果与原分辨率逐帧持平或更好（WAM 上明显更好），
  单帧成本降到 1/4（512 输入）~1/21（720×1280 输入）；代价是 PSNR 低约
  3~8dB（24~27dB vs 30~32dB），UI 文案已标注「画质与清除率需复核」。
  详细矩阵：`research/output/purify_speed_tiers.md`、
  `research/output/purify_speed_tiers_extra.md`。
- **端到端实测**：走 GUI 任务路径（解码 → 净化 → 编码 → 封装），
  720×1280 / 300 帧 / 极速档 = 64.1s，即 **0.214s/帧**，输出 PSNR 27.9dB；
  换算 5684 帧（约 3 分钟 30fps）≈ **20 分钟**，对比原分辨率逐帧约 6.3 小时。
  追加研究档（192 长边 / 1 步扩散 / batch8）可到 0.165s/帧（≈16 分钟），
  未进 UI，留作后续需要时再开。
- **进度语义**：`_purify_progress` 把批次内进度换算成整段视频的全局帧数，
  进度文案变为 `扩散净化 processed/total_out`，不再每 240 帧批次显示
  `240/240 → 1/240` 而让用户误以为任务重启。
- **成片时长与帧数（2026-09-10 修复）**：端到端测速时发现 300 帧输入产出
  309 帧。根因不是封装补帧，而是「音频回声扰动」按设计把音画同步放慢 3%
  （`speed × 0.97` + `atempo` 保调）。由此修两处：
  1. 源**无音轨**时不再做这次放慢——没有回声水印可打，变速只会白白拉长成片；
     `ffmpeg.has_audio()` 作为音频类武器是否生效的统一判据，`_mux_output`
     改用 `state.audio_tempo` 判断拉伸分支，避免两处判据不一致。
  2. 结果新增 `out_frames`（成片帧数），与 `frames`（输入帧数）并列上报；
    有声源开回声清除时两者相差约 3%，API/前端不再按输入帧数误算产物时长。
- **音画同步与分段取整（2026-09-10 修复）**：验收标准是「成片音画不能不同步」，
  3% 放慢本身可以接受。实测发现真正的风险在取整：`_build_segments` 原来对每个
  镜头独立 `round(eff_len / factor)`，等长镜头下每段都固定多/少零点几帧，
  **50 个镜头会累积 14 帧（467ms）**；而音轨是按整段均匀拉伸的，多出来的
  视频帧就是音画不同步。改为累计取整（先累加精确输出长度再取整到帧，
  Bresenham 式），任意切点的偏差都夹在 1 帧内，总长恒等于
  `round(total_in / speed)`。回归测试：
  `test_shot_rounding_does_not_accumulate_av_drift`（50 镜头单元级）与
  `test_echo_defeat_keeps_audio_video_content_in_sync`（端到端用白闪帧 +
  1.5kHz 脉冲当锚点，要求各锚点音画偏差相对输入漂移 ≤1 帧）。
  另外单独实测 `atempo=0.97` 在 60s 上的速率误差为 1.2ms（无漂移），
  即放慢后的音轨本身不会引入累积错位。
- **重启语义**：`jobs.restore()` 不再把未完成任务静默从 0 帧重新入队；
  中断任务标记为 failed，`progress_note=进程重启中断`，错误信息提示当前
  版本不支持断点续跑、请手动重试。断点续跑仍需后续实现（按段/批次持久化）。

### 3.10 潜空间净化引擎（TAESD，2026-09-10 追加）

**动机**：现役扩散净化端到端约 0.21s/帧（720p）= 6.4× 视频时长。成本拆解
（`research/scripts/profile_purify_breakdown.py`）显示瓶颈是 sd-turbo 的 VAE：
每帧 VAE 编码 35ms + 解码 84ms，比 2 步 UNet（70ms）还贵。

**机制**：公开深度水印写在低频，纯降采样带不走（BA≈1.0，已用消融确认）；
真正起作用的是**潜空间瓶颈下的学习式重建**——重建由解码器先验主导，叠加式
水印无法被复现。于是用 10MB 的 TAESD（8 倍 AE）替代 2.4GB 的 sd-turbo：

- 引擎开关：`purify_engine`（`latent` 默认快档 / `diffusion` 强力档），
  贯通 schema → TransformOptions → pipeline → UI，模板同步；
- 潜空间档默认 128 长边、batch 8、细节回注 0.5；参数面板按引擎显隐
  （潜空间档不展示强度/步数/性能模式这些生成参数）；
- 权重内置 `packaging/models/madebyollin--taesd`（9.3MB，零下载），
  缺失时按需下载；`/api/purify/install?engine=` 支持按引擎安装。

**验证（产品代码路径，32 帧 × 3 方案 × 2~3 内容）**：

| 引擎 | BA@h264 区间 | PSNR（720p） | 净化 s/帧 |
| --- | --- | --- | ---: |
| 潜空间 `latent`@128 | 0.464~0.543 | 23.9~25.8 | 0.008~0.016 |
| 扩散 `diffusion`@256 | 0.474~0.551 | 24.3~27.4 | 0.153~0.224 |

**端到端**：3 分钟 720p 素材（5400 帧 + 音轨，走 GUI 任务路径含指标遍）
处理 133.8~139.2s = **0.74~0.77× 视频时长**，PSNR 25.2~25.4 / SSIM 0.89，
对比扩散档 6.4× 时长；120p 短片上扩散会把字幕与建筑重构成乱码，
潜空间档保留结构、字幕仍可读（对比图 `research/output/fast_visual_*.png`）。

**顺带修复**：`metrics.temporal_match` 在慢速/静态素材上会因相邻帧过于相似
而整体塌缩到同一帧（VMAF 直接变 0）。改为「等比例映射为主、内容匹配仅在
逐像素命中或明显更优时生效」，并改用去均值相关系数；段级重排匹配仍 100%
正确（新增两个回归测试）。

**强档复核（2026-09-10 追加，研究 §19.1）**：用产品代码路径做了边缘扫描、
同边缘对照与强档正面对照三组实验后，**扩散引擎在三方案上没有任何一维胜出**——
同边缘下 BA 差异落在噪声带（±0.02）内而成本 10~20×、重建更差；现行「强力净化」
（扩散 720 / strength 0.35 / 30 步 + 时序）被「潜空间@128 + 时序 0.5」追平并
略胜（BA@h264 0.4790 vs 0.4993），成本 1.942 → 0.048 s/帧。另外，潜空间瓶颈
的有效边缘窗口是 ≤256：512 档在原生 512 素材上几乎完全失效（BA 0.98~0.99），
说明「深度/强力」两档把边缘设到 512/720 是在这一维上反向。

由此得到的候选取舍（**待决策，未落地**）：移除 sd-turbo 与 `diffusion` 引擎，
包体 2.4GB → 10MB 级，同时删掉引擎开关、按引擎安装/预检分支与 UI 分段控件；
档位改为按「速度 × 保真」分层（默认 128 / 192 / 256），因为 ≤256 区间清除率
已饱和。边界：latent 域专门加固的方案（Stable Signature 等）未测。

**画质修正（2026-09-10 追加，研究 §19.2）**：上面这条「扩散没有一维胜出」的
判断**不成立**，补测「对原帧保真度」（与水印前干净原帧比，而非与带水印帧比）
后可见：扩散@512 = 26.02dB / SSIM 0.8112 是全部候选里最保真的配置，也是唯一
能把字幕、角落 logo 这类高频细节留住的档；潜空间档即便把 `purify_detail` 提到
1.0（SSIM 稳定 +0.02~0.03、清除率不变），在真实素材上仍会抹掉字幕与 logo。
两张素材的视觉对比见 `research/output/strong_visual_subtitle_zoom.png` 与
`strong_visual_720.png`。

不过「强力净化」（扩散@720）保真度垫底（21.67dB / 0.7084）而清除率并不占优，
退役换成潜空间@256 是净收益。因此删除 sd-turbo **不是无损提速**，取舍收敛为：
(a) 全潜空间、接受细节被抹；(b) 默认全潜空间 + sd-turbo 改按需下载，保留
「画质优先」档；(c) 保留内置 sd-turbo，只修掉强力档的边缘参数错误。

**宽带回注修正（2026-09-10 追加，研究 §19.3）**：上面「潜空间会糊掉字幕」是
在 `detail_sigma=0.6` 下的结论。σ 是回注带宽，水印在低频、字幕在中频，把 σ 放宽
即可两全：**σ1.5 + 边缘 256 + 细节 1.0** 的 SSIM 为 0.8686，高于扩散@512 的
0.8112，清除率持平（BA@h264 0.5228 vs 0.5215），成本 0.017 s/帧（扩散档
0.854），端到端 180s 720p 实测 30.5ms/帧 = 0.91× 视频时长。σ2.0 起清除率开始
回退，σ2.5 有 6/9 格水印回流，因此 σ1.5 是最佳折点；边缘 320/384 保真更高但
清除率明显变差。

即：**字幕可读 + 画面不糊 + 不依赖 sd-turbo** 是可以同时成立的，前提是把参数
通路打开——现版本有两个阻塞点：`detail_sigma` 写死 0.6 且未贯通 pipeline/UI；
`purify_detail` 被自动画像钳到 `0.35+0.35×complexity`（实测被降到 0.36），
用户填 1.0 也会被砍。

**已落地（2026-09-10，研究 §19.4）**：

- 扩散引擎与 sd-turbo（2.4GB）整包移除，包内只剩 9.3MB 的 TAESD（MIT）；
  `packaging/fetch_purify_model.py`、`/api/purify/install`、`model_status()`
  全部收敛到单一权重。
- 参数面：删除 `purify_engine/purify_steps/purify_guidance`，新增
  `purify_detail_sigma`（默认 1.5）；`purify_detail` 默认 1.0、
  `purify_max_edge` 256、`purify_batch` 8。旧模板里的废弃键在入参处丢弃，
  不会 422。
- 自动画像不再钳制 `detail`（它曾把 1.0 砍到 0.36~0.74，是字幕糊掉的主因），
  只保留 strength/temporal 的自适应；`Profile` 中失去消费方的建议字段已删。
- 预设（PRESET_VERSION=29）：深度＝画质优先 σ2.0@256、强力＝σ1.5@256+时序、
  极速＝σ1.5@192；UI 净化卡片从「引擎 + 强度 + 步数」简化为
  「速度/保真档位 + 细节带宽 σ + 时序减法」。

产品路径实测（真实素材）：σ1.5@256 → PSNR 21.47 / SSIM 0.7911；
σ2.0@256 → 21.90 / 0.8178（关净化的对照组为 29.63 / 0.9739）。
3 分钟 720p 长片端到端 166.4s = 0.92× 视频时长。字幕仍会发虚，若要更清晰只能
上 σ2.5 或边缘 320+，代价是清除率回退（BA 0.5579/0.5462，最差格 0.62~0.68），
因此默认未采用。

**分辨率修正（2026-09-10 追加，研究 §19.5）**：上面所有画质结论都来自 512 宽
素材，而长边 256 在 1080×1920 上是 7.5× 放大（144×256 铺满全屏），且
`detail_sigma` 是像素单位、不随分辨率缩放——1080p 用户实测「糊到无法观看」。
现已改为：σ 自动换算（`clip(短边×0.0024, 1.2, 3.0)`，512p→1.23、1080p→2.6）
＋引擎末尾自适应后置锐化（unsharp 0.6/1.2，约 4ms/帧），预设统一用 σ=0（自动），
面板不再暴露 σ（模板编辑器保留 0=自动/手动输入）。

1080×1920 复测：BA@h264 0.488（videoseal）/0.504（wam），对原帧 PSNR 26.1、
SSIM 0.925（旧默认 σ1.5 为 0.543 / 24.82 / 0.906）；字幕恢复可读
（`product_quality_1080p.png`）。长边扫描显示 256→0.543、384→0.637、
512→0.695，即**边缘不能随分辨率放大**，1080p 画面仍会偏软是这个攻击的固有代价。

## 5. 后续

- UI 接线已完成（2026-09-09）：清洗面板新增「扩散净化 · 生成式重建」分区
  （净化开关 + 强度/步数 + 内置模型状态/兜底下载进度卡片）与「嵌入域定向重写」
  （luma/chroma/both + 强度）；模板编辑器同步暴露这两组参数；
  `openapi.json → api-types.ts` 已重新生成，前端 typecheck/50 项测试/生产
  构建全通过，并做了浏览器视觉与交互验证。
- 打包体积与许可：torch/diffusers 进包后 dmg 体积显著增大；sd-turbo 适用
  Stability AI Community License，正式分发前确认条款与内容安全策略。
- 打包实机验证：本机磁盘余量不足，尚未执行完整 `pnpm package`；CI 打包
  流程已改为 `uv sync --extra purify`，需在 CI 或磁盘充裕机器上验证
  PyInstaller collect_all(torch/diffusers) 的体积与产物启动。
- 研究侧：扩大真实视频样本复核 detail/temporal 的内容相关边界；继续校准
  嵌入域 v2 的强度/频带权重；DWSF 权重或云 GPU 训练后的实测复核；白盒
  worker 的 PyInstaller 产物需在 CI 或磁盘充裕机器上完整构建验证。

## 5.1 端到端验收与耗时（2026-09-11）

四个内置模板在 1080×1920 / 24 帧素材上各跑一遍真实 `services.run_desensitize`，
结果说明与帧数全部符合预期：

| 模板 | 每帧耗时 | 结果说明 |
| --- | ---: | --- |
| 画质优先（推荐） | 379ms | `applied (潜空间重建, detail=1.00@σ自动·宽带回注+字幕增强)` |
| 平衡去水印 | 153ms | `applied (潜空间重建, detail=1.00@σ自动)` |
| 强力去水印 | 304ms | `applied (潜空间重建, detail=1.00@σ自动, temporal=0.50)` |
| 深度清剿（最狠） | 286ms | 同上 |

注意：**1080p 下每帧 153~380ms**，即 3 分钟片子约 14~34 分钟。此前"处理快于视频
时长"（0.92×）的结论只适用于 720p 素材（30.8ms/帧）；1080p 加上字幕掩膜后不再成立，
对外描述需按分辨率区分。

## 6. 已决策待实施：A 档（生成式重建）

**结论（2026-09-10 决策）：本版不引入，等本版测试通过后再加入，作为用户可选项。**

背景：现在三档之间存在"二选一"——要画面就用「画质优先」（长边 512 + 宽带回注
+ 字幕增强），但清除率打折（VideoSeal 类 BA@h264 0.74~0.79）；要清除就用
「平衡 / 清除优先」（长边 256/192），画面会糊、字幕可能不可辨。生成式档
（扩散 img2img）是唯一有望同时满足两者的方案，因此纳入路线图。

**为什么本版不引入**：B 档刚稳定，此时引入 2.4GB 权重与 0.2~0.9 s/帧的推理，
会把本版测试面（打包体积、冷启动、长片耗时、许可）扩大一倍；且 A 在 1080p 上
尚未验证，存在"投入后仍达不到画质底线"的风险。

**加入前必须先做的验证实验（30 分钟内可完成）**：

1. 重新获取 `stabilityai/sd-turbo`（fp16，约 2.4GB，此前按决策删除）；
2. 在 1080×1920 素材上跑：edge 512 + strength 0.15 + 2 步 + 现有细节回注/锐化；
3. 判定标准：**BA@h264 ≤ 0.55 且字幕可读** → 值得做；否则停止，B 档即为终态。

**加入时的落地形态（已确定）**：

- 作为**独立引擎档位**出现在「清晰度档位」旁边，默认关闭、不改变现有默认行为；
- 权重**按需下载**（不内置，避免安装包回到 2.4GB），沿用现有 `/api/purify/install`
  的进度与状态机制；
- UI 必须写明预期耗时（3 分钟片子约 20~30 分钟）与适用场景；
- 保留 B 档全部行为不变，用户按片子自行选择。
