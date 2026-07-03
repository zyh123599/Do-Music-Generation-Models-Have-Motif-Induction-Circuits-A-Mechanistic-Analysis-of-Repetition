# Motif Induction Circuits in Autoregressive Music Transformers

**音乐生成模型是否存在"动机归纳回路"：自回归音乐 Transformer 中重复、再现与变奏机制的因果分析**

本仓库是研究计划书（`docs/research_plan.md`）的完整可运行实现：在 MusicGen (small/medium/large) 上定位并因果验证**动机级归纳回路**——支持"重复但不复制"（移调重复、节奏变奏）的注意力机制——并将其做成免训练的 Motif Recurrence Knob。

- 所有度量均定义在 **delay-aware 帧坐标**上（`motif_circuits/delay_map.py`，已对 audiocraft 1.3.0 的 `DelayedPatternProvider` 做逐步等价性验证）；
- 回路定位在 **teacher forcing** 下进行，因果消融后果在**自由生成**下评测（两种模式严格分离，见计划书 §5）；
- 全部干预通过 attention 头输出（`out_proj` 输入）的 forward hook 实现，无需改动 audiocraft 源码。

## 仓库结构

```
motif_circuits/
  delay_map.py        # 帧<->步映射核心（一切分数的地基）
  stimuli/            # S1-S6 合成刺激集：MIDI 生成、FluidSynth 渲染、φ 对齐真值
  model/              # MusicGen 加载、attention 捕获、头级干预、teacher forcing、生成
  analysis/           # 对齐(移调不变/DTW)、lag 零模型、三层归纳分数、置换检验+FDR、activation patching
  metrics/            # SSM 条纹能量、动机回归率、loop 伪影、FAD/CLAP 包装
  knob.py             # Motif Recurrence Knob（γ 缩放候选头）
  utils/              # config/io/seeding/chroma
scripts/              # 00 冒烟测试 → 08 Knob 评测，全流水线
configs/              # 每步的 YAML 配置（默认迭代规模，注释里给论文规模）
docs/
  research_plan.md    # 研究计划书 v1.0
  audiocraft_api_notes.md  # 已验证的 audiocraft 1.3.0 内部语义（必读）
  interfaces.md       # 模块间接口契约
  runbook.md          # 服务器执行手册（顺序、算力预估、断点续跑）
tests/                # CPU 可跑的单元测试（含与 audiocraft pattern 的交叉验证）
```

## 服务器环境配置

要求：Linux + NVIDIA GPU（≥24 GB 显存跑 small/medium；large 建议 48 GB）、CUDA 12.x 驱动、Python 3.10/3.11。

```bash
# 1. 环境
conda create -n motif python=3.10 -y
conda activate motif

# 2. PyTorch 2.1.0（audiocraft 1.3.0 硬性锁定此版本）
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 \
    --index-url https://download.pytorch.org/whl/cu121

# 3. audiocraft（会带上匹配的 xformers<0.0.23）
pip install audiocraft==1.3.0

# 4. 本仓库及分析依赖
cd <repo>
pip install -e ".[server,dev]"

# 5. MIDI 渲染器 + GM 音色库（刺激集渲染用）
sudo apt install -y fluidsynth fluid-soundfont-gm
# 无 sudo 时：conda install -c conda-forge fluidsynth，再下载任一 GM SoundFont（如 FluidR3_GM.sf2），
# 并设置环境变量：export MOTIF_SF2=/path/to/FluidR3_GM.sf2
```

模型权重（facebook/musicgen-small/medium/large，公开权重）首次运行时自动从 Hugging Face 下载；可用 `HF_HOME` 指定缓存位置。

## 快速开始

```bash
# 先跑单元测试（CPU 即可，验证坐标映射等核心逻辑）
pytest tests/ -q

# GPU 冒烟测试（<2 分钟）：模型加载、delay 模式等价性、编码/解码、
# attention 捕获行和=1、头消融改变 logits、干预下生成
python scripts/00_smoke_test.py --config configs/default.yaml
```

冒烟测试全 PASS 后，按 `docs/runbook.md` 顺序执行完整流水线：

| 脚本 | 作用 | 对应计划书 |
|---|---|---|
| `01_build_stimuli.py` | 生成 S1–S6 刺激集（MIDI→渲染→manifest+φ 真值） | §4 |
| `02_encode_stimuli.py` | EnCodec 编码为 token | §3 |
| `03_head_screening.py` | 全头筛查：S6 零模型/周期头 + S1–S5 三层归纳分数 | §5 |
| `04_stats_heatmaps.py` | 置换检验 + BH-FDR、候选头排名、热图 | §5 |
| `05_patching.py` | Activation patching：恢复率 R、头数–恢复率曲线、对照组 | §6 |
| `06_ablation_generate.py` | 候选/随机/周期头 mean-ablation 下自由生成 | §6 |
| `07_output_metrics.py` | 输出侧度量：条纹能量、动机回归率、loop 率（+bootstrap CI） | §7 |
| `08_knob_eval.py` | Motif Recurrence Knob：γ 扫描 trade-off、T1 增强 / T2 loop 修复 | §8 |

所有脚本支持 `--config <yaml>` 与 `--override key=value`，幂等（已有输出自动跳过，`--force` 重跑），并在输出目录写 `run.json`（配置快照 + git 版本）。

## 关键实现说明

- **帧–步映射**：帧 t、码本 q 的 token 位于步 `s = 1 + t + q`（起始特殊 token 占步 0）。teacher forcing（`compute_predictions`）下序列长 `S = T + 1`，各码本尾部若干帧被截断并被 mask——`DelayMap.is_valid_step` 与上游 mask 已做等价性测试。
- **attention 权重获取**：audiocraft 的 memory-efficient attention 不落盘权重；我们在每层 `self_attn` 上注册 pre-hook，捕获输入并用该层自己的 `in_proj_weight` 精确复算 per-head softmax（无 RoPE/qk-norm，复算与模型完全一致，测试中用"复算概率 @ V ≍ 模块输出"验证）。筛查通过 reduce 回调即时把 `[B,H,S,S]` 归约为帧聚合矩阵/lag 谱，不整层落盘。
- **头级干预**：hook 在 `out_proj` 输入上按头连续通道切片做 zero/mean/scale/patch，自动跟踪流式生成的绝对步偏移（首次调用可含 prompt 多步），对 CFG 双批次同时生效。
- **统计**：周期头 lag 零模型 + S5 负对照 + 置换检验（对 φ 做全局 lag 平移，基于每样本 A′ 查询帧的 lag 谱即可复算，无需重跑模型）+ 全头 BH-FDR。

## 测试

```bash
pytest tests/ -q          # 全套 CPU 测试（144 个）
pytest tests/test_delay_map.py -q   # 坐标核心（与 vendored audiocraft pattern 交叉验证）
```

## 当前状态与注意事项

- 全部模块与脚本已实现并通过 144 个 CPU 单元/集成测试（坐标映射、hook 数学、
  对齐、统计、度量、刺激构建、脚本接线）。
- **GPU 路径（模型加载、真实 attention 捕获、patching、生成）在本仓库内是
  mock 验证的**——上服务器后第一件事跑 `scripts/00_smoke_test.py`，7 项全
  PASS 再进入流水线；任何 audiocraft 版本差异都会在这里暴露。
- 可选项：FAD/CLAP 质量评测需要额外 `pip install fadtk laion_clap`；
  S7 真实验证集需按 `motif_circuits/stimuli/real.py` 的说明准备
  `riffs.csv`；听测（计划书 §7）不在代码范围内。

## 许可

MIT。`tests/vendor_codebooks_patterns.py` 为 Meta audiocraft (MIT) 的原样副本，仅用于测试交叉验证。
