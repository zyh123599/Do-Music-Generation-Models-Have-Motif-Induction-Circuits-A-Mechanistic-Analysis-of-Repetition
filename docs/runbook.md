# 服务器执行手册（Runbook）

按顺序执行；所有脚本幂等（已有输出自动跳过，`--force` 重跑），支持
`--config configs/<x>.yaml` 与任意个 `--override key.path=value`。
每步在输出目录写 `run.json`（配置快照 + git 版本 + 时间戳）。

先决条件见 README「服务器环境配置」。首次运行前务必：

```bash
pytest tests/ -q                                   # CPU 单测（~1 分钟）
python scripts/00_smoke_test.py                    # GPU 冒烟（<2 分钟，7 项全 PASS）
```

## 执行顺序与产出

| 步骤 | 命令 | 输入 | 输出 | 迭代规模耗时(A6000, small)* |
|---|---|---|---|---|
| 1 | `python scripts/01_build_stimuli.py` | — | `data/stimuli/<CAT>/{audio,midi,manifest.jsonl}`；S1–S3 的 patching 配对进 S6 | ~20 min（fluidsynth 渲染为主） |
| 2 | `python scripts/02_encode_stimuli.py` | 步骤 1 | `codes/<id>.npy` int16 [4,T] + `codes_meta.json` | ~10 min |
| 3a | `python scripts/03_head_screening.py --pass A` | S6 codes | `results/<m>/screening/null_model.npz`（lag 谱 + 周期头掩码） | ~1–2 h |
| 3b | `python scripts/03_head_screening.py --pass B` | S1–S5 codes + null | `screening/<CAT>/persample_scores.npz`（motif/token IS、lag profile） | ~4–8 h |
| 4 | `python scripts/04_stats_heatmaps.py` | 步骤 3 | `candidates.json`（排名候选头）、热图 PNG、S5-vs-S2 散点 | ~10 min（CPU 为主） |
| 5 | `python scripts/05_patching.py` | 3、4 + 配对 codes | `patching/{recovery.jsonl,curves.npz,mean_vectors.npz,recovery_curve.png}` | ~2–4 h |
| 6 | `python scripts/06_ablation_generate.py` | 4、5 | `ablation/<cond>/<mode>/`（wav + codes + manifest） | ~4–8 h |
| 7 | `python scripts/07_output_metrics.py --target ablation` | 步骤 6 | `ablation/metrics.json` + 对比图 | ~30 min |
| 8 | `python scripts/08_knob_eval.py` | 4 + S1 音频 | `knob/{sweep.json,tradeoff.npz,repair_report.json,tradeoff.png}` | ~4–8 h |
| 8b | `python scripts/07_output_metrics.py --target knob` | 步骤 8 | `knob/metrics.json` | ~20 min |

\* 迭代规模 = 配置默认值（每类 100 条 ×3 渲染种子、消融每条件 50×3 生成、Knob 每 γ 30 条）。
论文规模（每类 500、生成 ≥200×3、n_perm=1000）对应计划书 §11 的估算：
全头筛查 ~200 GPU·h、自由生成评测 ~300 GPU·h、patching ~150 GPU·h
（medium/large 分别约 ×2 / ×3.5；先用 small 打通全流程再上大模型）。

## 模型切换

```bash
python scripts/03_head_screening.py --override model.size=medium
```
所有结果按 `results/<size>/...` 分目录，互不覆盖。跨规模对比（计划书 §9）
即依次以 small/medium/large 跑 3→4（+5）。

## 磁盘预估（迭代规模）

- 刺激集：6 类 ×300 wav ×10 s ≈ 2.3 GB + codes ~0.1 GB
- screening：`aprime_lag_profiles`（float16）small ≈ 0.1 GB/类，large ≈ 0.7 GB/类
- ablation/knob 生成音频：~4 GB
- 合计预留 ≥ 20 GB；论文规模预留 ≥ 100 GB。

## 断点与常见故障

- **断点续跑**：所有脚本幂等；02/03/06/08 按输出文件粒度跳过已完成部分。
  中断后直接重跑同一命令即可。
- **OOM（medium/large 筛查）**：`--override screening.layers=[0,1,...,23]`
  分段跑（先做一半层，`--force` 换另一半时注意 pass A 的层集合必须一致——
  脚本会校验并报错）。
- **`All ufuncs must have type numpy.ufunc`（import scipy 即崩）**：
  numpy/scipy 二进制 ABI 不一致（按 numpy 2 编译的 scipy 撞上被 numba/librosa
  依赖降级回来的 numpy 1.x）。修复：
  `pip install --force-reinstall "numpy==1.26.4" "scipy==1.11.4"`，
  然后 `python -c "import scipy.signal, scipy.special, torch, audiocraft"`
  验证。仓库 `[server]` extra 已固定该组合，重装请用 `pip install -e ".[server,dev]"`。
- **`import av` 报 `CXXABI_1.3.13 not found`（libopenvino）**：环境里存在
  conda-forge 的 ffmpeg/openvino 构建，av 链接到了它而系统 libstdc++ 过旧。
  首选：`pip install --force-reinstall --only-binary av "av==11.0.0"`
  （官方 wheel 自带捆绑 ffmpeg，不依赖环境库）。若仍失败：
  `conda install -y -c conda-forge "libstdcxx-ng>=13"` 并
  `conda env config vars set LD_LIBRARY_PATH=$CONDA_PREFIX/lib`，
  重新激活环境后验证 `python -c "import av"`。
- **fluidsynth 缺失**：`apt install fluidsynth fluid-soundfont-gm`；无 sudo 用
  conda-forge 并设 `MOTIF_SF2`。
- **HF 下载失败**：预下载 `facebook/musicgen-<size>` 到 `HF_HOME`，或配置代理。
  权重公开，无需 token。
- **S2 sanity 提示 mismatch**：chroma 对齐验证仅是数据质量信号（分析始终用
  MIDI 真值 φ）；mismatch 率 >20% 时检查渲染音质/音色设置。
- **05 出现大量 degenerate pair**：clean 与 corrupted 的 Δ 差距过小，多见于
  样本过短或候选头为空——确认 04 产出了非空 `candidates.json`。
- **周期头不足 K 个**：`periodic_<K>` 对照自动取可用数量并在日志说明。

## 结果核对清单（每步跑完看什么）

1. 00 全 PASS；03a 日志中周期头数量占比通常 <15%。
2. 04 热图应出现层间结构（而非均匀噪声）。**正式跑**要求
   `stats_summary.json` 的 `n_candidates` > 0 才能进入 5/6/8（为 0 说明
   筛查功效不足或假设不成立，应先加大样本量）；**pilot/管线验证**可用
   `--override patching.allow_ranking_fallback=true`（06/08 同名开关）让
   5/6/8 退化为按 excess 排名取头跑通流程——此时下游数字仅验证代码，
   无科学意义，日志会显式警告。
3. 05 `recovery_curve.png`：候选头曲线应明显高于 random/periodic 对照并随 K 饱和。
4. 07 对比图：`ablate_candidates` 相对 baseline 的动机回归率/条纹能量应下降，
   `ablate_periodic` 主要影响节拍稳定性相关指标（H3 功能分离）。
5. 08 `tradeoff.png`：重复度指标应随 γ 单调；γ<1 时 loop 率下降（T2 修复率）。
