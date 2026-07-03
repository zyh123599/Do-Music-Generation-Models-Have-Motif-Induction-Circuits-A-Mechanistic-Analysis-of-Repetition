# 研究计划书：音乐生成模型中的动机归纳回路

**英文题目：** Do Music Generation Models Have Motif Induction Circuits? A Mechanistic Analysis of Repetition, Recurrence, and Variation in Autoregressive Music Transformers

**中文题目：** 音乐生成模型是否存在"动机归纳回路"：自回归音乐 Transformer 中重复、再现与变奏机制的因果分析

版本 v1.0 · 2026-07 · 目标投稿：ICLR 2027（备选 ICML 2027）

---

## 0. 一句话研究问题

我们研究自回归音乐生成模型内部是否存在一种不同于文本 token-copy induction head 的**动机级归纳回路（motif induction circuit）**：它能在底层 token 不完全重复的条件下，支持主题再现、移调重复与节奏变奏（即"重复但不复制"），并且可以被因果干预，用于控制生成的重复度与缓解 loop 伪影。

---

## 1. 背景、定位与新颖性声明

### 1.1 领域地图（为什么这个坑还空着）

现有音乐生成模型可解释性工作可归为四类，均非回路级因果分析：

| 类别 | 代表工作 | 与本文差异 |
|---|---|---|
| 概念探针 | SynTheory（Wei et al. 2024）；Ma & Xia 2024（和弦根音/性质） | 问"编码了什么"，不问"如何计算" |
| 激活转向 | MusicRFM；SMITIN（头级探针控制乐器）；Facchiano et al.（DiffMean patching 控制二元属性）；线性探针转向（流派/音色）；TADA（扩散模型） | 目的是控制属性，未逆向工程任何算法机制 |
| SAE 概念发现 | Singh et al.（MusicGen 残差流）；Paek et al.（音频潜空间声学属性）；AudioSAE | 静态特征字典，无时序算法、无因果回路 |
| 注意力编辑 | Melodia（AudioLDM2 注意力图探针编辑）；Instruct-MusicGen | 编辑应用，非机制解释 |

**必须主动引用并差异化的符号域先例：** Music Transformer（Huang et al. 2018）曾展示注意力回看早先动机的可视化——但那是轶事性可视化、符号域、无系统度量与因果验证。本工作在音频 token 域给出可复现的量化度量与因果证据。

### 1.2 叙事钩子

在语言模型中，重复是 bug："repetition curse"已被机制性地归因于 induction head 在重复时的过度主导（"toxicity"）、分布于中间/顶层的"重复神经元"、以及 SAE 潜空间中显式的重复特征方向。在音乐中，重复是 feature：动机、riff、副歌回归是音乐性的核心。**同一类机制、两种相反的命运**——音乐模型是研究"良性重复回路"的天然实验室。工程旁证：MusicWeaver 需外挂 Motif Memory Retrieval 模块来保证长曲中的动机同一性不被稀释，说明领域默认预训练模型"自己做不好这件事"，却无人打开看过其内部到底怎么做。

### 1.3 新颖性声明（论文措辞，谨慎版）

> To the best of our knowledge, prior music interpretability work has studied concept probing, activation steering, SAE-based concept discovery, and attention-based editing, but has not causally characterized motif-level recurrence circuits in autoregressive music transformers.

### 1.4 作用域限定（主动收窄，先于审稿人）

MusicGen 上下文 ≤ 30 s（50 Hz × ~1500 步）。本文研究 **riff / 乐句 / 小节级的中短程再现**，不主张解释分钟级副歌回归；后者留作长曲模型迁移实验（§9）。全部机制分析在**无条件生成或固定中性提示词**下进行，排除文本侧混淆（与 Singh et al. 使用无条件音频的做法对齐）。

---

## 2. 研究问题与假设

- **RQ1（存在性）：** 音频 token 自回归模型中是否存在 prefix-match-and-copy 型注意力回路？其匹配发生在 token 级还是感知/动机级？
- **RQ2（因果性）：** 消融 / patch 候选回路是否因果地削弱重复能力（并连带影响调性保持、节拍稳定性）？
- **RQ3（可控性）：** 调节候选回路能否得到免训练的"重复度旋钮"，在动机回归与 loop 修复专项任务上优于 prompt 与通用转向基线？

预注册式假设（均为待验证，写作时不得当作已知事实）：

- **H1（码本分工）：** 粗码本（codebook 0/1）承载更多动机归纳信号，细码本（2/3）更多承载音色纹理；
- **H2（分布式回路）：** 若不存在文本式"明星头"，动机再现由分布式回路（多头组合 + 残差流方向）实现；
- **H3（功能分离）：** 存在可与"节拍周期头"（固定 lag 回看）在统计上区分的动机匹配头。

---

## 3. 技术前置：delay-aware 帧–步映射（所有度量的地基）

MusicGen 以 delay 交织模式生成 EnCodec token：4 个码本、50 Hz 帧率；**帧 t 的码本 k token 出现在生成步 s = t + k + s₀**（s₀ 为特殊 token 偏移）。因此"attend 到上一次出现的后继位置"中的"后继"，对每个码本的步坐标偏移都不同。

**约定：本计划书中所有分数与干预均定义在帧对齐坐标上。** 交付开源工具 `delay_map`，功能包括：帧↔步双向映射、按码本切片注意力矩阵、帧级 patch 目标定位、跨码本聚合。若无此步，§5 的三层分数均无法正确定义——这是与文本域方法最大的工程差异，也是可独立复用的贡献。

---

## 4. 刺激集设计（阶段一）

合成管线：程序化 MIDI 生成 → 多乐器渲染（FluidSynth SoundFont / DawDreamer）→ 重采样 32 kHz → EnCodec 编码。每样本约 10 s，结构为：动机 A（1–2 小节，4–12 音）+ 间隔材料 G（随机生成、不含 A 素材）+ 再现段 A′。随机化维度：调性、速度（80–140 BPM）、乐器、力度。**每个样本自带 ground-truth 帧级对齐 φ: A′ 帧 → A 帧。**

| 编号 | 类别 | 构造方式 | 用途 |
|---|---|---|---|
| S1 | 精确重复 | A … A | token 级与动机级分数的上限参照 |
| S2 | 移调重复 | A … A(+k 半音)，k ∈ {±3, ±5, ±7} | 检验超越 token-copy 的匹配（核心） |
| S3 | 节奏变奏重复 | 音高轮廓保持、时值改写（增/减值、切分） | DTW 对齐下的动机匹配（核心） |
| S4 | 音色变化重复 | 同 MIDI 换乐器渲染 | 码本分工假设 H1 |
| S5 | 同节奏乱音高（负对照） | 保留 A 的节奏骨架、音高随机重采 | **排除节拍周期头混淆的关键对照** |
| S6 | 无重复 | 两段独立乐句 | 零模型语料 + patching 配对样本 |
| S7 | 真实验证集 | POP909（含乐句/段落结构标注）+ 自建 100–200 段 riff / 主题回归片段（SALAMI、HookTheory 辅助定位） | 外部效度 |

规模：S1–S6 每类 500 条（迭代期先各 100 条）；每条用 ≥3 个随机渲染种子，以平均 EnCodec 编码噪声。

---

## 5. 三层归纳分数（阶段二，teacher forcing 模式）

**分析模式约定：** 回路定位一律在 teacher forcing 下进行（把刺激音频编码后整段喂入模型，位置有 ground truth）；因果消融的行为后果一律在自由生成下评测（§7）。二者不得混用。

### 5.0 零模型（先于一切分数）

在 S6（无重复）语料上，对每个头 h 统计注意力滞后谱 P_h(ℓ) = E_t[α_h(t → t−ℓ)]（帧坐标）。在拍长 / 小节长整数倍 lag 上显著成峰的头记入**周期头集合 B**。任何候选归纳分数都以"lag 匹配零模型"为基线：

> Null_h(t) = P_h(ℓ_t)，其中 ℓ_t = t − (φ(t)+1) 为该查询位置对应的目标滞后。

候选归纳头须同时满足：(a) 分数显著超过 Null_h（置换检验 n=1000，跨全部头做 FDR 校正 q<0.05）；(b) 在 S5 负对照上**不**显著。仅满足 (a) 而在 S5 上同样得高分的头归入 B（周期头），不计入候选回路。

### 5.1 Token 级归纳分（S1 适用）

对 A′ 中帧 t、码本 k，定义匹配分与复制分：

> **IS_tok^(k)(h)** = E_t [ α_h( step(t,k) → step(φ(t)+1, k) ) ] − Null_h
>
> **CS_tok(h)** = E_t [ Δ log p( x_{φ(t)+1} ) ]，其中 Δ 为仅保留头 h 对残差流写入时（单头 logit attribution / OV 路径分解）真值后继 token 对数概率的提升量。

### 5.2 码本感知归纳分（S1 / S4 适用）

分别报告 k = 0..3 的 IS_tok^(k) 与跨码本联合分（先按帧聚合注意力再打分）。检验 H1 的两条预测：(i) IS^(0,1) ≫ IS^(2,3)；(ii) 在 S4（同 MIDI 换乐器）上粗码本分数保持、细码本分数坍塌。

### 5.3 动机级归纳分（S2 / S3 适用，核心贡献）

对齐 φ 的构造：S1 用恒等映射；S2 用**移调不变 chroma 对齐**（12 维 chroma 向量做循环移位、以最大互相关确定移位量 k̂ 与帧对应关系，并与 ground-truth 移调量核对）；S3 用 **DTW**（特征 = chroma ⊕ onset envelope）。

> **IS_motif(h)** = E_{t∈A′} [ Σ_{s ∈ W(φ(t)+1, w)} α_h(t → s) ] − Null_h，帧坐标，容差窗 w = ±2 帧（40 ms）
>
> **CS_motif(h)** = teacher forcing 下将头 h 输出置换为 S6 均值后，A′ 中"动机延续帧"真值 token 平均对数概率的**下降量**（分码本报告）。

**汇总产物：** MusicGen-small / medium / large 的全头（layer × head）三层分数热图 + 周期头图谱 + 跨规模涌现曲线。

---

## 6. 回路定位与因果验证（阶段三）

### 6.1 Activation / path patching 协议

- **配对设计：** clean = S1/S2 样本；corrupted = 与 clean 共享"前奏 A + 间隔 G"、但把 A′ 替换为无关乐句的 S6 配对样本（成对生成，逐帧对齐）。
- **方向与指标：** 采用 denoising 方向——把 clean 前向中候选组件的激活 patch 进 corrupted 前向；指标为**动机延续 log-prob 恢复率** R = (Δ_patched − Δ_corr) / (Δ_clean − Δ_corr)，其中 Δ 定义为 A′ 起始 L 帧（默认 L=25，即 0.5 s）真值 token 的平均对数概率。若 R 不敏感，备选 KL 散度或真值 token 秩指标。
- **粒度递进：** 单头输出 z_{l,h} → 头组合（按 §5 分数排序累积 patch，绘制"头数–恢复率"曲线）→ 残差流方向级（以 SAE / RFM **作为工具**在候选层定位"重复特征方向"并 patch 之；仅工具化使用，不作为论文主线，呼应 H2 与 LLM 侧 SAE repetition features 的先例）。
- **两跳回路图：** 对 top 头做 QK 侧（匹配信号来自何处——是否存在前 token 头组合，类比文本 previous-token head）与 OV 侧（写入内容是否为音高类信息——用线性探针读出）的 path patching，绘制回路示意图。

### 6.2 消融实验（自由生成模式）

mean-ablation（以 S6 上的头输出均值替换），三组对照：**top-K 候选头 vs 随机 K 头 vs K 个周期头**，K ∈ {4, 8, 16}。提示词固定为无条件或"simple melody"；每配置 ≥200 条生成 × ≥3 种子。

### 6.3 干预–观测矩阵

| 干预 | 观测指标（§7 定义） | 若假设成立的预期 |
|---|---|---|
| 消融候选头 | 动机回归率 ↓、SSM 条纹强度 ↓；FAD 变化有限 | 回路的必要性 |
| 消融周期头（对照） | 节拍稳定性 ↓，动机回归率基本不变 | 与候选头的功能分离（H3） |
| corrupted ← clean patch | log-prob 恢复率 R 高且随头数快速饱和 | 回路的充分性 |
| γ 缩放候选头输出 | 输出重复度随 γ 单调变化 | 可控性（→ §8） |

---

## 7. 输出侧重复度量（自由生成评测管线）

- **结构重复分：** chroma 自相似矩阵（SSM）off-diagonal 条纹能量（lag 域峰值强度），辅以 Foote novelty 曲线对比结构清晰度；
- **动机回归率：** prompt 音频含动机 A，统计续写 30 s 内出现"与 A 的移调不变 chroma 相关 > τ"片段的样本比例；报告 τ ∈ [0.6, 0.9] 的敏感性曲线，避免单阈值被质疑；
- **Loop 伪影率：** 末 5 s 的 token n-gram 重复率与音频帧级自相关塌缩联合判定（两条件同时触发才计为 loop）；
- **质量与提示遵从：** FAD/FD（Gui et al. 2024 音乐适配版）+ CLAP 分数，保证机制干预不以牺牲质量为代价；
- **听测：** 12–16 人、MUSHRA 式界面，双问项（① 重复 / 主题回归的感知强度；② 整体音质），样本按自动指标分层抽取；伦理审批与知情同意提前两周申请。

---

## 8. 应用兜底：Motif Recurrence Knob（阶段四）

**实现：** 对候选回路头输出统一乘系数 γ（γ>1 增强、γ<1 抑制），或沿 §6.1 定位的残差"重复方向"注入 ±η。

**专项任务与基线（只打专项，不做全面属性对比）：**

- **T1 动机回归增强：** 给定含动机的 prompt 音频，最大化动机回归率，约束 FAD / CLAP 退化不超过阈值。基线 = 文本提示（"...repeats the opening motif"）、DiffMean 方向注入、SMITIN、MusicRFM（各按其原法适配"重复"概念，保证公平）。
- **T2 Loop 修复：** 对基线生成中检出 loop 的样本施加 γ<1 抑制，报告修复率与质量代价。

产出 **γ–重复度 trade-off 曲线**，与各基线的强度超参对标。论证逻辑：MusicRFM 控 notes/chords 等概念方向、SMITIN 控乐器等 trait，均非为 motif 级再现设计——本方法只需在该专项上取胜。

---

## 9. 附加与扩展实验（正文小节或附录）

- **文本条件信息流小节：** 对比中性提示 vs "repetitive / looping"提示对候选头激活的调制强度，解释"prompt 为何控制不了细粒度重复"（呼应 MusicRFM 中 prompt-only 基线接近随机的已发表现象）；
- **跨规模涌现：** small / medium / large 的回路分数与消融效应对比（呼应 Singh et al. 关于规模改变表征组织的发现）；
- **跨模型迁移（主结果之后）：** 在一个开源长曲或其他音频 AR 模型上复跑 §5 筛查的缩减版，验证结论外推性。**主实验坚持 MusicGen S/M/L**——AudioCraft 开源、结构清楚、token 可取、attention 可 hook，不在启动期拉爆工程复杂度。

---

## 10. 风险登记与预案

| 风险 | 概率 | 预案 |
|---|---|---|
| R1 找不到"明星头" | 中 | 转分布式回路叙事（H2 预置）：头组合累积曲线 + 残差方向级证据；结论改写为"音乐重复由分布式 motif-recurrence circuit 实现" |
| R2 token 级复制信号弱 | 高 | 正是论题卖点："音乐重复不是 token copy 而是感知级再现"；动机级分数为主线，token 级仅作参照 |
| R3 合成数据外部效度质疑 | 中 | S7 真实验证集上复现关键分数排序与消融趋势；合成负责因果可控、真实负责外部效度 |
| R4 周期头混淆 | 中 | S5 负对照 + lag 零模型 + §6.3 功能分离消融，三重防线 |
| R5 EnCodec 噪声 / 高熵 | 中 | 渲染参数控制变量、多种子平均、帧窗 w 容差 |
| R6 patching 指标不敏感 | 低–中 | 换 KL / 秩指标；增大 L；改用 S2 扩大 clean–corrupted 差异 |
| R7 时间窗（≤30 s）质疑 | 低 | §1.4 已主动限定 claim 为中短程再现；长程留给迁移实验 |

---

## 11. 工程与算力

- **自研 hook 框架：** TransformerLens 不支持 AudioCraft，需自写 forward hook 注册于每层 self-attention；逐头统计采用**流式累积**——前向时只累加落在目标索引集（φ 映射窗口）上的注意力质量，不落盘完整注意力矩阵（large 为 48 层 × 32 头 × ~1500 步，全量存储不可行）；SMITIN 逐头探针的先例证明该规模可行。
- **硬件：** 单卡 48 GB（A6000 / L40）全程可行；small 迭代调试、large 出主结果。粗估 GPU 时：全头筛查 ~200 h、自由生成评测 ~300 h、patching ~150 h。
- **开源承诺：** 刺激集生成器、`delay_map` 工具、逐头分数管线、Knob 实现、demo 页（音频样例 + 回路可视化）。

---

## 12. 贡献包装（论文四点）

1. **问题：** 首次（按 §1.3 谨慎措辞）系统且因果地刻画自回归音乐 Transformer 中的动机级再现机制，提出"重复但不复制"这一音乐特有的机制问题；
2. **度量：** delay-aware、码本感知、配周期零模型的三层 music induction score（token 级 / 码本级 / 动机级）；
3. **发现：** 候选头 / 回路 / 残差方向对动机再现的必要性与充分性因果证据，及其跨规模涌现规律；
4. **应用：** 免训练 Motif Recurrence Knob，在动机回归增强与 loop 修复专项任务上优于 prompt 与通用转向基线。

---

## 13. 里程碑（14 周基线；两人并行可压缩至约 11 周）

| 周次 | 交付物 |
|---|---|
| W1–2 | `delay_map` + hook 框架 + S1–S6 刺激集 v1（各 100 条）；映射与分数的单元测试 |
| W3–4 | MusicGen-small 全头筛查、周期头零模型、三层分数 v1；初版热图 |
| W5–6 | medium / large 筛查 + 置换检验与 FDR；H1 码本分工检验；刺激集扩至 500/类 |
| W7–9 | patching 协议跑通（单头 → 头组合 → 方向级）；消融实验 + 输出侧度量管线 |
| W10–11 | Knob 实现 + T1/T2 基线对比；S7 真实验证集复现 |
| W12 | 听测执行（伦理与知情同意于 W10 前提交） |
| W13–14 | 写作、图表、demo 页、代码整理与内审 |

**投稿窗口：** ICLR 2027（摘要截止历年约在 9 月下旬，以当年官网为准）——W1–W9 的结果已足以支撑投稿版本，W10–W12 可在 rebuttal 前补强；若时间不足则顺延 ICML 2027，结果只会更扎实。

---

## 14. 必引文献清单（按定位分组）

- **LLM 机制先例：** Olsson et al. 2022（induction heads）；repetition curse 机制系列（induction head toxicity；repetition neurons；SAE repetition features）——提供可移植的方法论模板与"重复在语言是 bug、在音乐是 feature"的对照叙事；
- **音乐概念探针：** SynTheory（Wei et al. 2024）；Ma & Xia 2024；
- **转向系（差异化对象）：** MusicRFM；SMITIN；Facchiano et al.（DiffMean activation patching）；线性探针转向；TADA（音频扩散）；
- **SAE 系（工具化引用）：** Singh et al.；Paek et al.；AudioSAE；
- **编辑系：** Melodia；Instruct-MusicGen；
- **层级解释：** DecoderLens（Vásquez et al.）；
- **符号域注意力可视化（主动引、正面差异化）：** Music Transformer（Huang et al. 2018）；
- **长曲动机工程（动机与痛点旁证）：** MusicWeaver（Motif Memory Retrieval）；
- **数据与评测：** POP909 及其结构标注；SALAMI；HookTheory；FAD/FD 音乐适配（Gui et al. 2024）；CLAP。

---

*本计划书整合了前期两轮文献侦察（约 40 篇 2024–2026 顶会/arXiv 工作）的收缩结论：主攻机制回路、SAE 仅作工具、benchmark 作为后续项目（MusicSteerBench）保留。核心研究问题一句话版：*

> *我们研究自回归音乐生成模型是否存在一种不同于文本 token-copy induction head 的 motif-level induction circuit；该回路能在 token 不完全重复的情况下支持主题再现、移调重复和节奏变奏，并可被因果干预用于控制重复度与缓解 loop 伪影。*
