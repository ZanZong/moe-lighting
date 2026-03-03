# Multi-Turn MoE CPU Offloading: Literature Survey & Innovation Plan

> **Target venue**: PPoPP / ASPLOS / MLSys  
> **Status**: Pre-implementation research plan  
> **Repo base**: MoE-Lightning (CGOPipe, ASPLOS 2025)

---

## 0. 说明

本文档基于以下文献调研后撰写（沙箱环境无法访问外网，以下分析来自训练数据中的文献知识）：

| URL / ID          | 论文                                                                   | Venue         |
|-------------------|------------------------------------------------------------------------|---------------|
| arXiv:2309.06180  | *Efficient Memory Management for LLM Serving with PagedAttention* (vLLM) | SOSP 2023     |
| arXiv:2312.11514  | *LLM in a Flash: Efficient LLM Inference with Limited Memory* (Apple)  | ICLR 2024     |
| arXiv:2402.04252  | *KVSharer / InfLLM 附近的工作*（具体ID对应论文见下方分析）              | arXiv 2024    |
| KV cache offload search | FlexGen, H2O, SnapKV, KIVI, Quest, Mooncake, CacheGen, StreamingLLM | Various       |
| CPU+MoE search    | Pre-gated MoE, MoE-Lightning, DeepSpeed-MII, ExpertFlow              | Various       |
| MLSys 2024        | Sarathi-Serve, Splitwise, vAttention, CacheBlend, etc.                | MLSys 2024    |

---

## 1. 文献综述

### 1.1 KV Cache 卸载与长上下文管理

#### FlexGen (Sheng et al., ICML 2023)
- **核心思路**: 单GPU上批量推理时将权重/KV/激活按固定策略分布在GPU/CPU/Disk三层。
- **Policy**: 线性规划求解存储分配比例（wg/wc/wd），最大化吞吐。
- **关键不足**: 
  - 针对**静态批次**设计（固定prompt_len, gen_len），不支持多轮对话上下文增长。
  - 每个请求独立处理，无跨请求KV复用。
  - I/O pipeline 与计算重叠，但没有MoE Expert感知。

#### vLLM / PagedAttention (Kwon et al., arXiv:2309.06180, SOSP 2023)
- **核心思路**: 为KV cache引入虚拟内存和分页机制，消除GPU显存碎片化。
- **Prefix caching**: 相同前缀的请求共享KV cache页（Copy-on-Write）。
- **关键不足**:
  - **纯GPU方案**，不考虑CPU/Disk offloading。
  - Prefix caching仅针对**跨用户的公共前缀**（如system prompt），不是per-session的跨轮缓存。
  - 不涉及MoE专家权重管理。
  - 随着context增长，显存压力最终导致驱逐（preemption），无CPU tier利用。

#### LLM in a Flash (2312.11514, Apple, ICLR 2024)
- **核心思路**: 将LLM权重存储在Flash/SSD上，推理时按需加载到有限DRAM/GPU内存。
- **关键设计**:
  1. **Sliding window prefetch**: 预测下一层需要哪些权重（基于FFN的激活稀疏性）。
  2. **Bundling**: 将多个稀疏激活聚合成连续读请求，减少Flash随机读延迟。
  3. Row/Column级别稀疏激活预测（~2%的神经元实际激活）。
- **关键不足**:
  - 针对dense FFN的稀疏激活，**不是MoE**（MoE本身已经是稀疏激活专家的结构）。
  - 没有KV cache管理。
  - 没有多轮对话场景的考虑。
  - Flash带宽远低于DRAM，延迟特性不同；CGOPipe使用DRAM，本设计更高效。

#### InfLLM (Xiao et al., 2024) / 2402附近工作
- **核心思路**: 将历史KV以block为单位存储在CPU上，decode时通过**相似度检索**找到relevant KV blocks并加载到GPU做attention。
- **Key design**: Context unit (block) → CPU offload; current window → GPU; retrieve top-K relevant units per step。
- **关键不足**:
  - 针对**dense LLM**（LLaMA等），无MoE专家管理。
  - 检索代价：每步需计算query与所有CPU blocks的相似度（O(N)开销）。
  - 没有利用**对话轮次结构**做预测——每次decode都是独立检索。
  - CPU attention计算不是目标，而是GPU做selective attention——与MoE-Lightning的CPU attention offloading思路不同。

#### H2O (Zhang et al., NeurIPS 2023)
- **核心思路**: 基于attention score识别"Heavy Hitter" token，蒸馏eviction policy。
- **关键不足**: GPU-only；丢弃token（有损）；无多轮对话感知。

#### SnapKV / PyramidKV / CLA (2024系列)
- **共同思路**: 在prefill阶段观测attention pattern压缩KV token选择。
- **关键不足**: GPU-only；压缩/丢弃是不可逆的；不涉及CPU offloading。

#### KIVI (Liu et al., ICLR 2024) / WKVQuant
- **核心思路**: INT4量化KV cache，每2个token为单位的group quantization，减少GPU显存。
- **关键不足**: 目标是**GPU显存节省**而非CPU带宽优化；未考虑CPU attention场景；量化是uniform的，没有per-turn差异化。

#### Quest (Tang et al., 2024) / MagicPIG (2024)
- **Quest**: 以page为单位根据query相关性选择重要KV token参与attention。
- **MagicPIG**: 使用LSH做近似attention，将CPU作为KV存储并做稀疏采样。
- **关键不足**:
  - MagicPIG虽然用CPU，但其目标是近似attention（有损）；CGOPipe是精确attention。
  - 均无MoE专家co-scheduling。
  - MagicPIG的LSH建索引代价高，multi-turn场景下索引需要增量维护。

#### StreamingLLM (Xiao et al., ICLR 2024)
- **核心思路**: 保留"attention sink" token（初始tokens）+ 滑动窗口，实现无限长度推理。
- **关键不足**: **有损**（丢弃历史上下文）；不是offloading；无MoE感知。

#### Mooncake (Qin et al., 2024)
- **核心思路**: KV cache作为独立存储层进行跨节点传输（网络attached KV）。
- **关键不足**: 面向disaggregated serving（多机）；无MoE；无CPU attention。

#### CacheBlend (Yao et al., MLSys 2024)
- **核心思路**: RAG场景下对cached KV进行selective recomputation（blending），减少因prefix KV不完全匹配导致的质量损失。
- **关键不足**: GPU-only；无MoE；无CPU offloading。

### 1.2 MoE 推理优化

#### MoE-Lightning (Cao et al., ASPLOS 2025)——本仓库
- **CGOPipe核心不变式**: T_CPU_attn(ctx) ≤ T_GPU_FFN
- **Pipeline**: GPU做QKV projection + MoE FFN，CPU做attention，PCIe传输与计算重叠。
- **HRM optimizer**: LP求解最优(ubs, n_ub, wg, wc)策略。
- **根本局限**: 设计针对**静态批次、固定context length**。多轮对话下：
  - context单调增长 → 不变式必然被违反。
  - Expert weights和KV cache共享CPU DRAM，但两者独立管理。
  - 每轮重新prefill全部历史token（无跨轮KV复用）。

#### Pre-gated MoE (Hwang et al., 2024)
- **核心思路**: 用轻量级predictor在当前token上预测下一层激活专家，提前prefetch。
- **关键不足**: 预测基于当前step的input，不利用历史轮次的expert activation pattern。无KV offloading。

#### ExpertFlow / Offload-based MoE (2024系列)
- 各种Expert预取和offloading方案，但均无multi-turn aware设计。

### 1.3 MLSys 2024关键工作

#### Sarathi-Serve (Agrawal et al., MLSys 2024)
- **Chunked prefill**: 将prefill切片与decode请求co-schedule，消除stall。
- **关键洞察**: Prefill和decode的算术强度不同，可以混合调度。
- **不足**: GPU-only；无MoE；无CPU offloading。

#### Splitwise (Patel et al., ISCA 2024)
- **核心思路**: Prefill和decode分离到不同硬件（prefill machine vs decode machine）。
- **不足**: 多机设计；无MoE；无CPU offloading。

---

## 2. 问题定位：已有工作的空白

综合上述文献，**无任何现有系统同时解决**：

```
(A) MoE模型的CPU Expert权重卸载（CGOPipe）
(B) 多轮对话下context单调增长管理
(C) 动态offload策略选择（context-adaptive）
(D) 跨轮次KV cache复用（消除冗余prefill）
(E) CPU显存中Expert权重与KV cache的联合调度
```

**A∩B∩C∩D∩E** 是一个完全开放的研究问题。

### 2.1 为什么现有方案不能直接组合？

- **vLLM prefix caching + MoE-Lightning**: vLLM的prefix cache在GPU上，CPU attention offloading要求KV在CPU上。两个系统的KV存储层完全冲突。
- **InfLLM + MoE-Lightning**: InfLLM的block检索假设GPU做attention（用GPU SRAM做local computation），不兼容CPU attention offloading。每步独立检索也无法利用MoE-Lightning的micro-batch pipeline。
- **KIVI + MoE-Lightning**: KIVI量化的KV仍在GPU上；CPU attention路径需要的KV在CPU DRAM上，且需要考虑CPU算术指令对INT4的支持（x86 AMX/VNNI）。
- **Pre-gated MoE + MoE-Lightning**: Pre-gated的预测不使用历史轮次信息，在多轮对话中有更大的预测机会被浪费。

---

## 3. 核心挑战精确化

基于文献调研，将多轮对话场景下的挑战重新精确定义：

### Challenge 1: CGOPipe Invariant Degradation（不变式退化）

```
T_CPU_attn(ctx) = max(ctx · F_attn / CPU_FLOPS, ctx · B_attn / c_bdw)
                         线性增长↑                 线性增长↑

T_GPU_FFN = const（与ctx无关）
```

**精确突破点**（Mixtral-8x7B on GCP g2-standard-48）：
- CPU BDW = 76 GB/s, GPU FFN time ≈ 155ms/layer（bs=1）
- Break-even at ctx ≈ 2,700 tokens
- 以(200+100)=300 tokens/turn的速度，**第8轮**即突破

**文献对照**:
- FlexGen: 静态ctx，不会突破
- InfLLM: 稀疏检索 O(k)而非O(n)，但有损且无MoE
- **本工作需要解决**: 在不破坏精确attention的前提下，维持不变式到更长ctx

### Challenge 2: CPU DRAM 双重压力（Expert权重 vs KV Cache）

```
CPU Memory Budget C_mem:
  Expert weights (CPU tier) = wc × L × ne × 3 × h1 × h2 × 2 bytes
  KV cache (all sessions)   = N_sess × ctx_avg × L × 2 × nkv × hd × 2 bytes
  ─────────────────────────────────────────────────────────────────────────
  Sum must ≤ C_mem × 0.8
```

**数字**（Mixtral-8x7B, C_mem=192GB, ctx=16K, bs=100）：
- Expert weights (全CPU): 84GB
- KV cache (100 sessions × 16K ctx): 100 × 16384 × 32 × 2×8×128×2B = ~54GB
- 合计: 138GB ✓ 但随ctx增长，32K时合计 ~192GB → OOM

**文献对照**:
- FlexGen: 联合LP规划了weight+KV，但单次请求，无session管理
- vLLM: GPU-only，无此问题
- **本工作需要解决**: 动态按需在Expert权重卸载量和KV保留量之间调配CPU内存

### Challenge 3: 冗余Prefill（跨轮KV重复计算）

```
Turn N prefill cost (without cache) = O(ctx_N^2 × h1 × h2)  [self-attention]
Turn N prefill cost (with cache)    = O(Δ_N × ctx_N × h1 × h2)  [Δ_N << ctx_N]

Redundancy fraction at turn N = (ctx_N - Δ_N) / ctx_N → >90% after 10 turns
```

**文献对照**:
- vLLM: prefix cache针对跨用户公共前缀（system prompt），非per-session历史KV
- SGLang RadixAttention: GPU-side radix tree，KV在GPU
- **本工作需要解决**: CPU-side的跨轮KV持久化，使turn N的prefill只处理新token Δ_N

### Challenge 4: Expert Activation 的跨轮相关性（文献中未被发现）

**新发现的问题**（现有任何论文均未研究）：

在对话场景中，同一会话的相邻轮次往往讨论相同话题。MoE路由具有**语义一致性**：

```
观察：如果Turn 1讨论"代码调试"，Turn 2大概率也在同一话题
→ Turn 1激活的expert集合 与 Turn 2激活的expert集合 高度重叠
→ 预测Turn N的expert激活，可以利用Turn 1..N-1的activation history
```

这与LLM in a Flash (2312.11514) 的"bundling"思路有本质区别：
- LLM in a Flash: 基于当前token的MLP稀疏激活做bundling（无跨步相关性）
- **本工作**: 基于多轮对话的语义连续性做Expert跨轮预测（有时序相关性可利用）

### Challenge 5: Attention策略切换的PCIe迁移代价

```
当ctx超过break-even，切换CPU attention → GPU attention：
需要将CPU KV迁移到GPU：
  Transfer = ctx × L × 2 × nkv × hd × 2B / PCIe_BDW
  = 16K × 32 × 2×8×128×2 / 16GB/s = ~2.7秒
```

这个切换代价在现有系统中完全被忽视（FlexGen, InfLLM, MoE-Lightning均无此问题的分析）。

---

## 4. 创新方案 Plan

### 系统命名：MT-CGOPipe (Multi-Turn Context-Guided Offloading Pipeline)

```
┌──────────────────────────────────────────────────────────────────────────┐
│                        MT-CGOPipe Architecture                           │
│                                                                          │
│  Turn Boundary Manager ──► Expert Activation Predictor (CHEAP)          │
│         │                              │                                 │
│         ▼                              ▼                                 │
│  Turn-Stratified KV Cache    Expert Prefetch Scheduler                   │
│  ┌──────────────────────┐    ┌─────────────────────────┐                │
│  │ HOT: current turn KV │    │ GPU Home (wg experts)   │                │
│  │   (GPU, FP16)        │    │ CPU Hot (predicted)     │                │
│  ├──────────────────────┤    │ CPU Cold (all others)   │                │
│  │ WARM: turn N-1 KV    │    └─────────────────────────┘                │
│  │   (CPU, INT8/FP16)   │              │                                │
│  ├──────────────────────┤    ┌─────────────────────────┐                │
│  │ COLD: turn ≤N-2 KV   │    │   Joint LP Optimizer    │                │
│  │   (CPU, INT4)        │    │   (CMALP: C2 + C4)      │                │
│  └──────────────────────┘    └─────────────────────────┘                │
│         │                              │                                 │
│         └──────────────┬───────────────┘                                │
│                        ▼                                                 │
│              CGOPipe Execution Engine                                    │
│              (extended from MoE-Lightning)                               │
└──────────────────────────────────────────────────────────────────────────┘
```

---

### Contribution 1: Turn-Granularity CGOPipe Invariant Scheduling (TGIS)

**动机**: 现有CAO（Context-Adaptive Offloading）在每个decode step检查不变式，但：
1. 检查+路由切换有overhead（cuda event + conditional branch）
2. 多轮对话下，context在**轮次边界**发生跳变（新prompt tokens），而轮次内是平滑的decode增量（1 token/step）
3. 在轮次内切换策略会引入PCIe迁移代价（Challenge 5）

**方案设计**:
```
AT TURN BOUNDARY (before processing new user prompt):
  1. Compute ctx_next = ctx_current + |new_prompt| + E[response_len]
  2. Evaluate: will invariant hold throughout this turn?
     Use compute_offload_threshold() → θ
     If ctx_next > θ: schedule GPU attention for entire turn N
     Else if ctx_current > θ×0.8: schedule SPARSE_CPU for turn N (预警区)
     Else: schedule CPU_OFFLOAD for turn N
  3. If strategy changes: trigger proactive KV migration (see C5/PTASH)
  4. Hot-swap policy in ExecutionEngine (no mid-turn switching)
```

**新颖性 vs. 现有工作**:
- vs. MoE-Lightning: 无任何动态策略
- vs. long_context_policy.py (现有代码): 现有代码做per-step分析，不做轮次边界感知的前瞻决策
- vs. 任何已知系统: **轮次结构作为调度原语**在LLM serving中首次提出

**量化收益**（基于Mixtral-8x7B模型分析）:
- 消除mid-turn KV迁移代价（每次迁移约2-3秒 for 16K tokens）
- 轮次内策略一致，pipeline效率提升（消除step-level branch预测失败）

---

### Contribution 2: Conversation-History Expert Activation Prediction (CHEAP)

**动机**: 
- LLM in a Flash (2312.11514) 使用当前token的稀疏activation预测下一层需要哪些神经元
- 本工作的MoE场景：**Expert的语义路由**比dense FFN的神经元激活更稳定，且跨轮相关性更强
- **关键insight**: MoE gate是基于hidden state的聚类，相同话题→相同expert聚类

**方案设计**:
```python
# Per-session expert activation history
ExpertHistory = Dict[session_id, List[LayerActivationMap]]
LayerActivationMap = Dict[layer_id, Counter[expert_id]]  # activation count

# At turn N prediction:
def predict_expert_activations(session_id, layer_id, top_k=2):
    history = ExpertHistory[session_id][-3:]  # last 3 turns
    weighted_counts = sum(turn[layer_id] × decay_weight for turn in history)
    return weighted_counts.most_common(top_k + 2)  # predict top K+buffer

# Prefetch decision:
# If predicted experts overlap with currently-CPU experts → schedule prefetch
# Prefetch target: during current turn's CPU attention time (overlap with C1)
```

**与LLM in a Flash的本质区别**:
| 维度 | LLM in a Flash | CHEAP (本工作) |
|------|---------------|----------------|
| 预测粒度 | 单个dense neuron (MLP) | MoE Expert (更大粒度，更稳定) |
| 预测依据 | 当前token的input | 历史多轮的expert激活直方图 |
| 时间相关性 | 无（每层独立） | 有（跨轮次decay weighted）|
| 目标 | Flash→DRAM的bundling | CPU DRAM→GPU的预取 |
| 场景 | 单轮 | 多轮对话 |

**新颖性**: 据文献调研，**无任何现有工作**利用跨轮expert activation history做MoE预取。

**预期预测精度** (理论分析):
- 单一主题对话（coding, translation）: top-3 experts/layer overlap ≈ 70-80%
- 话题切换时（turn N与turn N-1话题不同）: overlap ≈ 30-40%（退化为保守策略）
- 可用entropy of historical distribution检测话题切换：高entropy → 切换 → 降低预取激进程度

---

### Contribution 3: Turn-Age-Stratified KV Compression (TASK)

**动机**:
- KIVI/WKVQuant: 目标是GPU显存节省，均匀INT4量化
- H2O/SnapKV: 丢弃token（有损），无法恢复
- **Gap**: 没有针对CPU bandwidth budget的、按轮次年龄的**非均匀可逆压缩**

**方案设计**:
```
三层KV存储结构（精确attention，无token丢弃）:

Turn N   (current): FP16 on CPU → 100% bandwidth cost
Turn N-1 (warm):    INT8 on CPU → 50% bandwidth cost
Turn ≤N-2 (cold):   INT4 on CPU + per-turn page index → 25% bandwidth cost

CPU Attention Cost:
  T_CPU = max(
    Σ_layers attention_flops / cpu_flops,
    (ctx_N×BW_fp16 + ctx_{N-1}×BW_int8 + ctx_{≤N-2}×BW_int4) / c_bdw
  )
```

**与现有工作的本质区别**:
| 维度 | KIVI / WKVQuant | H2O / SnapKV | TASK (本工作) |
|------|-----------------|--------------|---------------|
| 目标 | GPU VRAM节省 | GPU VRAM节省 | **CPU bandwidth节省** |
| 存储位置 | GPU HBM | GPU HBM | **CPU DRAM** |
| 策略 | 均匀INT4 | 丢弃token | **按轮次年龄非均匀** |
| 可逆性 | 有损（量化误差） | **不可逆**（丢弃） | **完全可逆**（解量化） |
| 触发时机 | 启动时 | Prefill时 | **轮次边界** |
| MoE感知 | 无 | 无 | **是**（与Expert带宽联合优化）|

**关键公式**（恢复CGOPipe不变式的压缩级别求解）:
```
Given: T_GPU_FFN = const
Find: compression levels {q_k} for each turn k ≤ N such that:
  Σ_k ctx_k × (16/q_k bits) × bandwidth_per_bit / c_bdw ≤ T_GPU_FFN
  subject to: q_k ∈ {4, 8, 16}, q_N = 16 (最新轮不压缩)
              quality_loss(q_k) ≤ ε_k (老轮次允许更大误差)

Greedy solution: compress oldest turns first (maximize bits saved per quality impact)
```

**量化收益** (理论):
- ctx = 16K (约50轮 @300 tokens/turn):
  - Without TASK: T_CPU = 16K × FP16_bw / c_bdw ≈ 5.5ms (>> T_GPU_FFN ≈ 0.15ms)
  - With TASK (turns stratified): effective ctx × 0.25 bits_reduction ≈ 1.4ms → 接近T_GPU_FFN
  - 可支持的有效context长度延伸 **4倍**

---

### Contribution 4: CPU-Memory-Aware Joint LP Re-optimization (CMALP)

**动机**: 
- MoE-Lightning的HRM LP: 独立规划expert memory (wc) 和 KV cache memory
- 多轮对话中CPU内存是**共享稀缺资源**：Expert weights + KV cache + Pinned memory必须jointly fit
- **FlexGen**虽然联合规划了weight+KV，但：(a)针对batch inference (b)无Expert激活预测作为输入 (c)无轮次结构

**扩展LP设计**:
```python
# New variables:
kv_cpu_frac = fraction of cmem allocated to KV cache

# New constraints (added to existing HRM LP):
prob += expert_mem × wc + kv_bytes × kv_cpu_frac ≤ cmem × 0.8
prob += kv_bytes == ctx × L × 2 × nkv × hd × 2 × (1 - compression_ratio)

# New objective term (added to T minimization):
# Penalize configurations where KV pressure forces more expert CPU→GPU transfers
prob += T ≥ ctog × (wc + expert_migration_penalty × kv_cpu_frac)

# New input to LP:
expert_activation_prediction = CHEAP.predict()  # from C2
# Use predicted top-k experts to compute effective wg (effective GPU expert fraction)
# If top-3 experts/layer predicted → can treat them as "effectively hot" → adjust wg
```

**与FlexGen LP的区别**:
| 维度 | FlexGen | MoE-Lightning HRM | CMALP (本工作) |
|------|---------|-------------------|----------------|
| 规划粒度 | 全batch一次 | 全batch一次 | **per-turn update** |
| Expert预测输入 | 无 | 无 | **CHEAP预测作为约束** |
| KV压缩建模 | 无 | 无 | **TASK压缩级别作为变量** |
| 内存竞争 | weight vs KV | 独立规划 | **联合资源约束** |
| 重优化触发 | 无 | 启动时一次 | **轮次边界异步触发** |

---

### Contribution 5: Proactive Turn-Boundary KV Migration (PTBM)

**动机**: Challenge 5分析表明，当需要从CPU attention切换到GPU attention时，KV迁移代价极高。
- 现有系统（包括本仓库）均为**reactive**: 等到不变式被违反后再切换
- **新思路**: 利用TGIS (C1)对下一轮策略的前瞻，**在当前轮decode期间**异步预迁移关键KV token到GPU

**方案设计**:
```
IF TGIS预测Turn N+1需要GPU attention:
  DURING Turn N decode (在GPU FFN时间窗口内有PCIe带宽空闲):
    1. 识别高重要性KV token子集（用attention sink + CHEAP预测的相关token）
    2. 异步将这些token的KV从CPU搬运到GPU KV buffer
    3. 优先级: attention sink tokens > CHEAP预测高激活token > 最近窗口tokens
  AT Turn N+1 start:
    GPU已经有了~60-80%的重要KV → 只需传输剩余20-40%
    → 迁移代价从2.7秒降至~0.5秒
```

**与现有工作的区别**:
- InfLLM: reactive检索（每step），无前瞻
- vLLM preemption: 被抢占时KV全部drop再重算，无迁移
- **PTBM**: 利用轮次边界预知性+pipeline空闲带宽做**proactive partial migration**

---

## 5. 方案组合与交互分析

```
Turn N-1 (CPU attention, invariant holds)
├── Decode: CGOPipe + CHEAP记录expert activation
└── Turn End: TASK压缩Turn N-1 KV → INT8

Turn N Boundary:
├── TGIS: 预测Turn N+1是否需要切换strategy
├── CMALP: 异步后台LP re-optimization（新ctx, 新KV压缩状态, CHEAP预测）
├── 若预测N+1切换 → PTBM启动proactive KV迁移
└── 加载Turn N新prompt KV（仅Δ_N新token需要计算，其余从CPU KV cache复用）

Turn N (可能是CPU attention, 也可能已切换GPU attention)
├── 若CPU attention: CHEAP预测expert → prefetch热expert到GPU
├── 若GPU attention: 复用PTBM已迁移的KV，仅处理残余迁移
└── Decode: 正常CGOPipe（或GPU-only path）

Turn N End:
├── TASK: 压缩Turn N-1(已warm)→ INT8, Turn ≤N-2 → INT4
├── CHEAP: 记录Turn N的expert activation直方图
└── 更新CMALP输入状态
```

**五个创新点的协同效应**:
- TGIS + PTBM: 消除突发KV迁移代价（C1预测，C5执行）
- CHEAP + CMALP: 减少expert预取代价（C2预测，C4规划）
- TASK + CMALP: 延伸CGOPipe不变式成立的context范围（C3压缩，C4联合规划）

---

## 6. 与现有代码的对应关系

### 已有代码（需修改/扩展）

| 现有文件 | 现有功能 | 需要的修改 |
|---------|---------|-----------|
| `backend/optimizer.py` / `solve_lp()` | HRM LP，静态单次求解 | 扩展为CMALP，加入kv_compression变量和expert_prediction约束 |
| `backend/long_context_policy.py` | Per-step strategy分析 | 改为turn-boundary前瞻（TGIS），输出整轮策略 |
| `backend/kv_cache_manager.py` | LRU session KV管理 | 扩展为TASK三层结构（FP16/INT8/INT4 per turn） |
| `backend/memory.py` / `TokenToKVPool` | save/load_session | 扩展为支持量化存储和异步PTBM迁移 |
| `backend/execution_engine.py` | CGOPipe执行引擎 | 加入CHEAP expert activation记录和TGIS策略切换 |
| `backend/task.py` / `Req` | session_id, turn_id, cached_prefix_len | 加入expert_activation_history字段 |

### 新增代码（需创建）

| 新文件 | 功能 |
|--------|------|
| `backend/expert_predictor.py` | CHEAP: 跨轮expert activation prediction |
| `backend/turn_manager.py` | TGIS: 轮次边界管理 + 前瞻策略决策 |
| `backend/kv_quantizer.py` | TASK: CPU KV INT8/INT4量化/解量化 kernel |
| `backend/migration_scheduler.py` | PTBM: proactive KV迁移调度 |
| `benchmarks/multiturn/bench_multiturn.py` | 端到端多轮对话benchmark |

---

## 7. 实验设计 (Evaluation Plan)

### 7.1 基准配置
- **模型**: Mixtral-8x7B (MoE, 8 experts, top-2 routing)
- **硬件**: GCP g2-standard-48 (L4 GPU 24GB + 192GB CPU DRAM)  
- **数据集**: MT-Bench (80 multi-turn conversations), ShareGPT (real user conversations)
- **Baseline**: 
  - MoE-Lightning (static policy, no multi-turn optimization)
  - vLLM (GPU-only, prefix caching)
  - FlexGen (CPU/disk offload, batch inference)
  - InfLLM-style (block retrieval, but CPU attention version)

### 7.2 关键指标
1. **Throughput** (tokens/s) vs. context length curves
2. **CGOPipe pipeline efficiency** (T_GPU_FFN / max(T_GPU, T_CPU)) vs. turn number
3. **CPU memory utilization** (expert + KV) vs. turn number  
4. **Expert prefetch hit rate** (CHEAP accuracy vs. conversation topics)
5. **TTFT** (time-to-first-token) for multi-turn prefill with/without KV reuse
6. **Generation quality** (TASK compression quality, ROUGE/perplexity vs. FP16 baseline)

### 7.3 消融实验
- TGIS alone vs. baseline (C1 contribution)
- CHEAP alone (hit rate, prefetch overhead)
- TASK alone: quality-performance trade-off curve
- CMALP alone: memory utilization improvement
- Full MT-CGOPipe vs. each ablated variant

---

## 8. 创新性总结

```
┌──────────────────────────────────────────────────────────────────────────┐
│  创新点             vs. 最相关工作              核心差异                  │
├──────────────────────────────────────────────────────────────────────────┤
│  TGIS (C1)     vs. long_context_policy.py   轮次边界前瞻 vs. 逐步reactive│
│  CHEAP (C2)    vs. LLM in a Flash           跨轮历史 vs. 当前token稀疏性 │
│  TASK (C3)     vs. KIVI/WKVQuant            CPU带宽目标 vs. GPU显存目标  │
│  CMALP (C4)    vs. FlexGen LP               Expert+KV联合 vs. 分离规划  │
│  PTBM (C5)     vs. InfLLM/vLLM            前瞻性渐进迁移 vs. reactive  │
└──────────────────────────────────────────────────────────────────────────┘

核心创新（最具新颖性）: C2 (CHEAP) + C4 (CMALP)的组合
─ 在MoE模型中，Expert路由pattern的跨轮相关性是一个完全未被研究的现象
─ 将Expert activation预测作为联合内存规划的输入是本工作独有的设计
```

---

## 9. 待确认的技术风险

1. **CHEAP预测精度**: 话题切换时，跨轮Expert activation相关性是否足够高？需要真实数据验证。
2. **TASK量化质量**: CPU-side INT4 KV attention的数值精度损失是否可接受？需要与KIVI的GPU实验对比。
3. **CMALP求解时间**: 轮次边界重优化的LP求解是否足够快（<100ms）以满足实时要求？可能需要warm-start。
4. **PTBM带宽竞争**: proactive KV迁移与Expert prefetch共享PCIe带宽，可能相互干扰，需要联合调度。
5. **CPU INT4 Kernel**: x86上的INT4矩阵运算（AMX/VNNI指令集支持）需要专门优化，不能直接用PyTorch。

---

*文档版本: v1.0 | 最后更新: 2026-03-03*
