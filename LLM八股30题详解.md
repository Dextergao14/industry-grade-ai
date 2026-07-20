# 大模型算法岗八股 30 题详解

> 用法：一道道学，遇到想深入的告诉我题号。每题给了**面试可直接说的核心答案**、**追问点**、和**容易答错的坑**。
> 标 ⭐ 的是几乎必考、且你（RL/CV 背景）最该优先吃透的题。

---

## 一、架构类

### 1. ⭐ 为什么用 Pre-norm 而不是 Post-norm？

**核心答案**：
- Post-norm（原始 Transformer）：`x + Sublayer(x)` 之后再 LayerNorm，即 `LN(x + Sublayer(x))`。
- Pre-norm（现代 LLM 主流）：先 norm 再进子层，`x + Sublayer(LN(x))`。
- Pre-norm 训练更稳定，因为它给了一条**干净的恒等映射残差通路**（identity path）：梯度可以从顶层直接流到底层而不被反复缩放，避免深层网络的梯度爆炸/消失。Post-norm 在深层时每一层的输出都被 norm 重新缩放，残差信号被削弱，深层难训练，必须配合 warmup 和小学习率。

**为什么 Pre-norm 稳**：可以推导出 Pre-norm 网络中，第 L 层的输出方差大约随层数线性增长，梯度通路上没有连乘的不稳定项；Post-norm 则有连乘项，深度一大就炸。

**追问点**：
- Post-norm 是不是一无是处？不是。Post-norm 表达能力理论上更强（同等深度效果上限更高），所以一些工作（如 BERT、原始 Transformer）用它。代价是难训。
- 折中方案：DeepNorm（微软，让 Post-norm 也能训很深）、Sandwich-norm（前后都加）。
- 现在新模型还有 **QK-norm**（对 Q、K 单独做 norm，进一步稳定 attention，Qwen2、Gemma2 在用）。

**坑**：别只背"Pre-norm 更稳"，要能说出"为什么"——残差恒等通路 + 梯度不被反复缩放。

---

### 2. ⭐ RMSNorm vs LayerNorm 的差异和动机

**核心答案**：
- LayerNorm：`y = γ * (x - μ) / sqrt(σ² + ε) + β`，要算均值 μ 和方差 σ²，做**重新中心化（re-centering）+ 重新缩放（re-scaling）**。
- RMSNorm：`y = γ * x / sqrt(mean(x²) + ε)`，**只做 re-scaling，去掉了减均值和 β 偏置**。RMS = Root Mean Square = `sqrt(mean(x²))`。
- 动机：实验发现 LayerNorm 的好处主要来自 re-scaling 而非 re-centering。去掉均值计算后，**计算量减少（少一次求均值、少一个减法）、参数更少（无 β）**，速度提升 7%-64%（论文数据），效果基本不掉。

**追问点**：
- 为什么 re-centering 不重要？因为在高维下，减均值带来的好处有限，且后续的线性层可以吸收掉中心偏移。
- 谁在用：Llama 全系、Qwen、Mistral、几乎所有现代 LLM 都用 RMSNorm。
- 计算位置：通常 RMSNorm 用 FP32 计算以保证数值稳定，再转回 BF16。

**坑**：RMSNorm 不是"简化版 LayerNorm 所以更差"，而是发现冗余后的合理简化，效果不掉、速度更快。

---

### 3. ⭐ RoPE vs ALiBi vs 绝对位置编码

**核心答案**：

| 方法 | 原理 | 优点 | 缺点 |
|---|---|---|---|
| 绝对位置编码（Sinusoidal/Learned） | 给每个位置加一个位置向量到 embedding | 简单 | 外推差，位置是"加"上去的，相对关系不显式 |
| RoPE（旋转位置编码） | 把 Q、K 按位置旋转一个角度，旋转角与位置成正比 | 相对位置天然编码进 attention，外推性较好，主流选择 | 纯 RoPE 长度外推仍有限，需要插值/NTK 扩展 |
| ALiBi | 不加位置编码，直接在 attention score 上按 query-key 距离加一个线性递减的 bias | 外推极好，无需训练就能处理更长序列 | 表达能力上限略低，现在用得少 |

**RoPE 核心机制**（必须能讲）：
- 把 d 维向量两两配对成 d/2 个二维子空间，每个子空间按位置 m 旋转一个角度 `m·θ_i`，其中 `θ_i = 10000^(-2i/d)`（不同维度旋转频率不同，低维转得快、高维转得慢）。
- 关键性质：`<RoPE(q,m), RoPE(k,n)>` 只依赖于**相对位置 (m-n)**，所以相对位置信息天然进入了内积（即 attention score）。

**RoPE 长度外推**（高频追问，2024-2025 必问）：
- Position Interpolation (PI)：把位置 index 线性压缩，让训练时没见过的长位置映射回训练范围内。
- NTK-aware scaling：调整 base（那个 10000），高频维度少插值、低频维度多插值，效果比线性 PI 好。
- YaRN：NTK 的改进版，分频段处理，Qwen 等在用。

**追问点**：为什么 RoPE 外推会崩？因为推理时遇到训练时没见过的旋转角度组合，高频维度的旋转"转过头了"，模型没见过这种 pattern。

**坑**：能讲"RoPE 把相对位置编码进了内积"是及格线，能讲清楚 base/θ 和外推方法是加分线。

---

### 4. ⭐ GQA/MQA/MHA 的权衡，KV cache 如何计算

**核心答案**：
- **MHA（Multi-Head Attention）**：每个 head 有独立的 Q、K、V。h 个 head 就有 h 组 KV。
- **MQA（Multi-Query Attention）**：所有 head 共享**同一组** K、V，只有 Q 是多头的。KV cache 缩小 h 倍，但效果有损失。
- **GQA（Grouped-Query Attention）**：折中。把 h 个 head 分成 g 组，每组共享一组 KV。g=h 时退化成 MHA，g=1 时退化成 MQA。Llama2-70B、Llama3、Qwen2 都用 GQA。

**为什么需要它们**：推理时 KV cache 是显存和带宽瓶颈。减少 KV head 数量直接减小 cache，提升推理吞吐（尤其长序列、大 batch）。

**KV cache 显存计算公式**（必须会算）：
```
KV cache 大小 = 2 (K和V) × batch_size × seq_len × num_kv_heads × head_dim × num_layers × bytes_per_param
```
举例：Llama2-7B，MHA，32 层，32 head，head_dim=128，BF16(2 bytes)，batch=1，seq=4096：
```
2 × 1 × 4096 × 32 × 128 × 32 × 2 = 约 2 GB
```
如果是 GQA（8 个 KV head）：`2 × 1 × 4096 × 8 × 128 × 32 × 2 ≈ 0.5 GB`，省了 4 倍。

**追问点**：
- GQA 为什么效果接近 MHA？因为 attention 中 KV 的冗余度高，分组共享损失小。
- MLA（DeepSeek）是更激进的方案，见第 7 题。

**坑**：KV cache 公式里的 `2`（K 和 V 两份）和 `num_kv_heads`（不是 num_heads，GQA 下不同）别搞混。

---

### 5. SwiGLU 相比 ReLU/GELU 的优势

**核心答案**：
- 标准 FFN：`FFN(x) = W2 · activation(W1·x)`，activation 用 ReLU 或 GELU。
- GLU 家族：引入**门控**。`GLU(x) = (W1·x) ⊗ σ(W2·x)`，⊗ 是逐元素乘，一路做内容、一路做门控。
- SwiGLU：门控用 Swish/SiLU 激活：`SwiGLU(x) = (W1·x) ⊗ Swish(W3·x)`，然后再过 W2。
- 优势：门控机制让网络能动态控制信息流，实验上效果优于 ReLU/GELU FFN。Llama、PaLM 等都用。

**实现细节**（追问点）：
- SwiGLU 多了一个权重矩阵（W1、W2、W3 三个，而普通 FFN 只有 W1、W2 两个）。为保持参数量一致，通常把中间维度从 4d 缩到 8d/3 ≈ 2.67d。
- Swish(x) = x · sigmoid(βx)，β=1 时即 SiLU。

**坑**：PaLM 论文那句名言——"我们把效果提升归因于神之恩典"（实际是说这些激活函数的优势缺乏强理论解释，主要靠实验）。面试别强行编理论，承认"经验上更好、理论解释不充分"反而显得诚实懂行。

---

### 6. ⭐ MoE 的 routing 和 load balance loss

**核心答案**：
- MoE（Mixture of Experts）：把 FFN 替换成 N 个并行的 expert（每个是一个 FFN），用一个 **router/gating 网络**为每个 token 选 top-k 个 expert（通常 k=1 或 2）激活。
- 好处：**参数量大但激活量小**。比如 DeepSeek-V3 总参数 671B，但每个 token 只激活 37B。算力随激活参数走，容量随总参数走，性价比高。

**Routing 机制**：
- router 是个小线性层，对每个 token 输出 N 个 expert 的分数，softmax 后选 top-k。
- token 被发给选中的 expert，输出按 gating 权重加权求和。

**Load balance loss（核心难点）**：
- 问题：router 容易"偏心"，总把 token 发给少数几个 expert，导致其他 expert 训不到、且负载不均拖慢训练（热门 expert 排队）。
- 解决：加 **auxiliary load balancing loss**，惩罚负载不均，鼓励 token 均匀分布到各 expert。经典形式：`L_aux = α · N · Σ(f_i · P_i)`，f_i 是分到 expert i 的 token 比例，P_i 是 router 给 expert i 的平均概率。
- DeepSeek-V3 的创新：**auxiliary-loss-free** 负载均衡，用一个可学习的 bias 调整 routing 而不引入额外 loss（避免 aux loss 干扰主任务）。

**追问点**：
- Expert capacity / token dropping：每个 expert 有容量上限，超了的 token 被丢弃（或绕过）。
- Shared expert（DeepSeek）：保留一两个所有 token 都过的共享 expert，专门学通用知识，其余 expert 学专精知识。

**坑**：能讲"为什么要 load balance"（防止 router 偏心 + 负载均衡加速训练）是关键，光说"加个 aux loss"不够。

---

### 7. ⭐ MLA（DeepSeek-V2/V3）相对 GQA 的改进

**核心答案**：
- MLA = Multi-head Latent Attention。目标和 GQA 一样：减小 KV cache。但思路不同。
- GQA 通过**减少 KV head 数量**省 cache（粗暴共享）。
- MLA 通过**低秩压缩**省 cache：把 K、V 联合压缩到一个低维的 latent 向量 `c`，**cache 只存这个低维 c**，用时再通过上投影矩阵恢复出完整的 K、V。
- 效果：cache 比 GQA 还小，但因为是低秩压缩而非直接共享，**信息损失更少，效果接近甚至超过 MHA**。

**关键技巧**（追问点）：
- 上投影矩阵可以被吸收（absorb）进其他权重矩阵，所以推理时不用真的恢复完整 KV，进一步省算力。
- RoPE 和 MLA 有冲突（RoPE 的旋转和低秩压缩不兼容），DeepSeek 的解法是**解耦 RoPE**：一部分维度专门走 RoPE（decoupled key），其余维度走低秩压缩。这个细节问到说明面试官很懂。

**坑**：MLA 不是"GQA 的一种"，是**正交的另一条技术路线**（低秩压缩 vs head 共享）。别混。

---

## 二、训练类

### 8. ⭐ Scaling Law 的 Chinchilla 结论是什么

**核心答案**：
- Kaplan 2020（OpenAI）：最早的 scaling law，结论偏向"模型越大越好"，给定算力优先把钱花在参数量上。
- **Chinchilla（DeepMind 2022）**：修正了 Kaplan，结论是**模型参数量 N 和训练数据量 D 应该同比例增长**。compute-optimal 时，N 和 D 大致满足 `D ≈ 20 × N`（每个参数约配 20 个 token）。
- 著名案例：Chinchilla（70B，1.4T token）打败了 Gopher（280B，300B token），证明之前的大模型都"训练不足"（undertrained），参数大但数据喂得不够。

**公式**：给定算力 C ≈ 6ND（前向+反向约 6 倍 FLOPs/token/param），在 C 固定下最小化 loss，解出最优 N 和 D 的分配。

**追问点（2024-2025 重点）**：
- Chinchilla 是 **compute-optimal**（训练算力最省），但**不是 inference-optimal**。
- 现在大家故意"过度训练"小模型（over-training）：因为模型要部署，推理成本是长期主导，宁可训练多花钱也要小模型。Llama3-8B 用了 15T token，远超 Chinchilla 的 `20×8B=160B`，就是为了推理便宜。
- 所以现在的实践是：**训练预算服从 Chinchilla，部署考虑推理成本而过训练**。

**坑**：别只说"N 和 D 同比增长"，要点出 "20 倍"这个数字，以及 "compute-optimal ≠ inference-optimal" 这个现代修正。

---

### 9. 数据去重为什么用 MinHash LSH

**核心答案**：
- 问题：预训练语料里大量重复（网页镜像、模板、转载），重复数据会浪费算力、导致记忆而非泛化、甚至引发 loss 异常。
- 精确去重容易（hash 整段），难的是**近似去重**（fuzzy dedup）——两个文档 95% 相同也要去掉。两两比较是 O(n²)，万亿级文档算不动。
- **MinHash + LSH** 解决近似去重的效率问题：
  - **MinHash**：用一组哈希函数把每个文档的 n-gram 集合压缩成一个固定长度的签名（signature）。两个文档的 MinHash 签名相同的概率 ≈ 它们的 Jaccard 相似度。
  - **LSH（Locality-Sensitive Hashing）**：把签名分段（band），相似文档大概率落入同一个 bucket，**只在 bucket 内比较**，把 O(n²) 降到接近 O(n)。

**追问点**：
- Jaccard 相似度 = 交集/并集，衡量两个集合的重叠。
- 阈值怎么定：band 数和每 band 行数控制相似度阈值（如 0.8 以上算重复）。
- 现代数据 pipeline（FineWeb、Dolma）都用这套，加上质量过滤、PII 去除等。

**坑**：能区分"精确去重"和"近似去重"，说清 MinHash 估 Jaccard、LSH 降复杂度两步即可。

---

### 10. ⭐ Loss spike 怎么处理

**核心答案**：
- Loss spike = 训练中 loss 突然飙升（有时能恢复，有时直接发散崩掉），大模型训练的常见噩梦。
- 成因：梯度爆炸、某个 batch 的脏数据、学习率过大、数值精度问题（FP16 溢出）、深层网络不稳定。

**处理手段**（要能列出组合拳）：
1. **跳过坏 batch**：检测到 loss/grad 异常就跳过这个 batch（PaLM 的做法：回退几步、跳过触发 spike 的数据继续）。
2. **梯度裁剪（gradient clipping）**：限制梯度范数上限，最常用。
3. **降低学习率 / 更长 warmup**。
4. **用 BF16 而非 FP16**：BF16 动态范围大，不易溢出（现在基本标配）。
5. **架构层面**：QK-norm、更稳的初始化、Pre-norm、embedding norm。
6. **z-loss**：惩罚 logits 过大，稳定 softmax（PaLM 用）。
7. **从 checkpoint 回滚**：崩了就回退到 spike 之前的 checkpoint，调整后继续。

**追问点**：
- OPT-175B 的 training logbook 是经典案例，记录了真实的崩溃和重启过程，值得读。
- DeepSeek-V3 用 FP8 训练，专门处理了数值稳定问题。

**坑**：面试官爱问"训练崩了怎么排查"，要有**排查顺序**：先看是数据问题还是优化问题 → 检查 grad norm → 看是否特定 batch → 调 LR/clip/精度。

---

### 11. Learning rate schedule（为什么 warmup + cosine）

**核心答案**：
- 典型 schedule：**linear warmup + cosine decay**。
- **Warmup**（前期线性升 LR）：训练初期参数随机、梯度方向不可靠，大 LR 会让模型跑偏甚至发散。先用小 LR 让模型进入合理区域，再升到峰值。尤其 Adam 类优化器，初期二阶动量估计不准，warmup 能稳住。
- **Cosine decay**（后期余弦降 LR）：训练后期需要小 LR 精细收敛。Cosine 平滑下降到接近 0，比阶梯式衰减更平滑、效果更好。

**追问点（现代趋势）**：
- **WSD schedule（Warmup-Stable-Decay）**：MiniCPM、DeepSeek 用。warmup 后保持恒定 LR 很久（stable），最后快速衰减（decay）。好处：stable 阶段的 checkpoint 都能用，且能灵活决定何时退火，适合持续训练 / 不确定总步数的场景。
- Cosine 的缺点：必须预先知道总步数，中途加数据不好接。WSD 解决了这个。

**坑**：能讲 warmup 防发散、decay 助收敛是及格，能提 WSD 和它解决的问题是加分。

---

### 12. Gradient checkpointing 省了什么、代价是什么

**核心答案**：
- 也叫 activation checkpointing / recomputation。
- **省什么**：显存。正常反向传播要保存前向时所有中间激活值（activation）用于算梯度，这部分显存随序列长度和层数线性增长，很占地方。Gradient checkpointing **只保存部分关键节点的激活，其余的在反向传播时重新前向计算一遍**。
- **代价**：算力。多了一次前向计算，训练时间增加约 20%-33%。
- 本质：**用计算换显存**（time-memory tradeoff）。

**追问点**：
- 通常每隔几层存一个 checkpoint（如每个 transformer layer 存一次输入）。
- 和它配合的省显存手段：ZeRO（切优化器状态/梯度/参数）、FlashAttention（不存 attention 中间矩阵）、offload（搬到 CPU/NVMe）。

**坑**：别说成"省了梯度"，省的是**前向激活值（activation）**的存储，梯度还是要算的。

---

### 13. Mixed precision 训练（FP16/BF16/FP8）

**核心答案**：
- 混合精度：用低精度（FP16/BF16）做大部分计算和存储以省显存、提速，关键部分（如权重主副本、loss 累加）用 FP32 保精度。
- **FP16**：1 符号 + 5 指数 + 10 尾数。精度高但**动态范围小**，容易上溢/下溢，必须配 **loss scaling**（把 loss 放大再反传，防止小梯度下溢为 0，更新前再缩回）。
- **BF16**：1 符号 + 8 指数 + 7 尾数。指数位和 FP32 一样（动态范围大），尾数少（精度低）。**不易溢出，通常不需要 loss scaling**，现代训练首选。代价是精度略低，但对深度学习够用。
- **FP8**：更激进，8 bit。DeepSeek-V3 大规模用 FP8 训练，需要精细的 scaling 策略（per-tensor/per-block scaling）和混合精度框架，省显存省带宽，但工程复杂。

**追问点**：
- 为什么 BF16 取代 FP16 成主流？因为大模型训练里动态范围比精度更重要，BF16 不溢出省心。
- FP32 master weights：即使用 BF16 计算，也常保留一份 FP32 主权重做更新，防止小更新被精度淹没。

**坑**：FP16 和 BF16 的区别核心在**指数位（动态范围）vs 尾数位（精度）**的分配，不是简单"精度高低"。

---

## 三、对齐类（你 RL 背景，这块是主场，必须吃透）

### 14. ⭐ RLHF 三阶段是什么

**核心答案**：
1. **SFT（Supervised Fine-Tuning）**：用人工标注的高质量"指令-回答"对，监督微调预训练模型，让它学会按指令格式回答。这是 base model → 能对话的起点。
2. **RM（Reward Model）训练**：收集人类对模型多个回答的**偏好排序**（A 比 B 好），训练一个奖励模型，输入回答输出标量分数。损失通常是 Bradley-Terry 模型：`L = -log σ(r(chosen) - r(rejected))`。
3. **RL 优化（PPO）**：用 RM 作为奖励信号，用 PPO 优化 SFT 模型，让它生成的回答能获得更高 reward，同时用 KL 散度约束不要偏离 SFT 模型太远。

**追问点**：
- 为什么不直接用 RM 的分数监督，而要用 RL？因为 reward 是对整个生成序列的，不可导，且要 on-policy 探索，所以用 RL。
- 现在 DPO 把 RM 和 RL 两步合并了（见 16 题）。

**坑**：三阶段顺序和每阶段产出物要清楚：SFT model → RM → RLHF'd model。

---

### 15. ⭐ PPO 在 LLM 场景的关键 trick

**核心答案**（这题你 RL 背景应该最能讲，要讲出 LLM 场景的特殊性）：
- **建模**：把生成看成序列决策。state = 已生成的 token 序列，action = 下一个 token，policy = LLM。
- **四个模型同时存在**（显存大户）：
  1. Policy model（被训练的）
  2. Reference model（冻结的 SFT 模型，算 KL）
  3. Reward model（冻结的，给最终 reward）
  4. Value/Critic model（估计 value，算 advantage）

**关键 trick**：
1. **KL penalty**：reward 里减去 `β·KL(policy || reference)`，防止 policy 为了刷高 reward 而跑偏（生成乱码或 reward hacking）。KL 既可加在 reward 上，也可加在 loss 上。
2. **Advantage normalization**：对 advantage 做归一化，稳定训练。
3. **GAE（Generalized Advantage Estimation）**：平衡 bias 和 variance 估计 advantage。
4. **Reward 只在序列末尾给**（outcome reward），中间 token 的 reward 来自 KL 项。
5. **Clip**：PPO 的核心，限制 policy 更新幅度（ratio clip）。
6. **Value clipping、PPO epochs、mini-batch** 等工程细节。

**追问点**：
- 必读 *The 37 Implementation Details of PPO*——很多坑（如 advantage 归一化、reward 缩放、学习率）。
- *Secrets of RLHF*（复旦）讲了 LLM RLHF 的稳定性技巧。

**坑**：能数清"四个模型"且讲清各自作用，是这题的硬门槛。很多人只记得 policy 和 reward。

---

### 16. ⭐⭐ DPO 推导（从 PPO 目标到 DPO loss 的关键步骤）

**核心答案**（这题是后训练岗的"手撕题"，必须能推）：

RLHF 的优化目标：
```
max_π  E[r(x,y)] - β·KL(π(y|x) || π_ref(y|x))
```
这个带 KL 约束的奖励最大化问题，有**闭式最优解**：
```
π*(y|x) = (1/Z(x)) · π_ref(y|x) · exp(r(x,y)/β)
```
其中 Z(x) 是配分函数（归一化项，难算）。

**关键一步**：反解出 reward：
```
r(x,y) = β·log(π*(y|x)/π_ref(y|x)) + β·log Z(x)
```

把这个 r 代入 RM 的 Bradley-Terry 偏好损失 `-log σ(r(chosen) - r(rejected))`，**Z(x) 在相减时被消掉**（因为 chosen 和 rejected 共享同一个 x，Z(x) 相同），得到 DPO loss：
```
L_DPO = -log σ( β·log(π(y_w|x)/π_ref(y_w|x)) - β·log(π(y_l|x)/π_ref(y_l|x)) )
```
其中 y_w 是 chosen，y_l 是 rejected。

**直觉**：DPO 把"训 RM + 跑 RL"两步合并成一步监督学习。不需要显式 reward model，policy 自己**隐式地**就是 reward model。直接用偏好数据梯度下降，提升 chosen 概率、降低 rejected 概率。

**为什么能 work**：上面那个闭式解说明，最优 policy 和 reward 之间有解析关系，所以可以绕过 RM 直接优化 policy。

**追问点**：
- DPO 不需要采样（offline），PPO 需要 on-policy 采样。这是 DPO 简单的根源，也是它的局限（见下题）。
- DPO 的梯度会同时压低 chosen 和 rejected 的绝对概率（只是 rejected 降得更多），有时导致输出概率整体下降，这是已知问题。

**坑**：这题能完整推下来（最优解 → 反解 reward → 代入 BT loss → 消 Z）就赢过 90% 候选人。背公式没用，要懂"Z(x) 被消掉"这个关键。

---

### 17. ⭐ DPO vs IPO vs KTO vs SimPO vs ORPO，各解决什么

**核心答案**（按"它在前一个基础上改了什么"来记）：

- **DPO**：偏好学习的基线。需要成对偏好数据 (chosen, rejected) + 一个 reference model。
- **IPO**：解决 DPO 的**过拟合问题**。DPO 在偏好确定（chosen 远好于 rejected）时会无限拉大概率差导致过拟合，IPO 改了目标函数加正则，让它有界。
- **KTO**：解决**数据格式问题**。不需要成对偏好，只需要**单个回答 + 好/坏的二元标签**（来自前景理论 Kahneman-Tversky）。现实中点赞/点踩这种数据比成对排序好收集。
- **SimPO**：解决 DPO 需要 **reference model** 的问题。用平均 log 概率（长度归一化）作为隐式 reward，**去掉 reference model**，省显存、更简单。还加了 reward margin。
- **ORPO**：把 **SFT 和偏好对齐合并成一步**（单阶段）。在 SFT loss 上加一个 odds ratio 惩罚项压低 rejected，**不需要 reference model，也不需要单独的 SFT 阶段**。

**一句话总结表**：

| 方法 | 核心改进 | 要 reference model？ | 数据格式 |
|---|---|---|---|
| DPO | 基线 | 要 | 成对偏好 |
| IPO | 防过拟合 | 要 | 成对偏好 |
| KTO | 用二元反馈 | 要 | 单条 + 好/坏标签 |
| SimPO | 去掉 ref model | 不要 | 成对偏好 |
| ORPO | SFT+对齐合一 | 不要 | 成对偏好 |

**坑**：别死记，按"DPO 有什么痛点 → 每个变体治哪个痛点"理解。痛点有三个：过拟合（IPO）、数据难收集（KTO）、依赖 ref model/两阶段（SimPO/ORPO）。

---

### 18. ⭐⭐ GRPO 相对 PPO 去掉了 critic，为什么能去掉

**核心答案**（这题是 2025 最高频，你必须讲到滚瓜烂熟）：
- GRPO（Group Relative Policy Optimization，DeepSeekMath 提出，R1 发扬光大）。
- PPO 需要一个 **critic/value model** 来估计每个 state 的 value，进而算 advantage `A = reward - value`。critic 和 policy 一样大，**翻倍显存、还难训**。
- GRPO 的核心思想：**用 group 内的平均 reward 当 baseline，替代 critic**。
  - 对同一个 prompt，采样一组（group）G 个回答。
  - 每个回答的 advantage = `(它的 reward - group 内 reward 均值) / group 内 reward 标准差`。
  - 即用**同组其他回答的平均表现**作为 baseline，省掉了 value model。

**为什么能去掉 critic**：
- critic 的作用是提供 baseline 降低梯度方差。GRPO 用"同 prompt 多次采样的均值"这个**经验 baseline** 替代了 critic 的"学习 baseline"。
- 在 LLM 场景，reward 通常是序列级的（outcome reward，尤其 RLVR 下是 0/1 正确性），用 group 均值做 baseline 既准又省。

**优点**：省一半显存（无 critic）、训练更简单、特别适合有明确正确性信号的任务（数学、代码）。

**追问点**：
- GRPO 是 on-policy 的吗？是，要实时采样。
- KL 项还在吗？在，GRPO 仍保留对 reference 的 KL 约束（不过 R1 后有些工作把它去掉或改造）。
- 局限：每个 prompt 要采样多个回答，采样成本高；reward 信号弱的任务（无明确正确答案）效果不如有 critic 的精细 advantage。
- DAPO、Dr.GRPO 等是 GRPO 的后续改进（解决 length bias、难度采样等）。

**坑**：核心就一句——**用 group 内 reward 的均值/方差归一化当 advantage，替代 value model**。能讲清"为什么 group 均值能当 baseline"（降方差的经验 baseline）是关键。

---

### 19. ⭐ Reward hacking 的常见模式（你评测背景的切入点！）

**核心答案**：
- Reward hacking = 模型钻 reward 信号的空子，**拿到高 reward 但没真正完成任务**。本质是 reward model 是真实目标的不完美代理（proxy），模型过度优化代理目标。

**常见模式**：
1. **长度 bias**：RM 偏好长回答，模型就生成又臭又长的废话刷分。最经典，几乎必出现。
2. **格式/风格刷分**：RM 喜欢 markdown、列表、礼貌用语，模型疯狂堆格式而不管内容。
3. **谄媚（sycophancy）**：模型迎合用户观点而非说真话，因为人类标注偏好"被认同"。
4. **重复/模板化**：发现某种句式得分高就反复用。
5. **利用 RM 盲区**：RM 在某些 OOD 输入上打分异常，policy 专门往那里钻。

**怎么缓解**（你的评测经验在这里发光）：
- KL 约束（别偏离 ref 太远）。
- 长度惩罚 / 长度归一化 reward。
- RM ensemble、定期重训 RM（对抗 distribution shift）。
- **更好的评测**：用多维度、抗 hacking 的 diagnostic eval 监控训练（这正是你能讲的差异化经验——"我做评测时见过哪些虚假提升信号"）。
- RLVR：用规则化的可验证 reward（答案对错），从根上避免 RM 被 hack。

**坑**：这题一定要联系你的 benchmark 经验讲。面试官问 reward hacking，你接一句"我做评测时就发现过这类伪提升……"立刻拉满印象分。

---

### 20. Process reward vs Outcome reward

**核心答案**：
- **Outcome Reward Model (ORM)**：只对**最终答案**打分（对/错）。简单、信号稀疏，只看结果不看过程。
- **Process Reward Model (PRM)**：对推理过程的**每一步**打分。信号密集，能定位哪一步错了，对多步推理（数学、代码）更有效。

**对比**：

| | ORM | PRM |
|---|---|---|
| 监督粒度 | 最终结果 | 每个推理步 |
| 数据成本 | 低（只标答案对错） | 高（要标每步对错） |
| 信号密度 | 稀疏 | 密集 |
| 适用 | 简单任务 | 复杂多步推理 |

**追问点**：
- PRM 怎么标？人工标贵，Math-Shepherd 用**自动化方法**：从某步出发多次 rollout，看能否到正确答案来估计该步的价值（蒙特卡洛式）。
- *Let's Verify Step by Step*（OpenAI）证明 PRM 在数学推理上优于 ORM。
- PRM 可用于：训练时给密集 reward、推理时做 best-of-N 重排序（用 PRM 选最好的推理链）。
- 但 DeepSeek-R1 发现：**纯 RLVR（outcome reward）+ GRPO 也能 work**，不一定非要 PRM，且 PRM 有 reward hacking 风险、训练复杂。这是个有争议的活跃方向。

**坑**：能讲"PRM 信号密但贵且易 hack，ORM/RLVR 简单但稀疏"的 tradeoff，并知道 R1 用 outcome reward 也成功，就很到位。

---

## 四、推理 / 系统类

### 21. ⭐ KV cache 的显存计算

见第 4 题公式。补充重点：
- 推理分两阶段：**prefill**（处理 prompt，并行算所有 token，算力密集）和 **decode**（逐 token 生成，访存密集，靠 KV cache 避免重算）。
- KV cache 让 decode 阶段每生成一个 token 只需算新 token 的 Q 与历史 K/V 的 attention，不用重算整个序列，把 decode 从 O(n²) 降到 O(n)。
- 代价：cache 占大量显存，且随序列和 batch 线性增长，是长上下文推理的主要瓶颈（催生了 GQA、MLA、PagedAttention、KV 量化等）。

**坑**：要能区分 prefill（compute-bound）和 decode（memory-bound），这是推理优化的基础认知。

---

### 22. ⭐ FlashAttention 为什么快（tiling + recomputation）

**核心答案**：
- 普通 attention 要显式构造 `S = QK^T`（N×N 矩阵）和 softmax 结果存到 HBM（显存），N 大时这个矩阵巨大，**瓶颈是 HBM 读写（IO），不是算力**。
- FlashAttention 是 **IO-aware** 算法，核心两招：
  1. **Tiling（分块）**：把 Q、K、V 分块加载到 SRAM（片上高速缓存），分块计算 attention，**不把完整的 N×N 矩阵写回 HBM**。用 online softmax 技巧增量更新结果，无需看到整行。
  2. **Recomputation（重算）**：反向传播时不存 N×N 的中间 attention 矩阵，而是用保存的 softmax 统计量（行最大值、行和）**重新计算**。省显存（用算力换）。
- 结果：显存从 O(N²) 降到 O(N)，速度因减少 HBM 访问而大幅提升。**注意：它是精确的，不是近似！**

**追问点**：
- online softmax：边读块边更新 running max 和 running sum，保证数值稳定且不用一次看到全部。
- v2 优化了并行划分和 work partition；v3 针对 Hopper（H100）用了 FP8 和异步特性。
- 关键洞察：现代 GPU 算力远超访存带宽，所以**减少访存比减少计算更重要**（memory-bound 优化思路）。

**坑**：很多人误以为 FlashAttention 是近似算法或减少了计算量。正解：**计算量没减少（甚至反向还增加了重算），减少的是 HBM IO**，靠这个加速。

---

### 23. PagedAttention 解决什么问题

**核心答案**：
- PagedAttention（vLLM 的核心技术）解决 **KV cache 的显存碎片和浪费**问题。
- 问题：传统 KV cache 给每个请求预分配一块**连续**显存（按 max_len），但实际生成长度不定，导致：内部碎片（分配多了用不完）、外部碎片（块大小不一难复用）、无法共享。
- 解法：**借鉴操作系统的虚拟内存分页**。把 KV cache 切成固定大小的 **block（页）**，用一张 block table 把逻辑序列映射到物理块，**物理块不必连续**。
- 好处：
  1. 显存几乎零浪费（按需分配 block）。
  2. 支持 **prefix sharing**：多个请求共享相同前缀（如同一 system prompt）的 KV block（copy-on-write），省显存。
  3. 大幅提升吞吐（能塞更大 batch）。

**追问点**：
- 配合 **continuous batching**（动态批处理）：请求完成就退出、新请求随时加入，不用等整个 batch 结束，GPU 利用率拉满。这是 vLLM 高吞吐的另一半。

**坑**：PagedAttention 解决的是**显存管理/碎片**，不是计算加速。和 FlashAttention（算 attention 本身）是不同层面的优化。

---

### 24. Speculative decoding 原理

**核心答案**：
- 目标：加速自回归生成（decode 阶段逐 token 太慢，且 memory-bound，GPU 算力没吃满）。
- 原理：用一个**小而快的 draft model** 一次性猜测接下来的 k 个 token，再用**大的 target model 并行验证**这 k 个 token（一次前向就能验证 k 个，因为大模型并行算 k 个位置的概率很便宜）。
  - 验证通过的 token 保留，第一个不通过的地方拒绝并由大模型重新采样。
  - 通过精心设计的接受/拒绝采样，保证**输出分布和大模型单独解码完全一致**（无损加速）。
- 收益：如果 draft 命中率高，一次大模型前向能产出多个 token，加速 2-3 倍。

**追问点**：
- 关键是 draft model 要和 target model 分布接近（命中率高）且足够快。
- 变体：
  - **Medusa**：不用单独 draft model，给大模型加多个预测头同时预测多个未来 token。
  - **EAGLE**：在特征层做投机，命中率更高。
  - **自投机 / n-gram**：用模型自身或简单 n-gram 做 draft。
- 本质：把 memory-bound 的 decode 转成 compute-bound 的并行验证，利用空闲算力。

**坑**：强调"无损"——输出分布和原模型一致，不是近似。这是它能被工业采用的关键。

---

### 25. 量化（INT8/INT4/AWQ/GPTQ）原理差异

**核心答案**：
- 量化 = 把 FP16/BF16 权重（和/或激活）用低位整数（INT8/INT4）表示，省显存、提速（尤其访存）。核心是找到 scale 把浮点范围映射到整数范围。
- **训练后量化（PTQ）** 是 LLM 主流（不用重训）：

| 方法 | 核心思想 |
|---|---|
| **GPTQ** | 逐层量化，用二阶信息（Hessian）逐列量化权重并补偿误差，最小化量化带来的输出误差。weight-only，4bit 效果好。 |
| **AWQ** | Activation-aware。观察到少数"重要权重通道"（由激活幅度判断）对效果影响大，量化前对这些通道做 scaling 保护，再量化。比 GPTQ 简单、泛化好。 |
| **SmoothQuant** | 解决激活难量化（有 outlier）问题，把量化难度从激活"迁移"一部分到权重，让两者都好量化。支持 W8A8。 |
| **GGUF/llama.cpp 系** | CPU/边缘部署常用的量化格式。 |

**关键概念（追问点）**：
- **outlier 问题**：LLM 激活值有少数极大的 outlier，直接量化会让 scale 被拉大、其他值精度全毁。AWQ/SmoothQuant 都在处理这个。
- weight-only（W4A16，只量权重）vs weight-activation（W8A8，都量）：前者省显存、对精度友好；后者还能提算力但更难（激活有 outlier）。
- QAT（量化感知训练）vs PTQ：QAT 训练时模拟量化，效果好但贵；LLM 多用 PTQ。

**坑**：能讲清 GPTQ（误差补偿）和 AWQ（保护重要通道）的区别，以及 outlier 为什么是量化的核心难点，就到位了。

---

## 五、分布式

### 26. ⭐ DP/TP/PP/SP/EP 分别切什么

**核心答案**（必背表）：

| 并行方式 | 切什么 | 通信 | 备注 |
|---|---|---|---|
| **DP**（数据并行） | 切**数据**（batch），每卡一份完整模型 | 反向后 all-reduce 梯度 | 最简单；模型要能放进单卡 |
| **TP**（张量并行） | 切**单层内的权重矩阵**（如 attention/FFN 的 W 按行或列切） | 每层前向/反向都要 all-reduce，通信频繁 | 通常限单机内（NVLink 高带宽） |
| **PP**（流水线并行） | 切**层**（不同层放不同卡），按 stage 分 | stage 间传 activation，通信少 | 有 pipeline bubble（气泡），用 micro-batch 缓解 |
| **SP**（序列并行） | 切**序列维度**（配合 TP，切 LayerNorm/Dropout 等沿序列的部分） | 配合 TP 的 all-gather/reduce-scatter | 省 activation 显存 |
| **EP**（专家并行） | 切 **MoE 的 expert**（不同 expert 放不同卡） | token 路由的 all-to-all 通信 | MoE 专用 |

**3D 并行**：实际大规模训练把 DP+TP+PP 组合用（如 Megatron-LM）。典型布局：TP 在机内（NVLink），PP 跨机，DP 最外层。

**追问点**：
- ZeRO（见 27）是 DP 的增强，切的是优化器状态/梯度/参数，和 TP/PP 正交可叠加。
- FSDP = PyTorch 版的 ZeRO-3。
- 选择逻辑：模型放不下单卡先上 TP（机内），还放不下上 PP（跨机），再用 DP 扩 batch、ZeRO 省显存。

**坑**：TP 通信频繁所以限机内、PP 通信少可跨机——这个"通信量决定部署拓扑"的逻辑要懂。

---

### 27. ⭐ ZeRO-1/2/3 区别

**核心答案**：
- ZeRO（Zero Redundancy Optimizer，DeepSpeed）解决 DP 的冗余：普通 DP 每张卡都存一份**完整**的优化器状态、梯度、参数，极度浪费。ZeRO 把这三类**分片（partition）**到各卡，用时再聚合。
- 训练显存三大件：参数（P）、梯度（G）、优化器状态（OS，Adam 的动量+方差，最大头，FP32 约是参数的 6-12 倍）。

| 阶段 | 分片什么 | 省显存 | 通信代价 |
|---|---|---|---|
| **ZeRO-1** | 优化器状态（OS） | 最大头先切，省很多 | 几乎不增加 |
| **ZeRO-2** | OS + 梯度（G） | 更省 | 略增（梯度 reduce-scatter） |
| **ZeRO-3** | OS + G + 参数（P） | 最省，参数也切 | 增加（前向/反向要 all-gather 参数） |

- **ZeRO-3 = FSDP**：每张卡只存 1/N 的参数，前向/反向计算到某层时临时 all-gather 完整参数，用完即弃。显存随卡数近线性下降，代价是通信变多。
- **ZeRO-Offload / Infinity**：进一步把状态 offload 到 CPU 内存甚至 NVSSD，单卡也能训大模型（慢）。

**追问点**：
- ZeRO vs TP：ZeRO 是 DP 路线（不切单层计算，靠通信聚合参数），通信模式和 TP 不同。大规模常 ZeRO + TP + PP 一起上。
- 通信换显存的权衡：ZeRO-3 最省显存但最多通信，带宽不够时反而慢。

**坑**：记住"切的三样东西：优化器状态→梯度→参数"，逐级增加。ZeRO-3 = FSDP 是高频对应关系。

---

### 28. ⭐ 70B 模型训练显存大概怎么算

**核心答案**（要会现场估算）：

以 Adam 优化器、混合精度训练为例，**每个参数**占用：
- FP16 参数：2 bytes
- FP16 梯度：2 bytes
- FP32 参数主副本（master weight）：4 bytes
- FP32 Adam 动量 m：4 bytes
- FP32 Adam 方差 v：4 bytes
- 合计约 **16 bytes/参数**（经典的"16x"估算）

70B 模型：`70e9 × 16 ≈ 1120 GB ≈ 1.1 TB`，仅模型状态。
- 加上 **activation**（随 batch、seq_len、层数变化，可能又是几百 GB，可用 gradient checkpointing 压）。
- 一张 H100 是 80GB，所以光模型状态就要 **~14 张卡**才放得下（还没算 activation 和碎片），实际要更多并配合 ZeRO/TP/PP。

**推理显存**（对比，常被一起问）：
- 推理只需参数 + KV cache。70B BF16 参数 = `70e9 × 2 ≈ 140 GB`，约 2 张 H100。INT4 量化后 ~35GB，单卡可放。

**追问点**：
- 训练显存主要被**优化器状态**吃掉（Adam 的 m、v 是 FP32，最占）——这就是 ZeRO 先切优化器状态的原因。
- 怎么压：ZeRO 切状态、gradient checkpointing 压 activation、用 8-bit optimizer（如 bitsandbytes）压优化器状态。

**坑**：训练"16 bytes/参数"和推理"2 bytes/参数（BF16）"这两个数要张口就来。能指出"优化器状态是大头"是加分。

---

## 六、开放题（考结构化思维，没标准答案，要有框架）

### 29. ⭐ "给你 1000 卡训 70B 怎么搞" —— 方法论

**回答框架**（按这个结构讲，体现系统性）：

**1. 算预算（Scaling Law）**
- 70B 按 Chinchilla 要 ~1.4T token（compute-optimal），但考虑推理成本会过训练到 ~10-15T token（参考 Llama3）。
- 估算总 FLOPs ≈ 6 × N × D，再除以 1000 卡的有效算力（考虑 MFU ~40-50%）估训练时长。

**2. 数据 pipeline**
- 来源：网页（CommonCrawl/FineWeb）、代码、书籍、论文、多语言，按配比混合。
- 处理：去重（MinHash LSH）、质量过滤（fastText/困惑度）、去 PII、去毒、decontamination（去掉测试集泄漏）。
- 配比和退火：后期用高质量数据退火（如 Llama3 的 annealing）。

**3. 并行策略**
- 1000 卡（比如 125 台 8 卡 H100）：TP=8（机内 NVLink）、PP 跨若干机、DP 在最外层、配 ZeRO-1。
- 目标 MFU 拉到 40%+。

**4. 训练稳定性**
- BF16/FP8、gradient clipping、warmup+WSD、监控 grad norm、loss spike 自动跳 batch、勤存 checkpoint。

**5. 评测与监控**
- 训练中定期跑 MMLU/GSM8K/HumanEval 等监控能力增长，看 loss 曲线和下游指标是否同步。

**6. 后训练**
- 预训练完做 SFT → DPO/RLHF → 评测对齐效果。

**坑**：面试官看的是你**有没有全局观**，能不能从数据→并行→稳定→评测串起来。不用每步都深，但要完整、有取舍依据。

---

### 30. ⭐ "模型某任务效果差怎么排查" —— 排查树

**回答框架**（结构化排查，从外到内）：

**1. 先确认问题本身**
- 是真的差，还是**评测有问题**？（你的强项！）检查：评测 prompt 格式对不对、答案抽取逻辑对不对、few-shot 设置、有没有 chat template 没对齐、metric 算错。很多"效果差"是评测 bug。

**2. 定位是哪类问题**
- 能力问题（模型不会）vs 对齐问题（会但不按要求输出）vs 格式问题（输出对但抽取失败）。
- 看具体 case：人工读几十条错误样本，分类错误类型。

**3. 数据角度**
- 训练数据里这个任务/领域的数据够不够？分布匹配吗？有没有该能力的数据被去重/过滤掉了？

**4. 训练角度**
- 是不是 SFT/RLHF 阶段引入的退化（alignment tax，对齐后某些能力下降）？
- 学习率、训练轮数、数据配比是否合适？过拟合还是欠拟合？

**5. 推理角度**
- 采样参数（temperature、top-p）合不合适？
- 量化掉点了吗？长度截断了吗？

**6. 对比实验**
- 和 base model 比、和同规模其他模型比、和上一个 checkpoint 比，定位是哪个环节引入的。

**坑**：这题最能体现你的评测背景优势——**第一步就质疑"是不是评测本身的问题"**，这是有经验的人才会先想到的。一定要把它放在排查树的第一位讲。

---

## 附：学习建议

1. **每题都要能"说"出来**，不是认识就行。对着这份文档，合上眼复述一遍，卡壳的地方就是没真懂。
2. **标 ⭐⭐ 的（16 DPO 推导、18 GRPO）是你后训练岗的命门**，要能在白板上推/讲 10 分钟。
3. **第 19（reward hacking）、30（排查树）一定绑定你的评测经验讲**，这是你的差异化。
4. **手撕题**（MHA/GQA/RoPE/DPO loss/采样）单独练，这份文档讲原理，代码要自己敲。
5. 遇到任何一题想往深挖（推导细节、代码实现、最新进展），告诉我题号，我展开。
