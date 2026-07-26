# Runtime 架构、硬件利用与优化审计

本文描述仓库当前代码的真实行为。接口或调度元数据已经存在、但尚未进入
执行热路径的能力，会明确标记为“基础设施”，不会按已完成的性能优化统计。

## 结论

当前 Runtime 已经不是仅能验证 MoE 数值正确性的最小框架：GPT-OSS 的
CPU-only 和 Vulkan Dense/Attention + CPU Expert 混合推理可以完整运行，
  MXFP4 Expert、按需权重缓存、Vulkan KV ring、QKV+RoPE/在线 Decode-SDPA
  fusion、跨 Session 分阶段合批和生成接口均已落地。

但它还不能最大化利用任意设备上的所有硬件资源：

- 单个 Session 的图仍由 `CpuExecutor` 按自回归依赖波次推进。Vulkan
  Attention 完成并下载输出后才开始 CPU Router/Expert；同一 token 的真实
  数据依赖不能被伪造为 overlap。
- pinned ncnn 的 `VkCompute` 已拆分为 `submit_async()`/`wait()`。两个
  transfer slot 可并行准备 host staging，并持久 reset/reuse command
  object/buffer；共享 ncnn layer/allocator 的 GPU 命令会持有设备锁直到
  fence 完成，以保证跨 Session 确定性。另一 Session 的 CPU Expert 仍可
  与当前 GPU 阶段重叠；单 Session 跨层流水仍未完成。
- Vulkan 负责 Dense Transformer 和 Attention。Expert 默认在 CPU；
  可选 executable Expert cache 使用原生 Vulkan MXFP4 Gate/Up + 激活 +
  Down kernel，并与 CPU Expert 并行。在线端到端校准若发现异构阶段更慢，
  会停止该 token-count 桶的新 admission，并把 resident 命中退回 CPU。
- Runtime 会枚举 Vulkan 设备及其类型、性能分、队列、heap 和精度能力。
  `Auto` 不会把软件 CPU Vulkan 当作硬件加速器；显式
  `RuntimeOptions::vulkan_device_indices` 可提供多个候选。编译器按设备
  score、层数和 `expected_concurrency` 最小化 latency + pipeline
  bottleneck，并给每层记录实际 device；Expert cache 与层 device 共置。
- x86 支持运行时选择 Scalar、AVX2/FMA、AVX-512，ARM 支持 NEON 与
  vector-length-agnostic SVE2。VNNI/AMX capability 可被准确探测，但
  MXFP4×FP32 不能在不量化 activation 的情况下等价映射到整数 dot；
  因此当前精确模式不会错误地分派到 VNNI/AMX。
- 当前只注册 GPT-OSS Adapter，并已用真实 20B/120B checkpoint 验证。
  `MoeIR` 和编译边界保持模型无关，但不会为尚未具备完整加载、执行、
  checkpoint acceptance 和基准协议的模型保留生产实现或占位分支。

因此，当前版本应定义为“具有真实异构执行和大模型存储能力的 Runtime
主干”，而不是“已经完成所有硬件的峰值优化”。

## 当前代码结构

```text
Application
    |
Runtime / Model / Session / BatchScheduler
    |
Model Adapter -> MoeIR -> ModelCompiler
                            |
                 ExecutionGraph + RuntimeScheduler
                            |
            +---------------+---------------+
            |                               |
     ncnn CPU/Vulkan                  CPU Expert Engine
 Dense + Attention + KV       Dispatch + Cache + MXFP4
            |                               |
            +---------------+---------------+
                            |
              MemoryManager + ExpertStore
```

责任边界：

| 目录 | 当前职责 |
| --- | --- |
| `include/ncnn/moe` | 公开 Runtime、MoeIR、执行图、Expert、内存与调度 API |
| `src/compiler` | MoeIR 验证/规范化、模型无关图构建、Compiled execution graph |
| `src/graph` | ModelCompiler、拓扑调度、路由分组、模型内存规划 |
| `src/engine` | Session、同步执行器、跨 Session scheduler、Tensor residency |
| `src/storage` | mmap、Expert ARC、异步读取、Vulkan victim cache、ExpertStore |
| `src/kernels` | Portable CPU Attention/Linear 和 MXFP4 SIMD kernel |
| `src/backends/ncnn` | ncnn CPU/Vulkan operator、完整 Vulkan Attention block |
| `src/models` | GPT-OSS Adapter、Safetensors 映射和 canonical tensor name |

## 代码规范化与模块边界

- 项目 C++ 代码统一由根目录 `.clang-format` 管理：四空格、左对齐指针、
  namespace 不额外缩进、函数/类型/控制流使用 ncnn 的 Allman brace 风格。
  与上游 ncnn 一样使用 `ColumnLimit: 0`，不做机械的固定列宽折行。语言标准
  使用 C++11 解析模式，避免 C++03 formatter 破坏项目所需的 raw string。
- 生产 Adapter registry 只保留 GPT-OSS。模型中立性由 `MoeIR`、Compiler 和
  Backend 契约承担，而不是通过未验收模型的解析分支体现。
- 对外 API 只保留调用方实际需要的行为。`Expert` 的构造、统计更新和存储
  细节由 `ModelCompiler`/`ExpertStore` 私有拥有；`WeightTable`、`MemoryManager`
  等删除了未使用的查询入口。项目类按“private helper/data 在上、public API
  在下”排列，使可见边界集中且避免热路径通过多层薄包装跳转。
- 算法热路径不使用临时 lambda 作为局部抽象；ARC、placement、dispatch、
  sampling、MXFP4 autotune 等改为具名 static/private helper 或直接循环。
  lambda 仅保留在线程任务、条件变量 predicate、异步完成、一次初始化和
  资源 deleter 等需要闭包语义的位置。
- 注释仅说明 API 契约、正确性不变量、平台约束或非显然的性能原因。重复
  字段名、逐行叙述实现、历史调试过程和模板化说明已从生产代码删除。
- 未使用的 Expert 权重包装、查找/snapshot API、占位 kernel 枚举和重复
  模型分支已移除。测试仍可使用仅链接到测试目标的 `test_moe` fixture；
  它不进入生产 registry，也不构成第二个受支持模型。

## 已完成的性能优化

### 执行图与调度

- Adapter 输出会被规范化为真实 `MoeIR` 图。IR 节点和 Value 可以描述
  Attention、Router、ExpertGroup、Combine、KV Cache 和 QuantConfig。
- `ExecutionGraph` 记录节点依赖、输入输出 Tensor、后端候选/选择、
  Tensor location 和跨后端 event。`RuntimeScheduler` 生成最大拓扑波次，
  并把 CPU/Vulkan 节点分 lane；schedule 保留检测到的 OpenMP 并行度和
  默认 Vulkan device compute queue 数。
- 同一 Expert 的 token 会先 regroup 成连续 `ExpertBatch`，每个投影按
  Expert batch 执行，不走逐 token Expert 调用。
- Cache acquire/token regroup 先按活跃 Expert 并行。全部为 resident
  MXFP4 decode 时，Gate/Up 与 Down 的 row-pair 任务会跨 Expert 扁平化，
  线程数由实际矩阵运算量、OpenMP 上限和检测到的物理核共同决定；避免
  小 shape 过度并行，也避免带宽型 kernel 被 SMT 线程争用。
- `BatchScheduler` 支持多个 Session 并发 Decode，同时保证同一 Session
  串行；兼容 Session 可按层同步推进，Embedding/Router/LM Head 形成多行
  batch，并把相同 Expert ID 的 route 合并成一次 MXFP4 GEMM 后再散回各
  Session。自动策略按 context-length 与 batch-size 分桶，在线比较 staged
  和 independent 的端到端 ms/request。每个 workload 桶再按上一轮 Expert
  cache-wait 占 Expert 时间的比例拆为 resident（≤5%）、mixed（5–50%）和
  storage（≥50%），避免存储抖动污染热驻留样本。策略使用 5% 基础滞回和
  指数加权绝对偏差作为噪声保护，并按配置间隔低频探索非首选路径。
  worker 足够覆盖两个请求时，初始选择 independent；
  larger batch 仍以 staged 为先验。两种路径都保留强制/禁用开关。Linux
  仍可按 CPU topology/NUMA 划分并绑定 worker。
- 相互独立的单请求 `submit_decode()` 调用会先进入跨调用收集器。默认
  最多等待 200 微秒且最多收集 `worker_count` 个不同 Session；同一 Session
  不会在一个 batch 中出现两次。连续 4 次无同伴后进入 bypass，并按
  `adaptive_probe_interval` 低频复探。等待、探测、超时、bypass、最大队列
  和实际合批大小都有独立统计；`SchedulerOptionDisableCrossCallBatching`
  或零等待可关闭。

注意：跨后端 event 是可验证的调度契约；ncnn fence 已具备异步生命周期，
多 Session 可并发提交。单 Session 的 CPU/GPU 节点仍按依赖消费，尚未
形成跨 token 的 speculative pipeline；跨调用 micro-batch 只利用独立
Session，不违反自回归依赖。

### Runtime 对象布局、分配与同步开销

- 配置类开关按域收进 `uint32_t flags`。`RuntimeOptions` 的 mmap、direct
  I/O、buffered I/O 和禁用 device-resident victim 直执行共用一个 flags；
  `BatchScheduler` 的 pin/staged/cross-call 与自动 topology 状态也共用一个
  内部 flags。状态机结果和受不同 mutex 保护的停止状态仍使用独立字段，
  不会为了节省数个字节而引入跨锁 read-modify-write 数据竞争。
- 每个 bitmap 位号由所属模块的 `NCNN_MOE_<DOMAIN>_<FEATURE>_BIT` 宏定义，
  typed enum 再从该位号生成 mask；调用侧只使用 enum 名。配置、IR、Graph、
  Cache、Scheduler、Runtime capability、CPUID 和 XSTATE 不再直接写
  `1 << N`，数值编码算法不套用这条规则。
- `CpuSessionState` 持久保存 Attention、Expert 和 staged execution scratch。
  Router norm/logits、Combine、Embedding、FinalNorm、LM Head、Attention
  norm/Q/K/V/fused-QKV/SDPA/projection/output 以及 Expert backend request/
  index/result 容器在 warm-up 后复用 capacity。完全覆盖写入的 tensor 不做
  预先 zero-fill；SDPA 和 Combine 等累加输出仍显式清零。
- Session 以双缓冲方式复用事务性 `SessionStatistics`；成功时 swap，失败时
  保留原统计，因此 `expert_token_counts` 不再每 token 重新分配。greedy
  sampling 直接线性 argmax，不构造单元素候选 vector；生成结果预留最大
  token 容量。
- 单 token Expert dispatch 的常见 `top_k <= 16` 路径使用栈上定长候选数组，
  直接复用每层 `ExpertDispatchPlan` 和 `ActiveExpertExecution` 的 routes
  capacity；不会再为每层构造一个 `expert_count` 大小的临时 batch 数组。
  较大 top-k 或多 token Prefill 继续使用通用动态路径，避免固定栈对象失控。
- 大 tensor 不放栈。栈只承载最多 16 个路由候选和最多 16 个 ready-cache
  索引；超过阈值自动回退复用/动态 vector。这样既减少小对象 allocator
  开销，也不把模型 shape 转化为线程栈溢出风险。
- 稳定 Expert key 的 cache/backend 查找使用透明 `string_view` 哈希，
  热路径不再构造临时 `std::string`。ARC 本体使用 hash index + T1/T2/B1/B2
  list，平均查找/移动为 O(1)；红黑树的 O(log N) 和 Trie 的节点/指针开销
  在固定编译 key 场景没有优势。
- Expert pair request/prefetch 在一次 cache mutex 临界区内返回“入队时已
  ready”状态，Executor 不再先 `is_ready()` 再重复 request。staged Session
  对相同预测 Expert 使用三态 scratch，避免重复 cache 查询锁。
- 多个 pending Expert read 不再被最早的慢请求队首阻塞：
  `wait_acquire_ready_pairs()` 等待任意完成项，并在同一临界区取得当前全部
  ready lease。`NCNN_MOE_EXPERT_READY_FIRST=0` 仅保留为同二进制诊断开关。
- Vulkan victim operation 的惰性创建可能分配 device 资源，因此在
  executable backend scheduler mutex 之外执行；重新加锁后复核普通
  admission 是否已经完成。单 GPU submission 记录 `waited_`，避免调用方
  已 wait 后析构又复制一次结果 vector。
- staged batch 在发布 future 结果前释放 Session queue ownership。调用方
  看到 future ready 时可以立即进入下一轮合批，不会因短暂的“结果已返回、
  Session 尚 busy”窗口退化成独立任务。
- `Session` 使用普通 mutex；`generate()` 在一次临界区内调用
  `prefill_unlocked()`/`decode_unlocked()`/`sample_unlocked()`，不再每个
  token 反复获取 recursive mutex。token callback 在锁外执行，重新进入后
  复核 sequence length，既允许读取统计，也拒绝 callback 对同一 Session
  的并发状态修改。
- `MemoryManager::record_execution()` 对整张图只取一次锁并调用
  `transition_unlocked()`，不再每个 tensor transition 重复锁/解锁。
- `ExpertStore` 在编译阶段单线程建立，模型发布后保持 immutable；因此使用
  连续 vector 与 Expert 原子统计，不再同时维护 hash map、snapshot copy 和
  store mutex。读侧统计遍历数量有限的 Expert，换取更紧凑的模型常驻对象。
- shard/file handle cache 使用 shared mutex。命中只取读锁；`CreateFile`、
  file mapping 等 syscall 在锁外执行，写锁阶段只做 double-check/insert，
  并关闭竞争产生的重复 handle。Windows direct-I/O completion event 为
  thread-local，每个 I/O worker 创建一次，不在 Expert read 循环中创建销毁。
- Expert cache entry 的 speculative、job-started、first-exact-miss 状态收进
  一个 `uint32_t flags`。对齐 direct-I/O payload 只在 cache admission 时
  `VirtualAlloc` 一次并随 eviction 释放；循环内 scratch 使用小型栈数组或
  复用 vector。不会用独立于 ARC capacity 的内存池长期保留大 payload。
- 当前没有把 ARC 或 scheduler 全面改成 lock-free。ARC eviction 同时修改
  resident bytes、T1/T2、B1/B2、pin/lease 和两级容量，单一无锁队列不能
  保证这些复合不变量；还会引入 ABA、忙等、内存回收和较差尾延迟。当前
  策略是先消除重复加锁、缩小外部操作的锁范围，并用统计/基准确认争用；
  只有独立生产消费队列被证明是热点后才引入有界 ring/CAS 实现。

### CPU Expert 与 MXFP4

- MXFP4 权重直接融合 decode 与累加，不先完整展开为 FP32 矩阵。
- 运行时 ISA dispatch：
  - x86：Scalar、AVX2/FMA/SSSE3、AVX-512；
  - ARM：NEON、可选 SVE2；
  - 不满足 SIMD 条件时自动退化到 Scalar，不要求 AVX。
- MSVC 将 AVX2 和 AVX-512 放在独立 translation unit 中，baseline
  translation unit 使用 CPUID/XGETBV 选择，旧 x86-64 CPU 不会执行非法
  指令。
- 两个相邻输出 row 共享 activation load；完整 row 只做一次水平归约，
  避免每个 32-value block 单独归约。
- interleaved Gate/Up 使用 fused MXFP4 Gate-Up kernel；激活和 GPT-OSS
  SwiGLU 直接接在计算结果后，不物化解码后的权重。
- Scaled-SiLU 同时提供标准 `libm` 和 IEEE-754 多项式实现。进程首次使用
  时执行小型交错 microbenchmark，仅在近似路径有明确优势时选择它；
  选择结果在整个 Expert batch 外提并可观测，避免逐元素策略分支。
- `token_count == 1` 选择 GEMV 路径；Prefill/Expert batch 选择复用已解码
  block 的 GEMM 路径。
- 2–4 token 的 AVX-512 bulk row-pair 为每个 token 保持独立 SIMD 累加器，
  遍历全部 MXFP4 blocks 后才做一次水平归约；AVX2 对 2-token 提供同类
  路径，避免跨 Session 合批退化成逐 block 标量归约。
- 小 Dense Linear 限制线程扩张；CPU Expert cache hint 每个 buffer 最多
  预取 4 KiB。该 hint 仅在显式 `VulkanWithCpuPrefetch` 模式启用，因为
  长基准没有证明它能稳定优于默认模式。

### Vulkan Dense 与 Attention

- 混合模式把每层 RMSNorm、fused QKV、RoPE、GQA SDPA、learned sink、
  output projection、residual 和 KV 更新记录到一个 Vulkan Attention
  block：每层一次 hidden upload、一次 compute submission、一次 hidden
  download。
- Q/K/V 在模型编译时融合成一个 ncnn InnerProduct；同步上传完成后释放
  临时拼接的 CPU tensor。
- FP32 Vulkan Decode 会继续用一个原生 shader 融合 QKV Slice、三次
  Reshape、head/token Permute 和 Q/K RoPE；不满足 dtype/layout 条件时
  回退既有 ncnn 子图。`vulkan_attention_qkv_rope_fusions` 记录真实命中。
- 单 token Decode 可用第二个 FP32 shader 直接读取双写 KV ring，以在线
  Softmax 合并 QK、Softmax、PV，并直接生成 output projection 的 token-major
  输入；不再物化 score matrix，也不再执行其后的 Permute/Reshape。learned
  sink 常驻 device，命中时跳过 expanded mask 的 CPU 构造和上传。Runtime
  按 device、head shape 和 context 桶先各采样两次 ncnn/fused 路径，以 2%
  margin 固定偏好并每 256 block 稀疏复探；环境变量
  `NCNN_MOE_VULKAN_DECODE_SDPA=0|1` 仅用于诊断覆盖。
- KV Cache 在混合模式保持在 GPU，支持 full/sliding Attention。每个逻辑
  slot 同时写入 `slot` 与 `slot + capacity`；跨尾部的窗口因此仍是单个
  连续 VkMat offset view。容量不足时按几何级数扩展，只复制当前逻辑历史；
  正常 Decode 不再 concat/Slice 整段 KV。CPU KV 支持 FP32/BF16。
- 单 token 的 QKV+RoPE shader 直接把旋转后的 K/V 写入双写 ring 的两个
  物理副本，省去 K/V 中间 VkMat 和独立 append dispatch；它与后续选择原生
  Decode SDPA 或 ncnn SDPA 解耦。原生路径把 learned sink 作为逻辑零 K/V
  token，ncnn 回退只在需要时补写 sink 行。`NCNN_MOE_VULKAN_QKV_RING=0`
  仅用于隔离诊断。
- 当前 Vulkan Attention 的物理 activation/KV buffer 保持 FP32，即使模型
  元数据与 CPU KV cache 为 BF16；统计使用真实 device allocation 字节。
  原生 Decode SDPA 命中后不上传 mask，Prefill/ncnn 回退仍使用 FP32
  mask 与 RoPE staging。
- Prefill 默认按 256 token 分块，将 expanded mask 的峰值从完整 prompt
  二次规模降为 `chunk × context`；可将 `prefill_chunk_size` 设为 0
  恢复一次性 Prefill。
- LM Head 可使用 Vulkan。Router 保留在 CPU，避免在 CPU Expert 边界再做
  一次小矩阵上传、提交和下载。
- 两个独立 Vulkan staging slot/allocator 支持 staging buffer 容量复用，
  并记录 resize、reuse、acquisition、contention 指标。
- `VkCompute::submit_async()` 建立真实 fence 边界，再由 `wait()` 完成
  download/cast；析构/reset 会等待未完成 submission。双 staging slot
  允许并发 Session 使用不同 host buffer 并在取得设备锁前准备传输。
  共享 ncnn layer/allocator 的设备锁持有到 fence 完成，避免并发命令破坏
  确定性；每个 slot 的 `VkCompute` 跨调用 reset/reuse，减少 command
  object/buffer 重建。这仍不等于单 Session command-buffer pipeline。

### Expert 权重与内存

- `Expert` 是 Runtime 对象，记录 layer/id、权重 handle/bytes、cache
  state、memory location、kernel、热度、token 数和 cache hit/miss。
  `ExpertStore` 提供模型级查询和统计。
- 预加载 Memory Planner 根据物理内存、用户预算、Dense/Expert 估算以及
  最小活跃 Expert pair 大小，在 Eager 与 OnDemand 间选择。
- Dense BF16/F32 和 Eager MXFP4 优先使用 copy-on-write mmap，失败时回退
  到 buffered read。
- OnDemand MXFP4 cache：
  - 直接引用原 Safetensors shard 中的 block/scales；
  - 使用 byte-aware ARC：T1/T2 保存 recent/frequent 常驻项，B1/B2 只保存
    被淘汰 key 与逻辑字节，按 ghost hit 自动调节 T1 目标；
  - 容量、分区目标和 ghost history 均按字节计算，支持不同大小 Expert；
  - I/O worker 数默认由两倍模型最大 Top-K 与物理核预算共同决定，也可显式覆盖；
  - exact read 高优先，speculative read 低优先；过期预测会取消未开始读取；
  - speculative admission 只能替换未 pin 的 speculative resident，不能
    挤掉 demand-resident Expert；
  - 所有本层路由先 request，再优先计算已 ready 的 Expert；
  - 使用同层上一 token 路由做低优先级历史预取；不会在下一层 Attention
    之前用错误 hidden state 预测 Router；
  - Windows 默认使用对齐的 overlapped direct I/O，并在线程内复用 event
    handle；能力或对齐不满足时自动回退 buffered I/O；
  - compiled `ExpertPlan` 预计算稳定 cache key，热路径不再反复拼接 shard
    路径、offset 和 size；
  - overwrite buffer 不做无意义的预先 zero-fill。
- 可选 Vulkan executable Expert cache 直接保存 MXFP4 blocks/scales，并用
  原生 shader 融合 decode、FP32 accumulation、clamped SiLU、Gate/Up
  multiply 与 Down projection。两个投影共享 allocator 与 pipeline；E8M0
  scale 用精确 IEEE exponent bit 构造，避免逐 block `exp2`。cache 采用
  byte-aware ARC、two-touch admission、layer fairness 和异步 upload。
  一批 resident Expert 可与 CPU miss 并行执行；端到端 phase EWMA
  按 token-count 桶决定是否继续使用 GPU，失败或负收益均回退 CPU。
- `MemoryManager` 为每个 Session 维护 Execution Tensor 的 location、字节、
  使用次数和 transition 统计。当前记录真实执行边界：Vulkan Attention
  的 KV 留在 Vulkan，而传给 CPU Router 的 hidden 已回到 CPU。

### 可观测性与基准

- Session 统计覆盖 Attention/Router/Expert 总时间，并细分 Expert cache
  wait、task compute、regroup、combine、Embedding、FinalNorm、LM Head；
  还记录路由热度、GEMV/GEMM row、ARC T1/T2/B1/B2 字节与 ghost hit、
  取消/丢弃的预测、GPU Expert cache/execution、Vulkan submit/upload/
  download、QKV+RoPE/QKV-to-ring/Decode-SDPA fusion、staging reuse 和
  command-buffer reuse。
- `tools/benchmark_gpt_oss.py` 固定 warm-up/repetition，校验 token parity，
  输出 JSON，并可采集 process RSS 和 NVIDIA device memory。stdout/stderr
  使用临时文件承接，避免 Windows 匿名 pipe 在详细统计超过容量时形成
  子进程写入/父进程等待的死锁。
- greedy `temperature == 0` 使用线性 argmax，并保持相同 logit 选择较小
  token id；不再对完整词表做 `O(V log V)` 稳定排序。
- BF16 Router 的 dot product 运行时分派到 AVX2、AVX-512、ARM NEON 或
  scalar fallback。120B 的 32-token Router 时间由 232.4 ms 降至约
  46 ms，输出 token 保持一致。
- 9800X3D + RTX 5070 Ti 最终三次中位数：GPT-OSS-20B 64-token 为
  14.239 token/s；GPT-OSS-120B 32-token、16 GiB ARC 为 2.435 token/s。
  后者每次测试读取约 20.66 GB Expert 数据，仍受存储驻留限制。
- 仓库已记录 GPT-OSS-120B 在 32 GiB RAM + 16 GiB VRAM 设备上的重复基准；
  这些数字是该设备上的验证结果，不是其它 CPU/GPU 的性能承诺。
- 20 GiB layer-aware ARC 经一次同 Model warm-up 后，120B 默认单 Session
  最新三次中位数为 **10.945 token/s**。三个样本为
  10.574/10.945/11.815，token 序列全部一致。两个独立 Session 通过
  scheduler 执行时，
  64 tokens 的三次中位数为 4.792 s，即 **13.36 token/s aggregate**；
  三个样本为 13.36/13.47/13.02，token 序列全部一致。该数字是并发吞吐，
  不是单请求 latency。
- 最新 16-token A/B 中，QKV+RoPE fusion 在 576/576 个 Attention block
  命中，单 Session 中位从 11.609 提升到 **11.941 token/s**，Attention
  时间从 549.1 ms 降到 512.8 ms。双 Session 自适应 scheduler 使用每
  worker 8 个逻辑线程时，三次样本为 19.71/19.98/20.30 token/s，
  中位 **19.977 token/s aggregate**。后续默认复验为 17.62–19.58
  token/s；因此这是“接近 20”的本机短窗口服务吞吐，不是单 Session 或
  长上下文保证。
- 在线 Decode SDPA 的同 build 强制 A/B 将 Attention 从 506.5 降到
  494.5 ms（-2.39%），吞吐从 12.234 提升到 12.397 token/s（+1.33%）。
  最终默认自动策略报告在 576 block 中命中 538 次，辅助上传从禁用路径的
  1,728 次/1,769,472 bytes 降到 1,190 次/337,920 bytes；三次单 Session
  中位为 **12.359 token/s**、Attention 493.5 ms。双 Session 三次中位为
  **19.847 token/s aggregate**，仍未把 20 token/s 证明为稳定下限。
- QKV→ring 直写的隔离 A/B 在两侧都强制在线 SDPA：576/576 block 命中，
  Attention 494.368 → 487.553 ms（-1.38%），吞吐 12.380 → 12.555 token/s。
  最新默认路径在单/双 Session 均为 576/576 直写；双 Session 的一个三次
  报告为 20.039/21.101/19.863 token/s，中位 **20.039 aggregate**。后续
  同策略复验虽将 Attention 597.442 → 580.588 ms，但因 Expert 中位升至
  760.813 ms，整体中位为 19.481；因此 20 仍是短窗口工作点，不是稳定下限。

## 最低运行时与退化行为

最低运行时强调“能正确运行”，不承诺峰值性能：

| 条件 | 行为 |
| --- | --- |
| 无 Vulkan 或 `NCNN_MOE_USE_VULKAN=OFF` | `Auto` 选择 CPU-only；Attention、Dense、Router、Expert 均在 CPU |
| 仅检测到软件 CPU Vulkan | `Auto` 选择 CPU-only；显式 mixed/device 选择仍可用于调试 |
| `NCNN_MOE_USE_NCNN=OFF` | 使用项目内 portable CPU Linear/Attention/MXFP4，不依赖 ncnn operator |
| 无 OpenMP | Expert group 单线程执行；结果保持一致 |
| x86 无 AVX2/AVX-512 | MXFP4 自动使用 Scalar kernel |
| ARM 构建提供 SVE2 | 独立 SVE2 translation unit 参与运行时 microbenchmark；不胜出则 NEON/Scalar |
| ARM 仅提供 NEON | 使用 NEON MXFP4；否则使用 Scalar |
| x86 提供 VNNI/AMX | 只报告 capability；精确 MXFP4×FP32 模式不做有损 activation 量化，因此不误分派 |
| 内存足够 | `Auto` 选择 Eager Expert residency |
| 内存不足但 checkpoint 可按需读取 | `Auto` 选择 file-backed OnDemand + bounded cache |
| Expert pair 大于 cache | 精确请求返回容量错误；预测请求静默丢弃并增加统计，不破坏 demand resident |
| mmap 不可用/失败 | Dense/Eager tensor 回退 buffered read |
| 显式 mmap OnDemand Expert | 可以运行，但已验证设备上可能因大量 range mapping 变慢，因此不由 `Auto` 选择 |
| `prefill_chunk_size = 0` | 一次处理整个 prompt，速度可能更高，但 mask/activation 峰值更大 |
| 多个 Session | 可用 `BatchScheduler` 并发；CPU Expert 按 worker 分配线程，共享 Vulkan 命令执行到 fence 完成前串行；同一 Session 请求仍串行 |
| 多个 Vulkan 设备 | 可传候选 index 集合；编译器按 score、层数和 expected concurrency 选择实际层/Tensor placement，零收益设备保持闲置 |
| Vulkan Expert 实测更慢或执行失败 | 当前 token-count 桶退回 CPU，停止无界 admission；输出保持 CPU fallback |

模型最低内存没有单一常数，它取决于 Dense 常驻字节、KV Cache、一个活跃
Expert pair、工作 buffer 和用户 cache 预算。Memory Planner 会在读完整
Expert 权重之前验证预算；无法容纳最小活跃集合时返回明确错误。

显式拒绝而不是静默降级的行为：

- `VulkanOnly` 尚未实现；
- 未注册的 model type；
- 共享 Expert；
- 当前不支持的 Dense/KV/Expert dtype 或 shape；
- `TopKCandidates`/`SampledToken` logits output mode；
- 超过安全 Vulkan heap 比例或小于一个 Expert pair 的 GPU cache；
- 已编译 Vulkan operator 缺失所需设备能力。

## 与目标 Phase 的对应状态

| Phase | 状态 |
| --- | --- |
| 1 MoeIR | 已建立真实 Value/Node 图、QuantConfig、KV state 和兼容规范化；生产 registry 只注册已通过真实 checkpoint 验收的 GPT-OSS Adapter |
| 2 Execution Graph/Scheduler | Tensor dependency、backend lane、event、topological wave、跨调用 Session 收集、同 Expert 合批与分桶在线 staged/independent 选择已完成；单 Session speculative/跨 token pipeline 待补 |
| 3 Expert Runtime | Expert/ExpertStore、token grouping、batching、OpenMP 和热度统计已完成 |
| 4 Expert Kernel | fused MXFP4、Scalar/NEON/SVE2/AVX2/AVX-512、GEMV/GEMM、原生 Vulkan MXFP4 Expert 已完成；VNNI/AMX 有能力探测但无精确默认 kernel |
| 5 Expert Memory | mmap、lazy load、byte-aware ARC、unload、受保护预测 prefetch、可执行 GPU ARC cache 已完成 |
| 6 Hybrid | GPU Dense/Attention + QKV/在线 Decode-SDPA fusion + 双写 KV ring + CPU/Vulkan Expert overlap、async submit/wait、双 staging slot、command reuse 和多 Vulkan placement 已完成；单 Session 跨 token pipeline 待补 |

当前 120B 的单 Session 16-token 最佳既有路径达到 12.359 token/s 三次中位数；
双 Session 的最佳短窗口服务吞吐中位数为 20.039 token/s aggregate，最新
同策略复验为 19.481 token/s。单 Session 32-token 的既有验收仍为
10.945 token/s。
关键通用优化不是设备特例：
Execution Graph 每层只生成一个动态 `ExpertGroup`，而 decode regroup 根据实际
复制元素数保持串行，仅在大批量达到阈值时启动 OpenMP。冷工作集仍会被每 token
约 2 GB 的活跃 Expert 权重流量限制，增加线程或单独替换淘汰策略不能消除这个
I/O 下界。

20 token/s 的阶段预算、PCIe/Expert 热集下界以及 Vulkan Expert、
扩展 ISA、Attention 融合、多 GPU 和自适应 micro-batch 的实现/验收见
[`gpt-oss-120b-20tps-roadmap.md`](gpt-oss-120b-20tps-roadmap.md)。当前
Runtime 已增加 CPU ISA、Vulkan shader 能力、Expert batch/route 权重流量
以及按设备 heap 比例计算的静态热集覆盖观测；这些指标用于后续实测选址，
不是设备或模型名称分支。

## 2026-07-25 Profiler、Autotune 与 120B 单 Session 验收

- Expert Profiler 将总墙钟拆分为 engine、compute、cache management/wait、
  regroup、combine 和未归因 orchestration；旧 compute 指标重复累计 Top-K
  Expert wave 的问题已经修正。
- MXFP4 ISA 由运行时交错 microbenchmark 在实际支持的 Scalar、NEON、
  AVX2 和 AVX-512 候选中选择；`NCNN_MOE_MXFP4_KERNEL` 仅作为显式诊断覆盖。
- 连续 row-pair 分组独立 Autotune，将 OpenMP outer scheduling 纳入测量，
  只有超过 5% 的实测优势才切换，
  `NCNN_MOE_MXFP4_DECODE_GROUP=1|2` 可复现实验。默认验收选择 AVX-512、
  group size 2，不含设备名称或模型名称分支。
- 热缓存批量 acquire 在一个锁区内验证并 pin 当前层全部 Expert，第二遍直接复用
  已找到的 Entry；Expert 热度/命中统计使用 relaxed atomic，避免观测锁进入热路。
- 每个 Session 持有 Expert 执行 scratch arena；ActiveExpert、输入/输出 Batch、
  cache request/lease、decode task 和 MXFP4 临时 Batch 容量跨 token 复用。
- 120B 最终验收命令使用 `D:\Models\gpt-oss-120b`、28 GiB host budget、
  20 GiB ARC、一次 cache warm-up、32 tokens 和三次测量。中位数为
  **10.945 token/s**，Expert cache hit rate 100%，输出序列三次一致。报告位于
  `build-ncnn/gpt-oss-120b-runtime-final2-32x3.json`。
- Vulkan KV 已替换为设备端双写 ring；测试连续推进超过初始 16-slot 容量，
  确认 wrapped view 发生且 CPU/Vulkan logits 在 `1e-4` 内一致。Profiler
  记录 append、resize 和 wrapped-view 次数。
- `BatchScheduler` 已提供真实跨 Session Expert 合批：同输入的 120B 双
  Session 试验将 8,928 个逻辑 Expert batch 合并为 4,464 个物理 batch，
  32 tokens × 2 的单次结果为 13.624 token/s aggregate。该数字是一次架构
  验证，不替代上文已有的三次服务吞吐验收。

## 2026-07-26 原生 Expert、Attention、多设备与自适应调度验收

- 原生 Vulkan MXFP4 Expert microbenchmark 在 RTX 5070 Ti 上将一个完整
  Expert 的 token-1 时间从 0.1603 ms 降到 0.1261 ms，token-4 从
  0.3972 ms 降到 0.3128 ms，归一化误差小于 `2e-6`。真实 120B 串行
  GPU cache 仍会因 CPU/GPU 争用变慢，因此 Runtime 以整段 Expert phase
  而非孤立 kernel 决策，并在负收益时自动禁用该桶。
- CPU row-pair group A/B：group 1 的 16-token 中位为 10.868 token/s，
  group 2 为 11.609 token/s（+6.8%），Expert 时间 787.6 → 723.9 ms。
  自动 benchmark 包含 OpenMP outer scheduling 后在当前 host 选择 group 2。
- QKV+RoPE shader 的 120B A/B 报告为
  `build-ncnn/gpt-oss-120b-attention-fused-auto-group-16x3.json`：
  576/576 block 命中，中位 11.941 token/s。
- 在线 Decode SDPA + device-resident sink 的最终自动报告为
  `build-ncnn/gpt-oss-120b-decode-sdpa-maskless-auto-final-16x3.json`：
  538/576 block 命中，Attention 493.5 ms，三次中位 12.359 token/s。
  强制同 build A/B 的 Attention 为 506.5 → 494.5 ms；自动无预热单次
  将辅助上传从 1,728 次/1,769,472 bytes 降到
  1,190 次/337,920 bytes。
- 双 Session 在线策略报告为
  `build-ncnn/gpt-oss-120b-adaptive-parallel2-threads8-16x3.json`：
  19.71/19.98/20.30 token/s，中位 19.977 aggregate。runner 已移除物理核
  数覆盖，默认由 scheduler 按逻辑处理器/worker 计算线程数。
- 加入自动 Decode SDPA 后的双 Session 复验为
  `build-ncnn/gpt-oss-120b-decode-sdpa-maskless-auto-parallel2-threads8-16x3.json`：
  19.868/19.847/19.597 token/s，中位 19.847 aggregate。
- QKV+RoPE 现可直接双写 KV ring，并在 ncnn SDPA 回退时保持可用。隔离 A/B
  将 Attention 从 494.368 降到 487.553 ms；最佳双 Session 三次报告为
  20.039/21.101/19.863 token/s，中位 20.039 aggregate。随后同策略报告中
  Attention 继续降至 580.588 ms，但 Expert 抖动使总体中位回到 19.481。
- RTX 5070 Ti + AMD iGPU 的强制静态 29/7 层分配只有 7.979 token/s，
  Attention 由 568.2 增到 2302.3 ms。加入 expected-concurrency makespan
  目标后，同一候选集合自动选择 `0:36`，单次复验回到 18.599 token/s。
- 32-token 双 Session、20 GiB ARC 试验因热集扩张达到约 20.5 GiB RSS 和
  31.5 GiB private commit，发生严重内存压力并被终止。该配置没有吞吐
  acceptance；短驻留窗口的接近 20 token/s 不能外推到持续长上下文。
## 2026-07-26 scheduler and MXFP4 follow-up

- The public GPT-OSS runner and benchmark expose
  `--scheduler-staging auto|force|off`. These are policy controls, not
  model/device branches.
- A new scheduler bucket first measures its predicted path and delays the
  alternative sample until the adaptive probe interval (32 decisions by
  default). Long-running workloads still explore and switch with hysteresis;
  short requests no longer pay an immediate cold probe.
- Single-token grouped MXFP4 execution maps flattened row groups to active
  Experts in O(1) when their projection shapes are uniform. Heterogeneous
  Expert groups preserve the original shape-driven scan.
- The MSVC AVX-512 batch-two kernel shares activation input loads across four
  adjacent output rows. `NCNN_MOE_MXFP4_BATCH2_ROW_GROUP=0` is an isolated
  diagnostic fallback; other ISAs retain their existing runtime-selected
  kernels.
- Rejected measurements remain rejected defaults: an eight-row AVX-512
  experiment caused register spilling, forced staged execution remained slower
  than independent execution, OpenMP wait-policy tuning was unstable, and
  concurrent Vulkan Attention submissions increased contention and memory.
- The latest two-Session 120B samples are 20.315, 19.953, and 19.982 token/s.
  The median-generation result is 19.982 token/s with identical output tokens.
  This is a local short-window service result, not a hardware-independent floor.

## 2026-07-26 confidence-aware long-window scheduling

- Staged-versus-independent policies remain keyed by context length and
  request count. Each new bucket waits for its base probe interval before
  sampling the alternative.
- A prospective switch is confirmed on the immediately following batch. The
  required advantage is 10% with fewer than four candidate samples, 7.5% with
  four to seven samples, and 5% after eight samples.
- When the measured path gap exceeds 10%, 25%, or 50%, the probe interval is
  expanded by 2x, 4x, or 8x. This reduces repeated exploration cost without
  encoding a model, CPU, GPU, or fixed context threshold.
- In a 96-token x two-Session GPT-OSS-120B diagnostic, the former policy used
  staged execution for 45/95 decisions and measured 7.366 token/s. The final
  policy paid two staged probes, made no switch, and measured 8.026 token/s
  with identical generated sequences.
- That final run still read 30.56 GiB of Expert data at a 90.88% ARC hit rate.
  It characterizes storage pressure and does not establish sustained
  20 token/s.
- An exact E8M0 CPU bit-construction experiment was reverted: seven alternating
  MSVC AVX-512 microbench rounds regressed median Expert latency by about 1.8%
  because the existing 1 KiB scale table remained L1-resident.

## 2026-07-26 compressed Vulkan Expert L2

- `RuntimeOptions::expert_gpu_victim_cache_bytes` configures a
  compressed-weight Vulkan L2 behind the host Expert ARC. It may coexist with
  the executable native Vulkan Expert cache.
- Each tier's capacity is split independently across active Vulkan devices by
  runtime capability score, while deterministic victim-key sharding preserves
  device-local allocators, queues, and locks. Their combined per-device
  capacity is limited to one quarter of the reported heap budget; every
  enabled tier/device must hold at least one complete Expert pair.
- `Runtime::synchronize_model_caches()` drains host Expert reads, all
  victim-cache admissions, and executable Expert admissions only at an
  explicit warm-up or traffic-transition boundary. The GPT-OSS runner uses it
  after `--cache-warmup-runs`; ordinary inference remains asynchronous.
- A synchronized three-run GPT-OSS-120B 96-token/two-Session acceptance with
  a 3 GiB L2 measured 8.035, 8.193, and 8.798 token/s, or 8.193 median.
  Median Expert reads were 28.20 GB and median GPU restores were 2.42 GB in
  0.772 seconds. The comparable no-L2 diagnostic read 32.81 GB at 8.026
  token/s.
- Standard ARC was rejected for this lower tier after a 7.886 token/s screen.
  The lower cache observes host-ARC evictions instead of original accesses,
  and frequency promotion retained stale entries. RAM and executable Expert
  caches keep ARC; the compressed L2 keeps LRU. A 3.5 GiB screen showed no
  clear phase gain and raised peak NVIDIA memory to 15,216 MiB; 4 GiB was
  rejected by heap-budget validation.

## 2026-07-26 packed ExpertStore ranges

- `tools/pack_mxfp4_experts.py` creates an optional Safetensors sidecar and
  preserves the original checkpoint. The current writer recognizes GPT-OSS;
  the runtime naming convention and contiguous-range reader are model-neutral.
- A packed Expert stores Gate/Up blocks, Gate/Up scales, Down blocks, and Down
  scales consecutively. The cache validates exact contiguity, issues one
  read, and exposes four aliasing byte-buffer views over that allocation.
  Non-packed tensors retain the existing four-range path.
- The official 120B sidecar is 56.73 GiB and was generated in 62.4 seconds on
  the test NVMe device. It is automatically detected; absence has no runtime
  behavior or storage cost.
- Three 96-token/two-Session runs with a 22 GiB RAM ARC and 3 GiB compressed
  Vulkan L2 measured 9.833/9.125/9.371 token/s, or **9.371 median**. Median
  Expert wait was 8.127 seconds and every direct disk miss was one 12.61 MiB
  range. Token output matched.
- A 23 GiB ARC regressed to a 6.873 token/s median on the 31.14 GiB host
  because Expert compute roughly doubled under memory pressure. The runtime
  therefore continues to treat capacity as a hardware-budget decision.

## 2026-07-26 coexisting GPU cache tiers

- The executable Vulkan Expert ARC and compressed Vulkan victim L2 now share
  one per-device safety envelope. Empty-device and capacity-overflow paths
  return explicit errors.
- Session, runner, and JSON benchmark metrics separate executable-cache
  admissions/uploads/native executions from victim-cache
  admissions/restores/downloads. The staged multi-Session path records victim
  deltas as well.
- Under a 3 GiB aggregate GPU budget, the 512 MiB executable + 2.5 GiB victim
  split produced one 9.819 token/s median cohort but only 9.088 token/s in the
  independent split-metrics validation (9.402/9.088/8.622). Median cache wait
  remained 8.687 seconds.
- A 256 MiB + 2.75 GiB split reached 9.994 token/s once but regressed to 8.494
  by three-run median. Coexistence is accepted as general Runtime
  infrastructure; neither split becomes a cross-device default.
- Explicitly requesting eight I/O workers measured 9.441 token/s because the
  shape-driven Top-4 default already selects eight on this host. Leaving the
  read policy adaptive measured 9.787 token/s once and still converged to
  direct I/O before measurement. These screens reject fixed thread-count and
  filesystem-policy tuning as general optimizations.
## 2026-07-26 victim admission and Attention epilogue audit

- The optional Vulkan victim tier now supports a bounded cross-residency
  ghost probe. First host-ARC evictions are sampled at `1/N`; a key observed
  again before its payload-byte-bounded ghost expires is admitted
  unconditionally. `N=1` is the compatibility/default path and performs no
  ghost locking.
- This mechanism is retained as portable diagnostic infrastructure, not a
  default optimization. GPT-OSS-120B staging-off screens measured 8.633
  token/s at `N=2` and 8.318 token/s at `N=4`, below the 9.546 token/s
  admit-all cohort. Upload traffic fell to 13.35/6.62 GiB, but the reduced or
  delayed restore benefit did not improve end-to-end throughput.
- Moving the Attention residual add from Vulkan to the host projection
  download loop was tested and reverted. The three-run median fell to 9.151
  token/s and median Attention time did not improve. In a heterogeneous
  two-Session pipeline, fewer GPU dispatches are not automatically faster if
  work is shifted onto CPU cores executing Experts.
- The remaining general opportunity is to avoid the compressed
  victim-to-CPU round trip by handing a resident GPU weight entry directly to
  the executable Vulkan Expert backend, subject to its existing online
  CPU/GPU phase-benefit policy.

## 2026-07-26 设备驻留 victim Expert 直执行

- compressed victim entry 现在可直接作为 Vulkan MXFP4 Expert 的权重存储。
  Gate/Up blocks、Gate/Up scales、Down blocks、Down scales 在同一个 `VkMat`
  中按设备 offset alignment 对齐，kernel 通过共享 subview 读取，不再要求
  `GPU -> CPU -> GPU` 往返。
- borrowed operation 同时持有完整 `DeviceEntry` lease。victim 淘汰器继续使用
  通用 pinned-entry 规则，异步命令完成前不能回收对应 Vulkan buffer。
- lookup 只做 peek；只有成功执行才刷新 LRU。未执行的候选不会污染 victim
  热度，也不会因为一次调度探测改变后续 restore 命中分布。
- executable state 采用 lazy 构造。普通 admission 只上传压缩权重并记录轻量
  shape/offset/bias 元数据；只有在线 phase policy 批准的候选才上传 bias 并创建
  Expert operator，避免在所有设备上无条件争用 Vulkan command mutex。
- device-source 有独立的保守校准：先收集 CPU phase baseline，再按 1/64 稀疏
  探测，通过 0.90/1.02 hysteresis 判断启用/关闭。判断依据是完整 Expert phase，
  而非设备名称、模型名称或单独 kernel 峰值。
- 多 Vulkan 设备使用稳定 key sharding，使 executable backend 与 victim shard
  位于同一设备；单设备和非 Vulkan 构建保留原有回退。
- 新统计项包括 device-source hit/miss/execution/failure；runner 与 JSON
  benchmark 均可观察。`--disable-gpu-victim-execution` 提供同二进制
  A/B 和兼容回退。
- 120B 长程测试先发现并修复了未 pin entry 导致的 `0xC0000374`。最终 lazy
  版本在高后台负载配对测试中为 4.294 vs 4.245 token/s，10 次直执行、0 failure，
  restore 时间从 3.107 s 降至 2.710 s。该结果证明安全与低默认成本，不作为
  10 token/s 的统计验收；现有空闲度更高的三次长程中位数仍为 9.546 token/s。

## 2026-07-26 代码规范化与复测

- 生产 Adapter registry 已收敛为 GPT-OSS-only；未验收模型的 parser、
  weight mapping、测试分支和文档声明均已移除。测试专用 `test_moe` fixture
  仍只链接到测试目标。
- 73 个项目 C/C++ 文件按仓库 ncnn-style `.clang-format` 统一；style check
  同时拒绝被旧 C++03 formatter 破坏的 raw string。生产代码剩余 43 个注释
  位置，均属于契约、不变量、平台或性能原因。
- ExpertStore 删除重复 hash index、snapshot copy 和 mutex；Session 与
  MemoryManager 分别把一次 generate/整图 residency 更新收进单一临界区；
  file handle/mapping cache 使用 shared read lock，并把 open/mapping syscall
  移到写锁外；Windows I/O event 按 worker 复用。
- MSVC Vulkan、MinGW Vulkan 均通过 3/3 tests，portable CPU 通过 2/2 tests。
  Python benchmark tools 通过 bytecode compile；新诊断开关为
  `--disable-gpu-victim-execution`。
- 同一 120B、96-token、双 Session、22 GiB Host ARC、512 MiB executable
  cache、2.5 GiB victim、staging-off 协议的规范化后二进制测得
  5.712/5.485/8.580 token/s，中位数 5.712。立即用规范化前二进制控制复测也
  只有 5.090 token/s。两者 ARC hit/read volume 接近（约 92%/23 GiB），但
  Expert compute 分别升至 18.77/21.47 秒，说明本轮受到系统负载、频率或
  内存带宽状态污染，不能归因为格式/API 重构。
- 因此性能进度仍以最近的可比空闲三次样本
  9.546/8.483/10.137 token/s（中位数 9.546）为准；10 token/s 持续门槛
  尚未通过。污染样本保留在
  `build-ncnn/gpt-oss-120b-code-normalization-staging-off-96x3.json` 和
  `build-ncnn/gpt-oss-120b-diagnostic-old-binary-96x1.json`，不会被选择性
  删除或写成正向性能结论。
- 相邻的 16-token × 双 Session resident A/B 中，规范化后二进制为
  15.711/17.090/18.492 token/s（中位数 17.090），规范化前二进制为
  15.968/15.265/16.856 token/s（中位数 15.968）；同系统状态下新版本高
  7.0%，两组 cache wait 均为 0 且 token 完全一致。它证明重构没有造成
  可见退化，但两组都低于历史空闲 20.093 中位数，因此不替换峰值记录。
