> Historical design documentation inherited from the base snapshot. It is not the V72b release specification or a validated launch recipe. See README.md for current preparation status.

# M2 融合流水线调度 — 原型设计方案

> 状态:设计稿,待评审后动手。基于 `m2/baseline @ f4f54ae10` 实测代码。
> 关联记忆:`m2-bubble-isolation-verified` / `m2-roofline-constants` / `m2-graph-reuse-nonissue` / `m2-prefill-is-the-lever`

---

## 0. 一句话

把现有 M2 的「decode 轮」和「prefill chunk」从**两个交替的独立 forward**,合并成**同一辆车(同一个 ticket / 同一次 `llama_decode`)里混装**,同时保留 G=3 辆车异步在飞——这就是用户描述的 `Aa | Bb Aa | Cc Bb Aa` 三车混装流水线。

## 0.1 两个改动的真正分工(重要,2026-06-14 评审修正)

这次改动有**两个独立杠杆,大小不同**:

- **融合(decode 折进 prefill 车)= 隔离机构,不是吞吐机构。** 它省的只是 **decode 侧** 的那一次权重读(「白嫖 decode」与「省一次权重读」是同一枚硬币,prefill 自身的读取负担**不变**)。prefill 计算律速,IO 轻一点不会显著提速;折进来的 ~15 token decode 是**边角料**。融合真正的价值是**让 decode 每辆车都搭车 → decode 节奏恒定 → 锯齿(sawtooth)消失**——这是定性效果(不再 stall),不是定量提吞吐。→ **P1 的成功判据是「锯齿振幅收窄」,不是「吞吐跳升」。**
- **均匀装车(各车坐 ridge、大小齐平)= 真正的大头。** 流水线吞吐由**最慢的车**律速;一辆超载车 = head-of-line 堵塞,其余车全等它。实测崩溃引信正是「闭环 × **大小交错**」组合(0.65,见 `m2-controlled-variable-findings`),大小交错 = 车不齐平。均匀装车直接潰掉崩溃要因。**吞吐红利在 P2/P3,不在 P1。**

> 「chunk 切到 GPU 跑满为止、越小越不堵」——**对,但只对到 ridge 为止,过了就反转**。GPU「跑满」点 = ridge(~512–1024)的定义本身;切到 ridge 以下:prefill 效率崩(256→64%、128→36%)+ 每车多付 ~0.1–0.15s 固定跨卡同步。两个效率反向:充填率 `N/(N+G-1)` 要车多(小),单发 roofline 要车 ≥ridge。**最优 = 能飽和 GPU 的最小车 = ridge,别再细。**

---

## 1. 现状:为什么现在是「交替」而不是「混装」

两套机制各自独立,从不进同一个 batch:

| 机制 | 入口函数 | 组装的 batch | 货物 |
|---|---|---|---|
| decode 流水线 | `pipe_issue(li)` `server-context.cpp:850` | `pipe_batch`(line 866 清空) | **仅** 每个 slot 一个 `slot.sampled`(line 873) |
| prefill 灌注 | `emit_prompt_chunk(i,n,lane)` `:4350` | `batch_view`(主 `batch` 的切片) | **仅** prompt chunk tokens |

驱动循环(`:4806-4836`):`pf_settle()` 跑一个 **纯 decode** 的 `pipe_round(true)`,然后 `emit_prompt_chunk()` 跑一个 **纯 prefill** 的 `llama_decode`,二者交替。

**代价(已实测,见 `m2-roofline-constants`)**:纯 decode forward 要付满 ~0.045s 的权重读地板;而把 8 个 decode 折进一个 1024-prefill forward 只加 ~0.008s(便宜 ~12×,**b≈0**)。现在每个 `pf_settle` 的 decode 轮都在白付地板,且 prefill chunk 经过三张卡时,decode 抢不到那三张卡的算力(用户的气泡洞察)。

> 注意:现有 `pf_lanes`(`:4757`)已经做了**请求间** prefill 的错位(不同序列的 chunk 轮流上车),但**没有把 decode 折进 prefill 车**。我们要补的正是后者。

---

## 2. 目标:混装车 = decode 折进 prefill,且 3 车异步

每辆车(= 一个 `pipe_issue` = 一个 ticket = 一次 `llama_decode`)装:

```
[ 本车 lane 所有 generating slot 各 1 个 decode token ]   ← 折进来,≈免费
  + [ 一段 prefill chunk,凑到 ridge ~512–1024 token ]    ← 主货
```

- 保持 **G=3 辆车在飞**(M2 的 3× 重叠红利);
- 车结算时:既 sample decode token 推进那些 slot,又检查 prefill chunk 是否填完某 prompt → 该 slot 转 generating,**下一轮自动并入 decode 折叠**;
- **没有 prefill 时**,车退化成纯 decode 流水线(就是今天的 `pipe_round` 稳态);**没有 decode 时**,车就是纯 prefill chunk(今天的 `emit_prompt_chunk`)。两端都是现有行为的子集 → 可增量切换、可回退。

这是唯一能在**两个维度同时**赢 DP 的形态:既拿 stock 的融合(decode 免费),又拿 M2 的跨卡重叠。

---

## 3. 核心改动:`pipe_issue` 里给车加挂 prefill 货

### 3.1 数据结构(`pipe_lane`, `:770`)

给每条 lane 增加一个**可选的 prefill 载荷**:

```cpp
struct pipe_lane {
    std::vector<int> slot_ids;   // 既有:本车的 decode slots
    std::vector<int> task_ids;
    uint64_t ticket    = 0;
    bool     in_flight = false;
    int64_t  t_issue   = 0;

    // 新增:本车搭载的 prefill 货(无货时 pf_slot_id < 0)
    int     pf_slot_id = -1;     // 正在灌 prompt 的 slot
    int32_t pf_start   = 0;      // 主 batch 中该 chunk 的起始 token 下标
    int32_t pf_len     = 0;      // 该 chunk 长度
    int     pf_i_batch = -1;     // chunk 内需采样首 token 的位置(prompt 末尾命中时)
};
```

### 3.2 `pipe_issue(li)` 组装(`:866-881`)

在现有 decode 添加循环(`:869-875`)**之后、`llama_decode` 之前**,把 prefill chunk 追加进**同一个** `pipe_batch`:

```cpp
// 既有:decode tokens
for (int sid : lane.slot_ids) { ... common_batch_add(pipe_batch, slot.sampled, ...); }

// 新增:把本车的 prefill chunk 追加到同一 batch
if (lane.pf_slot_id >= 0) {
    server_slot & ps = *get_slot_by_id(lane.pf_slot_id);
    for (int32_t k = 0; k < lane.pf_len; ++k) {
        const int32_t src = lane.pf_start + k;
        const bool need_logits = /* prompt 末 token? */;
        common_batch_add(pipe_batch, batch.token[src], batch.pos[src], { ps.id }, need_logits);
    }
    // 记录 chunk 内采样位置 → 结算时用
}
```

> KV/seq 正确性:每个 token 自带 seq_id+pos,异质 batch 一次 `llama_decode` 完全合法——stock 的 `update_slots` 正是如此(`can_batch_with` 不看 kv_unified,见 `m2-prefill-is-the-lever`)。

### 3.3 结算 `pipe_sample_oldest`(`:957`)

车结算时,在现有 decode 采样**之后**,补一段 prefill 收尾(等价于把 `emit_prompt_chunk` 的 `:4500-4565` 那段并进来):
- 推进该 prompt slot 的 `prompt.n_tokens`;
- 若本 chunk 填完了 prompt → `slot.state = SLOT_STATE_GENERATING`,**下一次** `pipe_issue` 时它就自动进 `slot_ids` 被折叠;
- 清 `lane.pf_*`(货已卸)。

---

## 4. 装车策略(调度器,新函数 `pipe_plan_cars()`)— **本方案的大头**

> 这一节是真正的吞吐杠杆(§0.1)。融合只负责隔离/灭锯齿;**齐平装车才是赢 DP 的关键**。
> 核心原则:**每辆车坐 ridge(~768),大小齐平,绝不出现超载车**(超载车 = head-of-line 堵全线)。

替换掉现有 `:4751-4836` 的 `pf_lanes`/plain 双分支。每次 `update_slots()` 在有 pending prompt 时:

1. **decode 优先填**:每辆车先挂上它 lane 的 generating slots(免费料,b≈0)。
2. **prefill 切货**:把所有 pending prompt 的 token 总量 `T`,切成 `ceil(T / target)` 份;`target` ≈ **768**(坐落 ridge 512–1024,见 `m2-roofline-constants` R1)。
3. **均车,不砍上限**:`target` 当**均数**而非天花板——按用户要求,装到「最接近」而非「不超过」。每车允许 ±,目标是**每车过 GPU 时间相近**(流水线效率核心)。
4. **权重折算填车**:一车「时间成本」≈ `prefill_tok + w · decode_tok`,实测 `w ≈ 1/12`(decode 折叠几乎免费)。用它把 decode 多的车 prefill 少装一点,使各车时间齐平。
5. **错位入场**:多个请求即使错峰到达,也按 lane 轮转上车(沿用现有 `pf_lanes` 的 round-robin 思想 `:4784-4802`),保证任意时刻 ≤3 车。
6. **(二期)上下文自适应**:ctx 每深 ~100k,`target` 调大补偿 +49% 注意力成本(`m2-roofline-constants` R3)。先用静态 768,二期再加慢跟踪控制器。

> 切几份的权衡:N 份流水线效率 = N/(N+G-1),N=4→67%、N=8→80%;再细则被每车固定开销 ~0.1–0.15s 吃掉。**4–8 份是甜点**。

---

## 5. 改哪些函数(精确清单)

| 文件:行 | 函数 | 改动 |
|---|---|---|
| `:770` | `struct pipe_lane` | +`pf_slot_id/pf_start/pf_len/pf_i_batch` |
| `:850` `pipe_issue` | **核心**:decode 循环后追加 prefill chunk 进 `pipe_batch`;commit 时一并推进 prompt slot |
| `:957` `pipe_sample_oldest` | 采样后补 prefill 收尾(prompt 完成→转 generating) |
| `:4751-4836` | prefill 驱动循环 | **重写**:`pf_lanes`/plain 双分支 → 统一 `pipe_plan_cars()` 给三车派 decode+prefill 货,再走 issue/settle 轮转 |
| 新增 | `pipe_plan_cars()` | §4 装车策略 |
| `:4313` `pf_settle` | 弱化/删除 | 不再需要「decode 轮 vs prefill chunk」交替;decode 已折进车 |

**不动**:`pipe_round` 稳态(纯 decode 时仍走它,prefill 货为空 = 现行为)、`pipe_drain`、KV contiguous lane(`pipe_lane_of` `:840`)、ticket/event 机制、回退到 legacy 的路径。

---

## 6. 最小改动增量路径(分期,每期可独立验证)

- **P0(脚手架,零行为变化)**:加 `pf_*` 字段,默认 -1;`pipe_issue` 里 `if (pf_slot_id>=0)` 分支不触发。跑闭环确认无回归。
- **P1(单请求融合)**:只在「1 个 prompt + 有 decode 在跑」时,把该 prompt 的 chunk 挂到一条 lane 上融合。**判据 = decode 锯齿振幅收窄(定性,灭 stall),不期望吞吐跳升**(§0.1)。
- **P2(多车切货)= 吞吐主战场**:接 `pipe_plan_cars()`,把大 prompt 切到 ridge(~768)、4–8 份、**齐平**铺满三车 + 错位多请求。**判据 = 闭环吞吐开始接近/超过 DP**。
- **P3(均车+权重精调)**:加 §4.4 时间折算 + §4.3 不砍上限装到最近,测每车过 GPU 时间方差,消灭残余超载车。
- **P4(上下文自适应)**:慢跟踪控制器。

环境开关:使用参数解析器的 `LLAMA_ARG_PIPE_FUSE=1` 门控，或显式 CLI `--pipe-fuse`。

---

## 7. 风险与回退

| 风险 | 评估 | 处置 |
|---|---|---|
| CUDA graph 复用被异质车破坏 | **已排除**(`m2-graph-reuse-nonissue`:FORCE_GRAPHS on/off ~0 差异,V100 上 stock 本就不开图) | 用 plain launch,不依赖 per-lane 稳定 shape |
| KV 连续性(`-kv` 非 unified) | decode lane 用 `pipe_lane_of` 绑定连续 slot 块;prefill slot 有自己的 seq | chunk 用 prompt slot 自身 seq,不破坏 lane 块 |
| 采样错位(decode 与 prefill 首 token 同 ticket) | i_batch 分别记录,vocab 行各取各的 | P1 单请求先验证 |
| KV 满 / retry / VIP suspend | 现有 `evict_one_for_capacity`/`pipe_drain` 已处理;融合后 chunk 失败需回滚未 commit 的 prompt 推进 | commit-after-decode 语义(`:895-901`)沿用 |
| 整体不达标 | — | `LLAMA_ARG_PIPE_FUSE=0` 回退；生产始终 DP，不受影响 |

---

## 8. 验证

- **工具**:闭环 `/tmp/m2_load.py`(已建,M2 77 vs DP 134 的那套),build-ipre 安全目录,绝不动 build-pipe / 9000 网关。
- **判据**:
  - P1:注入 prefill 时,bg decode 锯齿幅度显著收窄(对比 `m2-bubble-isolation-verified` 的 200→5)。
  - P3:闭环吞吐 ≥ DP(目标 >134),且 deep-context 不塌。
  - 每车过 GPU 时间方差 < 阈值(流水线齐平度)。
- 每轮测试后 `systemctl restart hulumed`,验证 3/3 healthy。
