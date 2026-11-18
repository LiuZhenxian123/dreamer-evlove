# 实验手册

这份文档写给执行实验的人或代理（Codex）。代码在一台无 GPU 的机器上只做过 CPU 冒烟测试，
**GPU 上的完整训练、torch.compile 路径和真实 LLM 接口都还没有跑过**，第一次运行请先做第 0 步。

## 0. 环境与自检

```bash
# Python 3.11，建议用 uv
pip install -e ".[dmc,metaworld]"
pip install pytest

export MUJOCO_GL=egl
python -m pytest tests -q            # 约 2 分钟，应全部通过（40 项）

# LLM：任何 OpenAI 兼容接口。密钥只通过环境变量传入，不要写进配置或提交到仓库。
export SEA_LLM_BASE_URL=https://api.openai.com/v1     # 或本地 vLLM: http://localhost:8000/v1
export SEA_LLM_API_KEY=...
```

GPU 冒烟（几分钟内应看到 `[sea] saved scoring dataset` 和 `[evolve:...]` 日志）：

```bash
python train.py env=metaworld env.task=metaworld_push env.steps=3e4 env.env_num=4 \
    model.rep_loss=dreamer sea.enabled=True sea.mode=evolve sea.evolve_at=[0.3,0.6] \
    sea.evolver.generations=1 sea.dump_data_at=[0.5] logdir=logdir/smoke
```

若 `torch.compile` 在语义头相关代码上报错，先用 `model.compile=False` 确认逻辑正确，再报告编译错误。

## 1. 试点：免训练适应度能否预测下游回报（决定整个方案是否成立）

这是必须最先做的实验。它回答：`sea/fitness.py` 给一个抽象打的分，和用这个抽象训练出来的智能体表现，
是否秩相关。没有这个证据，后面的进化就没有依据。

每个任务四个阶段，脚本是 `runs/sea_pilot.sh <stage> <task> [gpu]`：

| 阶段 | 做什么 | 产物 |
|---|---|---|
| `collect` | 训练一个不带语义头的基线，在 20% 和 50% 处保存打分数据集 | `logdir/pilot/<task>/collect/datasets/step_*.pt` |
| `candidates` | 生成候选：LLM 一次性采样 30 个 + 原始状态 1 个 + 随机投影 3 个，并打分 | `programs/*.py`、`scores.json` |
| `train` | 每个候选 × 每个种子训练一次（`sea.mode=fixed`） | `runs/<program>_s<seed>/` |
| `analyze` | 计算各适应度分量与最终表现的 Spearman 相关及自助法置信区间 | `rankcorr.json` |

建议任务（覆盖抓取、工具使用、运动）：`metaworld_pick-place`、`metaworld_stick-push`、`dmc_walker_run`。

可调的环境变量：`STEPS`（默认 5.1e5）、`N_ONESHOT`（默认 30）、`SEEDS`（默认 "0 1"）、`REP`（默认 dreamer）。
`train` 阶段是主要开销：34 个候选 × 2 个种子 = 68 次训练/任务，可按 GPU 数量把循环拆开并行（已完成的会跳过）。

**判定标准（建议，可与作者商定）：**

- 通过：综合分 `score` 的 Spearman ρ 在至少两个任务上 ≥ 0.5，且 95% 置信区间不含 0。
- 部分通过：综合分不行，但某些分量（如 `reward`、`visible`）单独相关。此时调整
  `sea.evolver.weights`，把不相关的分量权重设为 0，再用另一批候选验证，避免在同一批数据上调参。
- 不通过：所有分量都不相关。停止，向作者汇报，不要继续第 2 步。

请同时报告：被拒绝的候选比例及原因、各分量之间的相关、原始状态和随机投影基线在排序中的位置。

## 2. 主实验（试点通过后）

`runs/sea_main.sh <method> <task> <seed> [gpu]`，方法包括：

| 方法 | 含义 | 对应的已有工作 |
|---|---|---|
| `base` | 无特权监督 | DreamerV3 / R2-Dreamer |
| `raw` | 预测全部原始特权状态 | Informed Dreamer 式 |
| `random` | 原始状态的随机线性投影 | 对照：选什么是否重要 |
| `oneshot` | LLM 一次生成，不打分不迭代 | DreamTIP 式 |
| `evolve` | 训练中进化，由免训练适应度选择 | 本方法 |
| `evolve_lib` | 进化 + 跨任务抽象库 | 本方法（完整） |

Meta-World 默认在 50 个固定训练布局上训练，在 50 个留出布局上评估（`env.configs=train env.eval_configs=test`）。
DMC 在默认动力学下训练，训练结束后由 `eval.py` 在 12 种电机强度/摩擦偏移下评估。

建议 5 个种子。脚本结束时会写 `eval_*.json`；训练曲线在 `metrics.jsonl` 和 TensorBoard。

### 多任务与抽象库

```bash
# 多任务训练（10 个任务共享一个世界模型，编码器接收任务 one-hot）
python train.py env=metaworld_mt model.rep_loss=dreamer sea.enabled=True sea.mode=evolve \
    sea.library=logdir/library_mt sea.refactor_every=2 logdir=logdir/mt/evolve_s0 seed=0

# 留出任务：先在若干任务上建库，再在新任务上带库/不带库各跑一次
SEA_LIBRARY=logdir/library_A runs/sea_main.sh evolve_lib metaworld_door-open 0     # 建库任务，逐个跑
SEA_LIBRARY=logdir/library_A runs/sea_main.sh evolve_lib metaworld_drawer-open 0   # 留出任务，带库
runs/sea_main.sh evolve metaworld_drawer-open 0                                    # 留出任务，无库对照
```

抽象库需要报告的指标（已有文献指出学到的库常常没被复用，所以这些是必须的）：

- `logdir/.../sea/library_stats.jsonl`：被多个任务使用、被多个任务选中的条目数。
- 各任务 `sea/<task>/history.jsonl` 中的 `library_tried`、`library_kept`、`llm_calls`：
  带库时达到同样适应度所需的 LLM 调用数是否减少。
- 无库对照（`evolve`）与带库（`evolve_lib`）在留出任务上的样本效率和最终表现。

注意：同一个库目录不要被多个并发进程同时写入。并行跑多个任务时，给每个进程设 `sea.library_write=False`
只读共享库，或让建库任务串行执行。

### 消融

```bash
# 去掉适应度的某一项
... sea.evolver.weights.closure=0     # 同理 reward / visible / control
# 进化时机：只在开头进化一次 vs 多次
... sea.evolve_at=[0.03]
# LLM 能力：换模型（包括本地开源模型）
... sea.llm.model=<model>
# 目标维度预算
... sea.budget_dims=8                 # 或 16 / 48（48 需要 sea.max_dim>=48）
```

## 3. 离线工具

```bash
python -m sea.cli score    --data D.pt --task T --programs dir_or_file --out scores.json
python -m sea.cli evolve   --data D.pt --task T --out run/ --library lib/      # 不训练智能体，只进化
python -m sea.cli library  --path lib/ --insights                              # 复用统计 + 提炼经验
```

`evolve` 写出的 `selected.json` 可以直接作为 `sea.programs` 用于 `sea.mode=fixed` 训练
（"先离线进化、再训练"的两阶段版本）。

## 4. 产物位置

| 路径 | 内容 |
|---|---|
| `<logdir>/metrics.jsonl` | 训练与评估标量；`sea/head_error_mean` 是语义头的平均归一化误差 |
| `<logdir>/sea/active.json` | 当前生效的抽象程序、所占槽位和所属任务 |
| `<logdir>/sea/active_history.jsonl` | 每次换代后生效的特征名 |
| `<logdir>/sea/<task>/archive.json` | 该任务的全部存档程序及各项得分 |
| `<logdir>/sea/<task>/history.jsonl` | 每代的最佳分、LLM 调用数和 token 用量 |
| `<logdir>/sea/llm_log.jsonl` | 全部提示词和回复（不含密钥） |
| `<logdir>/datasets/step_*.pt` | 打分数据集 |

## 5. 尚未实现

以下内容论文需要但本仓库没有，执行前请先确认是否要补：

- **CARLA 环境**：未接入，原稿的 CARLA 实验需要另外移植环境和特权状态读取。
- **HarmonyDream、DALI 基线**：未实现。R2-Dreamer、InfoNCE、DreamerPro 可通过 `model.rep_loss` 切换。
- **"用 RL 训练当适应度"的对照（LIMEN / LESR 式）**：未实现。这是审稿人会要求的对照，
  可在 `sea/evolve.py` 的 `_score` 处换成"短程训练后的回报"来做，需要另写调度。
- **TD-MPC2 等多任务基线**：未实现。
- **Scaffolder / Reinformed Dreamer 基线**：未实现，`raw` 只是最简单的特权预测对照。
- **DMC 多任务**：不支持（各任务的特权状态布局不同）。

## 6. 已知局限

- 适应度是代理指标：缓冲区平均的 L2 误差远弱于状态抽象理论界所要求的条件，不能声称有理论保证。
- 回归器是随机傅里叶特征上的岭回归，输入里无关维度多时会低估可预测性。这客观上惩罚了不紧凑的抽象，
  但也意味着高维抽象的得分偏低。
- 打分数据集的训练/测试划分是按序列做的，同一回合的不同片段可能分在两边。
- 程序沙箱基于语法白名单，能挡住 LLM 的常见错误，但不是安全边界，不要用它运行来源不可信的代码。
- 训练中途更换目标是否会扰动潜空间，尚未验证。对照方法是比较 `sea.evolve_at=[0.03]`（只换一次）
  与默认的多次换代。
