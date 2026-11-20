# TelecomAgent-RL

本 recipe 将业务模拟与 Agent-R1 适配分开。当前是可本地验证业务逻辑的原型，
不是已训练模型或真实电信后端。任务和专家轨迹由规则生成，必须标注为 synthetic。
没有声称已拥有 700 个任务、1200 条专家轨迹或任何模型成功率。

## 环境边界

- 本地检查仅需 Python 3.11+ 标准库。Mac 不安装训练依赖，不运行 GPU 入口。
- `adapters.py`、`telecom_agent_flow.py` 只在 Agent-R1 环境中导入。
- SFT/模型评测/GRPO 实际运行会先检查平台和 CUDA，Mac 上在导入 torch 前拒绝。
- 没有修改原仓库 requirements、`agent_r1/` 或 VERL，没有自动安装包。
- GPU 环境按原仓库要求核验 `verl==0.7.0`；Transformers、PEFT、vLLM、CUDA
  的组合须在服务器确认。下列训练命令是入口说明，不代表已验证可训练。

## Phase 0：分支和本地检查

应在 `feature/telecom-agent` 工作，不自动切换或创建提交。
仅为电信环境源码增加了 `.gitignore` 的精确例外：原有 `ENV/` 在本机大小写不敏感
匹配中会忽略 `recipes/telecom/env/`，否则这些业务文件不会出现在 Git 状态中。

```bash
GIT_OPTIONAL_LOCKS=0 git status --short --branch
python3 -B -m unittest discover -s tests/telecom -v
git diff --check
```

业务测试不需要 pytest、numpy 或任何训练库；专家回放成功只能证明模拟器契约，
不能当作模型评测结果。语法检查可以用标准库 `compile(source, filename, 'exec')`，
不必导入训练模块。

## Phase 1–2：环境和业务工具

`env/simulator.py` 管理身份、账户、账单、套餐和工单；`env/session.py` 管理
消息、调用解析、预算、奖励和轨迹。`tools.py` 是统一工具 Schema 来源。
`adapters.py` 显式注册六个 BaseTool，并将独立模拟器注入每条轨迹的工具实例。

工具：`get_account_info`、`verify_identity`、`get_bill_details`、
`get_available_plans`、`change_plan`、`create_repair_ticket`。

规则：

- 基础账户信息允许查看；账单、办理资格和写操作必须先验证身份。
- 三次错误验证码后锁定；不允许模型猜测或覆盖账户归属。
- 变更套餐检查账户资格、用户确认原文和目标套餐；模型生成的“确认”不能代替用户授权。
- 报修必须使用用户提供的问题和地址；相同报修不重复建单。
- 每轮最多一个完整 Hermes `<tool_call>`；畸形、未知或并行调用均记录失败尝试。
- 普通最终回复根据实际状态和必要业务结果判断成功；中途违规不能被后来完成任务抹除。
- 步数或上下文预算耗尽在最后一个真实动作上结算失败；没有动作的初始超长样本不能伪造轨迹。

当前最终答复判据是窄范围的确定性检查（账期/金额、套餐 ID、真实工单号、拒绝词）。
它还不是完整自然语言事实核验器，也没有模拟用户澄清对话、跨账户服务或真实通信网络。
扩展这些业务时需要扩展任务 Schema、状态机、专家与评测判据。

## 模型切换

完整起始配置为 `config.toml`，支持相同结构的 JSON/YAML 覆盖文件。
配置加载会继承完整默认值，拒绝未知键。YAML 使用服务器已有的 PyYAML，
不为本地读取配置安装依赖；本地可用 TOML/JSON。

`config.example.yaml` 是可加载的模型切换覆盖示例：

```yaml
model:
  model_path: /models/Qwen3-4B
training:
  finetuning_type: lora
  lora_rank: 16
  lora_alpha: 32
  target_modules: auto
```

切换 8B 时改变 `model.model_path`；其他值只是起始值，仍须调整显存预算、
batch size、长度、TP 和并发。Python 不写死 hidden size、层数或 Qwen 投影名。

- SFT Tokenizer/模型从 `model.model_path` 加载。
- `target_modules=auto` 检查实际模型后使用 PEFT `all-linear`，排除输出头；
  显式列表先检查名称存在，再由 PEFT 验证支持的模块类型。
- `inference.model_path` 为空时跟随模型路径；可独立指向合并后的 SFT/RL 模型。
- `inference.adapter_path` 可选，验证 adapter 声明的基座与推理模型一致。
- `grpo.init_model_path` 可独立指向合并的 SFT 模型；为空时跟随模型路径。
  切换基座后不能继续使用旧模型的 Adapter 或旧架构的初始化产物。

## Phase 3：任务、专家和 SFT

以下命令创建合成演示数据，不下载外部数据，拒绝覆盖已有目标文件：

```bash
python3 -B -m recipes.telecom.data_preprocess.build_tasks \
  --output outputs/telecom/tasks.jsonl --count 40 --seed 42
python3 -B -m recipes.telecom.data_preprocess.build_expert_trajectories \
  --tasks outputs/telecom/tasks.jsonl --output outputs/telecom/experts.jsonl
python3 -B -m recipes.telecom.data_preprocess.process_telecom \
  --tasks outputs/telecom/tasks.jsonl --output-dir outputs/telecom/data
```

任务覆盖账单、套餐变更、报修、拒绝不合资格办理。划分基于 scenario ID 的稳定哈希；
导出检查重复任务、跨集合 scenario/account，并完整回放专家轨迹。
演示任务共享有限规则模板，不能据此声称覆盖真实业务分布或完成语义层面的防泄漏。
实际评测集应另外人工设计未见组合和对抗任务。

`process_telecom --parquet` 需要服务器已有 pyarrow；生成 train/validation/test Parquet。
没有 pyarrow 时仅生成 JSONL，绝不将 JSONL 文件伪装成 Parquet。

SFT 逐步训练：完整本轮 Prompt 为条件，仅监督本轮 Assistant Action 和结束 Token。
Prompt/工具反馈标签为 -100，Padding 为 -100，即使 pad_token 与 EOS 相同也不会遮掉真实 EOS。
不静默截断完成部分；模型模板不能保持生成前缀时明确失败，需要添加经过验证的格式适配。

GPU 服务器上的入口（本地未执行）：

```bash
bash examples/telecom/run_sft.sh --train-file outputs/telecom/data/train.sft.jsonl
python3 -m recipes.telecom.sft.export_model \
  --config recipes/telecom/config.toml \
  --adapter outputs/telecom/sft/final --output /models/telecom-sft-merged
```

SFT checkpoint 可通过 `--resume-from-checkpoint <checkpoint目录>` 恢复；最终输出包含
Tokenizer 和 Chat Template。Adapter 合并入口要求基座匹配，拒绝覆盖现有模型目录。
`training.finetuning_type=full` 也可使用同一 SFT 入口；此时不需要 Adapter 合并。

## Phase 4：奖励

`TelecomSession` 是业务奖励唯一写入点；工具只返回业务结果和事件。
`reward_fn.score_transition` 返回 scalar 和相同分量之和：

`task + progress + compliance - tool_cost - invalid - violation`

- `outcome_only`：仅在终局发放成功或失败奖励。
- `state_aware`：额外计一次性业务进展、合规写操作、调用成本和违规/非法调用惩罚。
- 相同事件只奖励一次；重复验证、重复查询、重复写操作不能刷进展奖励。
- 环境始终返回数值，包含 0.0，避免触发通用文本评分回退。
- `compute_score` 仅为兼容回调；缺少明确 step_reward 时失败，不凭模型文本虚构状态奖励。

当前原生 GRPO 将逐步奖励求和为轨迹总分，并广播同一个优势给轨迹内所有动作 Token。
本 recipe 没有把普通 GRPO 改造成 step-level credit assignment。
Outcome-only/State-aware 主对比改变了多个奖励项；若研究 Progress 的独立贡献，
需进一步做保持其余分量不变的权重消融。

## Phase 5：在线 GRPO

```bash
# 本地允许：只打印命令，不导入训练库、不生成运行目录。
bash examples/telecom/run_grpo_state_aware.sh
bash examples/telecom/run_grpo_outcome.sh

# GPU 服务器才执行；先将 grpo.init_model_path 设为合并 SFT 模型。
bash examples/telecom/run_grpo_state_aware.sh --run
bash examples/telecom/run_grpo_outcome.sh --run
```

可以用 `TELECOM_CONFIG=<配置路径>` 覆盖四个 shell 入口的配置文件。
两个 GRPO 入口只改变奖励模式，分别记录配置、Flow 配置、轨迹和 checkpoint。
训练/验证使用同样的环境规则。GRPO 参数映射到原训练入口，未修改核心训练器。

LoRA 策略为 `合并 SFT 基座 + 新 RL LoRA`；不直接继续挂载旧 SFT Adapter，
因此禁用 RL Adapter 的参考策略对应 SFT 初始化模型。默认未启用 KL；需要时可通过
`grpo.use_kl_loss` 打开。原生 FSDP/VERL 处理 Adapter 训练、权重同步和保存。
SFT 的 lora_dropout 可配置；VERL 0.7.0 此映射路径使用其原生 LoRA dropout 默认值。

恢复 GRPO：传 `--run --resume-from-checkpoint <VERL global_step目录>`，且运行配置必须一致。
VERL checkpoint 不是可直接传给 Transformers 的 HF 导出，也不是 Adapter-only 目录。
评测前需在 GPU 环境验证原生 checkpoint 导出，再配置 inference.model_path/adapter_path。

需要在 GPU 服务器验证：Hydra 配置合成、Parquet Dataset 字段保留、BaseTool 注册、
Token/Mask/Log Probability 对齐、rm_scores 直接消费不重复评分、LoRA-only 参数更新、
vLLM 在更新后使用新权重、参考策略语义、Checkpoint 保存恢复、4B/8B 显存预算。
这些能力当前只有源码接入与静态核对，没有声称已完成训练验证。

## Phase 6：独立评测

分别配置实际模型产物，保持任务、环境预算和解码参数一致：

```bash
bash examples/telecom/run_eval.sh --tasks outputs/telecom/data/test.tasks.jsonl \
  --label base --output outputs/telecom/eval/base.jsonl
# SFT/outcome_only/state_aware 使用各自配置和 label，不能复用同一模型冒充四组。
python3 -m recipes.telecom.evaluation.compare outputs/telecom/eval/*.metrics.json
```

指标包括任务成功率、无业务违规任务比例、非法调用率、平均调用数、每任务总输入/输出
Token、生成 Token 及截断任务数。总输入 Token 包含每轮重新输入的历史，不只计原题。
初始 Prompt 超长在评测中记作失败，不从分母删除；训练中明确拒绝无动作的超长样本。
比较器检查任务内容哈希、任务顺序及预算。模型加载异常属于运行错误，不伪装为评测结果。

当前没有真实模型指标；尚未增加置信区间、人工答复质量判定、多随机种子汇总和完整统计报告。

## 验证状态与来源

本地：标准库业务/回放/配置/Mask/指标测试、Python 源码编译、shell 语法、GRPO dry-run。
GPU：模型模板实际编码、SFT/LoRA训练和导出、VERL/vLLM 集成、模型评测全部待验证。

源码核对来源（只读查看，没有下载或安装其代码）：

- [VERL v0.7.0 模型配置](https://github.com/volcengine/verl/blob/v0.7.0/verl/workers/config/model.py)
- [VERL v0.7.0 FSDP Worker](https://github.com/volcengine/verl/blob/v0.7.0/verl/workers/fsdp_workers.py)
- [VERL v0.7.0 checkpoint 配置](https://github.com/volcengine/verl/blob/v0.7.0/verl/trainer/config/ppo_trainer.yaml)
- [PEFT LoRA 文档](https://huggingface.co/docs/peft/developer_guides/lora)
- [Transformers Trainer 文档](https://huggingface.co/docs/transformers/main_classes/trainer)
