# D-OPSD / OPSD-NFT / Krea2 LoRA

Krea 2 多参考图 LoRA、D-OPSD 蒸馏和 OPSD-NFT 奖励后训练的代码与实验记录归档。

## 效果对比网页

[`docs/comparison/index.html`](docs/comparison/index.html) 提供实验链路、公开同条件视觉代理、社区 D-OPSD 样例与本地训练日志曲线的交互式展示。

在线地址：<https://lmh9524.github.io/D-OPSD-OPSD-NFT-Krea2-LoRA/>

```bash
python -m http.server 8000
# 浏览器打开 http://localhost:8000/docs/comparison/
```

网页严格区分本项目训练证据与外部视觉参考；由于原始 preview 图片未进入归档，外部图片不会被标记为本项目输出。

## 内容

| 阶段 | 训练入口 | 配置 | 运行记录 |
| --- | --- | --- | --- |
| Phase 1：多参考图 LoRA | [`experiments/krea2_ref2img_lora.py`](experiments/krea2_ref2img_lora.py) | [`dflow/config/recipe/krea2_ref2img_lora.py`](dflow/config/recipe/krea2_ref2img_lora.py) | 正式运行的原始 `config.json`、日志和 provenance 未归档 |
| D-OPSD（2000 step） | [`experiments/dopsd_krea2_ref2img_lora.py`](experiments/dopsd_krea2_ref2img_lora.py) | [`dflow/config/recipe/krea2_dopsd_lora.py`](dflow/config/recipe/krea2_dopsd_lora.py) | [`records/dopsd-2000/`](records/dopsd-2000/) |
| OPSD-NFT（500 step） | [`experiments/krea2_ref2img_opsd_nft.py`](experiments/krea2_ref2img_opsd_nft.py) | [`dflow/config/recipe/krea2_opsd_nft_lora.py`](dflow/config/recipe/krea2_opsd_nft_lora.py) | [`records/opsd-nft-500/`](records/opsd-nft-500/) |

仓库同时包含运行这些入口所需的 `dflow` 源码、相关测试和 Krea 2 评估工具。

## 已确认的实验参数

### Phase 1：Krea2 多参考图 LoRA

- 训练基座：`krea/Krea-2-Raw`；推理基座：Krea 2 Turbo
- LoRA rank/alpha：64/64，约 229M 可训练参数
- 正式实验：2000 steps，batch size 1，学习率 `1e-4`，constant scheduler，无 warmup
- 目标分辨率：576 x 1008；9 张 reference；caption dropout 0
- disjoint reference registration；Qwen3-VL 对全部参考图进行 grounding
- flow shift `3.1582`（对应 `mu=1.15`）
- 单卡 H100，约 69.8 GiB，约 5.88 s/step

注意：代码默认值为 8000 steps、4 references、caption dropout 0.05、warmup 500。正式实验应当使用过命令行覆盖，但该次运行的最终配置与日志没有进入归档，因此 Phase 1 只能高置信重建，不能做到 bit-for-bit 复现。

### D-OPSD：2000 step

- Krea 2 Turbo，BF16；Phase-1 LoRA 热启动，rank/alpha 64/64
- 9 references，disjoint registration；target 768²，reference 384²
- 4-step on-policy rollout；student + EMA teacher（0.9999）
- teacher 额外接收 target latent；x0 loss；逐 step backward（`low_mem=true`）
- AdamW，学习率 `1e-5`，weight decay 0.01，warmup 50
- batch size 1，grad accumulation 1，seed 42
- activation checkpointing 与 FSDP 开启，compile 关闭
- 2000 steps，每 400 step 保存；日志记录完成于 step 2000

### OPSD-NFT：500 step

- group size 4；4 个训练时间步
- CLIP-I reference-fidelity reward
- `mix_beta=0.1`，`ref_kl_coef=1e-4`
- old-policy decay 0，每步同步
- AdamW，学习率 `1e-5`
- 500 steps，grad accumulation 2，seed 42，每 50 step 保存
- 日志记录完成于 step 500

启动脚本的默认 group size、时间步数、grad accumulation 和保存间隔与正式运行不同；审计或复现时以 `records/opsd-nft-500/config.json` 为准。

## 安装

需要 Python 3.12、CUDA/PyTorch 环境以及足够显存。依赖定义在 [`pyproject.toml`](pyproject.toml)：

```bash
python -m pip install -e .
hf auth login
```

Krea 2 的多参考图条件设计见 [`docs/krea2-reference.md`](docs/krea2-reference.md)。

## 数据格式与启动

数据目录需要包含 `train.jsonl` 及其引用的图片。每一行的核心结构如下：

```json
{"target": "target.png", "refs": ["ref-1.png", "ref-2.png"], "prompt": "..."}
```

Phase 1：

```bash
./scripts/krea2_ref2img_lora.sh /path/to/dataset [额外 tyro 参数]
```

OPSD-NFT：

```bash
CKPS=/path/to/checkpoints \
./scripts/train_opsd_nft_krea2.sh /path/to/dataset [额外 tyro 参数]
```

D-OPSD 可直接调用入口，并用 tyro 参数覆盖数据、模型、初始 LoRA 与输出目录：

```bash
python experiments/dopsd_krea2_ref2img_lora.py \
  --dataset.root /path/to/dataset \
  --backbone.model.path /path/to/Krea-2-Turbo \
  --distill.init-lora-from /path/to/phase-1-lora \
  --training.steps 2000
```

## 归档边界

- `records/` 保留两次正式运行的最终配置、训练日志和代码/环境 provenance。
- provenance 显示运行时工作区为 dirty；未提交 diff 已内嵌在相应 `provenance.json` 中。
- 训练数据、模型权重、LoRA checkpoint 和 demo 样本不在本仓库中，也不应被提交。
- 路径类参数记录的是原始运行环境，需要在新环境中覆盖。

本归档未附带开源许可证；除非权利人另行授权，公开可见不等于授予复制、修改或再分发许可。
