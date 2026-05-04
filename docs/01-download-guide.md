# 服务器下载指引

这份指引只负责“把模型下载到指定位置”，不会替你在服务器上直接执行。你的硬约束是系统盘空间紧，所有模型、缓存、临时文件都要落到 `/home/dataset-local/data/zos_download/model_support`。

先说结论：

- 如果你只是要求“不占系统盘”，那么使用 `/home/dataset-local/data/zos_download/conda/envs/model_support` 是可以的，因为它本身就在数据盘。
- 如果你要求“所有内容都必须在 `/home/dataset-local/data/zos_download/model_support` 项目目录里”，那么只靠这个 conda 环境不够，因为后续 `pip install` 的包会写进 `/home/dataset-local/data/zos_download/conda/envs/model_support/lib/...`。

## 1. 先切到项目根目录

```bash
cd /home/dataset-local/data/zos_download/model_support
conda activate model_support
```

## 2. 先把缓存和临时目录全部压到 `/data`

推荐先执行：

```bash
source configs/server.env.example
mkdir -p \
  /home/dataset-local/data/zos_download/model_support/.home \
  /home/dataset-local/data/zos_download/model_support/models/hf \
  /home/dataset-local/data/zos_download/model_support/models/mop \
  /home/dataset-local/data/zos_download/model_support/.cache \
  /home/dataset-local/data/zos_download/model_support/.config \
  /home/dataset-local/data/zos_download/model_support/.local/share \
  /home/dataset-local/data/zos_download/model_support/.tmp
```

这样做的目的有三个：

- `HF_HOME`、`PIP_CACHE_DIR`、`TMPDIR` 不再碰系统盘。
- vLLM 自己的缓存和配置目录也被压到 `/home/dataset-local/data/zos_download/model_support`。
- 后面不管你是用 Hugging Face、ModelScope、uv、PyTorch、Triton 还是 CUDA JIT，默认写盘点都尽量被压回项目目录。

## 3. 两种安装模式里你该选哪一个

### 模式 A：只要求不占系统盘

继续用你已有的 conda 环境：

```bash
conda activate model_support
source configs/server.env.example
python scripts/check_storage_paths.py --allow-prefix /home/dataset-local/data/zos_download/conda/envs/model_support
```

这里 `--allow-prefix` 的意思是：允许 Python 解释器和 `site-packages` 继续落在 `/home/dataset-local/data/zos_download/conda/envs/model_support`，但缓存和临时目录仍然必须在 `/home/dataset-local/data/zos_download/model_support`。

### 模式 B：要求绝对只在项目目录里

这种情况下不要把包装进 conda 环境，而是在项目里再建一个 `.venv`：

```bash
conda activate model_support
source configs/server.env.example
python -m venv /home/dataset-local/data/zos_download/model_support/.venv
source /home/dataset-local/data/zos_download/model_support/.venv/bin/activate
python scripts/check_storage_paths.py
```

`check_storage_paths.py` 会检查：

- `HOME`
- `HF_HOME`
- `MODELSCOPE_CACHE`
- `UV_CACHE_DIR`
- `PIP_CACHE_DIR`
- `TMPDIR`
- `TORCH_EXTENSIONS_DIR`
- `TORCHINDUCTOR_CACHE_DIR`
- `TRITON_CACHE_DIR`
- `CUDA_CACHE_PATH`
- 当前 Python 的 `sys.prefix`
- 当前 `site-packages`

只要它报 `OUTSIDE`，就说明还有内容会落到允许范围外面。

## 4. 安装下载脚本所需依赖

```bash
python -m pip install -U pip
python -m pip install -r requirements/requirements-server.txt
```

## 5. 先看有哪些模型别名

```bash
python scripts/render_commands.py --list
```

如果你想看某一个模型的推荐命令：

```bash
python scripts/render_commands.py --model qwen36_35b_a3b
```

## 6. 从 Hugging Face 下载

先 dry-run：

```bash
python scripts/download_model.py --model qwen36_35b_a3b --source hf --dry-run
```

确认路径没问题后正式下载：

```bash
python scripts/download_model.py --model qwen36_35b_a3b --source hf
```

脚本会把模型直接下载到：

```text
/home/dataset-local/data/zos_download/model_support/models/hf/<local_dir_name>
```

例如：

```text
/home/dataset-local/data/zos_download/model_support/models/hf/Qwen3.6-35B-A3B
```

如果某个仓库需要 Hugging Face Token，先导出：

```bash
export HF_TOKEN=你的token
```

## 7. 从 ModelScope 下载

你的目录里已经预留了 `models/mop`，所以我也把脚本兼容上了。但有一个现实情况要先说明：

- 这些模型在 ModelScope 上的镜像名称是否完整、是否和 Hugging Face 同名，会变化。
- 我这里没有在你的服务器上直接验证每个镜像名，所以更稳妥的方式是先在 ModelScope 页面确认仓库名，再把仓库名传给脚本。

先只生成命令：

```bash
python scripts/download_model.py \
  --model qwen36_35b_a3b \
  --source mop \
  --command-only
```

如果你确认这个镜像名没问题，再正式执行：

```bash
python scripts/download_model.py \
  --model qwen36_35b_a3b \
  --source mop
```

目标目录会是：

```text
/home/dataset-local/data/zos_download/model_support/models/mop/Qwen3.6-35B-A3B
```

## 8. 我建议你实际下载时按这个顺序来

先从小到大验证链路，避免一上来就下 200GB 以上的大模型：

1. `gui_owl_7b_desktop_rl`
2. `ui_tars_15_7b`
3. `gui_owl_15_32b`
4. `glm_46v`
5. `qwen35_35b_a3b`
6. `qwen36_35b_a3b`
7. `qwen35_122b_a10b`
8. `qwen3_vl_235b_a22b`

## 9. 下载完成后的检查

每下完一个模型都建议检查一次目录：

```bash
du -sh /home/dataset-local/data/zos_download/model_support/models/hf/*
find /home/dataset-local/data/zos_download/model_support/models/hf/Qwen3.6-35B-A3B -maxdepth 1 | head
```

重点确认：

- 文件确实在 `/home/dataset-local/data/zos_download/model_support/models/...` 下。
- 不是跑到了根目录 overlay 或用户家目录缓存。
- `config.json`、`tokenizer_config.json`、`model.safetensors.index.json` 这类核心文件都在。

## 10. 额外提醒

- `Qwen3-VL-235B-A22B-Instruct`、`Qwen3.5-122B-A10B`、`Qwen3.6-35B-A3B`、`GLM-4.6V` 都属于大模型或超大模型。下载时间长、文件多，最好单模型串行下载。
- `UI-TARS-1.5-7B` 和 `GUI-Owl-7B-Desktop-RL` 的权重页显示张量类型是 `F32`，单文件体积会比常见的 BF16 版本更夸张，别被目录大小吓到。
- 如果你使用 `hf download --local-dir ...` 这类 Hugging Face 官方命令，官方文档说明会在目标目录下生成一个 `.cache/huggingface/` 元数据目录。这是正常现象，不是“下错地方”。
