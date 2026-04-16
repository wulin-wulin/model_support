# model_support

这套目录是给你的 Linux 服务器准备的离线脚手架：在本地把下载指引、模型清单、vLLM 启动命令、OpenAI 兼容测试脚本和 20 并发压测脚本都整理好，之后你把代码同步到服务器即可。

## 你会用到的文件

- `configs/models.yaml`: 目标模型清单、默认端口、推荐的 vLLM 参数。
- `configs/server.env.example`: 统一把缓存、临时文件、配置目录都压到 `/home/dataset-local/data/zos_download/model_support`。
- `scripts/download_model.py`: 下载模型到 `models/hf` 或 `models/mop`。
- `scripts/start_vllm.py`: 按模型别名拼出并执行 `vllm serve` 命令。
- `scripts/smoke_test_openai.py`: 用 OpenAI SDK 打文本或多模态 smoke test。
- `scripts/load_test_openai.py`: 简单并发压测，默认适合验证 `20` 并发目标。
- `scripts/local_tiny_model_smoke.py`: 在本地机器上下载一个很小的模型并做最小可行验证。
- `docs/01-download-guide.md`: 服务器上如何把模型下载到指定位置。
- `docs/02-deploy-and-test.md`: 服务器上如何安装 vLLM、启动服务、人工测试和压测。

## 推荐使用顺序

1. 先看 `docs/01-download-guide.md`，在服务器上把模型拉到 `/home/dataset-local/data/zos_download/model_support/models/...`。
2. 把本目录同步到服务器后，按 `docs/02-deploy-and-test.md` 安装依赖。
3. 先跑 `python scripts/check_storage_paths.py` 检查当前 shell 的写路径是否都在允许范围内。
4. 用 `python scripts/render_commands.py --list` 看所有模型的别名、端口和推荐参数。
5. 用 `python scripts/start_vllm.py --model <alias> --source hf --print-only` 先检查命令。
6. 确认没问题后，加 `--run` 启动服务。
7. 用 `scripts/smoke_test_openai.py` 和 `scripts/load_test_openai.py` 做人工验证和并发验证。

## 存储隔离结论

如果你的要求只是“不占系统盘”，现有 `/home/dataset-local/data/zos_download/conda/envs/model_support` 可以继续用，因为它在数据盘。

如果你的要求是“所有安装、缓存、临时文件都必须在 `/home/dataset-local/data/zos_download/model_support` 项目目录里”，那就不要把依赖装进 `model_support` 这个 conda 环境，因为 `site-packages` 会写到 `/home/dataset-local/data/zos_download/conda/envs/model_support/...`，这虽然不在系统盘，但也不在项目目录里。

这种严格模式下，推荐在服务器上改用项目内虚拟环境：

```bash
cd /home/dataset-local/data/zos_download/model_support
conda activate model_support
source configs/server.env.example
python -m venv /home/dataset-local/data/zos_download/model_support/.venv
source /home/dataset-local/data/zos_download/model_support/.venv/bin/activate
python scripts/check_storage_paths.py
```

## 本地验证说明

本地这台机器是 Windows，并且显卡是 `GeForce MX350 2GB`。vLLM 官方文档要求原生运行环境为 Linux，且 NVIDIA GPU 计算能力至少为 7.0；因此这里不直接跑 vLLM，而是用一个很小的 `transformers` 模型做本地 smoke test，验证下载、权重加载、分词和生成链路都通。

## 快速命令

列出模型清单：

```powershell
python scripts/render_commands.py --list
```

查看某个模型的下载和启动建议：

```powershell
python scripts/render_commands.py --model qwen35_35b_a3b
```

本地小模型验证：

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements/requirements-local.txt
python scripts/local_tiny_model_smoke.py
```
