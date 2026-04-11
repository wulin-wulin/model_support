# 部署、启动和测试

这部分默认你已经把模型文件下载到了 `/data/model_support/models/hf` 或 `/data/model_support/models/mop`，并且已经把这份代码同步到了服务器。

## 1. 进入环境

```bash
cd /data/model_support
conda activate chartmodel_wulin
source configs/server.env.example
```

这里先把一句关键话说清楚：

- 用 `chartmodel_wulin` 直接安装依赖，不会占系统盘，因为这个环境本身在 `/data/conda/envs/...`。
- 但如果你要求“所有字节都只能写在 `/data/model_support` 项目目录里”，那就应该继续执行下面的“严格模式”。

### 严格模式：项目内虚拟环境

```bash
python -m venv /data/model_support/.venv
source /data/model_support/.venv/bin/activate
python scripts/check_storage_paths.py
```

如果你只要求“不碰系统盘”，可以继续用 conda 环境并这样检查：

```bash
python scripts/check_storage_paths.py --allow-prefix /data/conda/envs/chartmodel_wulin
```

## 2. 先装辅助依赖

```bash
python -m pip install -U pip
python -m pip install -r requirements/requirements-server.txt
```

## 3. 安装 vLLM

这里要分两层理解：

- vLLM 官方安装文档推荐 Linux 新环境，并明确说 vLLM 原生不支持 Windows。
- Qwen3.5 官方模型卡又明确写了：对 Qwen3.5，建议使用 vLLM 主干分支对应的 nightly。

所以这台服务器更稳妥的做法是：

```bash
python -m pip install -U uv
uv pip install vllm --torch-backend=auto --extra-index-url https://wheels.vllm.ai/nightly
```

如果你不想上 nightly，也可以先试稳定版；但只要碰到 Qwen3.5 / 新版 GLM / 新版 GUI 模型启动异常，我会优先怀疑 vLLM 版本过旧。

安装完以后建议再跑一遍：

```bash
python scripts/check_storage_paths.py
```

如果你是继续使用 conda 环境，则改成：

```bash
python scripts/check_storage_paths.py --allow-prefix /data/conda/envs/chartmodel_wulin
```

## 4. 启动前先检查命令

先只打印，不运行：

```bash
python scripts/start_vllm.py --model qwen35_35b_a3b --source hf --print-only
```

如果模型是从 ModelScope 下载的：

```bash
python scripts/start_vllm.py --model qwen35_35b_a3b --source mop --print-only
```

## 5. 正式启动服务

例如启动 `Qwen3.5-35B-A3B`：

```bash
python scripts/start_vllm.py --model qwen35_35b_a3b --source hf --run
```

例如启动 `Qwen3-VL-235B-A22B-Instruct`：

```bash
python scripts/start_vllm.py --model qwen3_vl_235b_a22b --source hf --run
```

脚本会自动带上：

- `--served-model-name`
- `--tensor-parallel-size`
- `--max-model-len`
- `--gpu-memory-utilization`
- `--max-num-seqs`
- `--generation-config vllm`
- 多模态模型需要的 `--allowed-local-media-path`

## 6. 文本 smoke test

服务起来后，先看模型列表：

```bash
curl http://127.0.0.1:8006/v1/models
```

再做一次最小文本测试：

```bash
python scripts/smoke_test_openai.py \
  --base-url http://127.0.0.1:8006/v1 \
  --model qwen3.5-35b-a3b \
  --prompt "请用三句话介绍你自己的能力边界。"
```

如果你设置了接口鉴权：

```bash
export VLLM_API_KEY=你的key
python scripts/smoke_test_openai.py \
  --base-url http://127.0.0.1:8006/v1 \
  --api-key 你的key \
  --model qwen3.5-35b-a3b \
  --prompt "你好"
```

## 7. 多模态 smoke test

远程图片 URL：

```bash
python scripts/smoke_test_openai.py \
  --base-url http://127.0.0.1:8004/v1 \
  --model qwen3-vl-235b-a22b-instruct \
  --image-url https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg \
  --prompt "请描述图像内容，并指出最显眼的主体。"
```

本地图片路径：

```bash
python scripts/smoke_test_openai.py \
  --base-url http://127.0.0.1:8001/v1 \
  --model gui-owl-1.5-32b-instruct \
  --image-path /data/model_support/test_assets/example.png \
  --prompt "这是一个什么界面？请指出主要按钮。"
```

注意：如果你要直接传本地文件，启动命令里必须保留 `--allowed-local-media-path /` 或者缩小到你自己的测试目录。

## 8. 20 并发验证

先做一版轻量并发压测：

```bash
python scripts/load_test_openai.py \
  --base-url http://127.0.0.1:8006/v1 \
  --model qwen3.5-35b-a3b \
  --concurrency 20 \
  --requests 60
```

如果你想用 vLLM 官方基准工具验证请求速率，也可以：

```bash
vllm bench serve \
  --backend openai-chat \
  --endpoint /v1/chat/completions \
  --model qwen3.5-35b-a3b \
  --dataset-name random \
  --random-input-len 2048 \
  --random-output-len 512 \
  --num-prompts 1000 \
  --request-rate 20
```

这里要特别说明一下：

- “支持 20 并发”不是一个脱离上下文的绝对值。
- 同样是 20 个请求，短文本、长上下文、多图输入、视频输入，对显存和吞吐的压力差异非常大。
- 所以这份工程默认把大模型的 `max_model_len` 调得比理论上限更保守，目的是先把 20 并发跑稳，再逐步放大上下文。

## 9. 什么时候需要调参

如果你遇到 OOM 或吞吐明显不够，优先按这个顺序调：

1. 先降低 `--max-model-len`
2. 再降低单请求图片数量或图片像素上限
3. 再降低 `--max-num-seqs`
4. 再考虑把 TP 从 `4` 提到 `8`
5. 对 Qwen3.5 文本场景，考虑加 `--language-model-only`

例如 Qwen3.5-35B-A3B 文本专用模式：

```bash
python scripts/start_vllm.py \
  --model qwen35_35b_a3b \
  --source hf \
  --language-model-only \
  --run
```

## 10. 关于“能不能把七个服务同时挂着”

不建议直接这样规划。

- `Qwen3-VL-235B-A22B-Instruct`
- `Qwen3.5-122B-A10B`
- `GLM-4.6V`

这几个都属于超大模型级别，虽然你的 8xA100-80GB 很强，但它们更适合“一次启动一个大服务”或者“大模型配合一个小模型共存”，而不是七个模型一起常驻。

更实际的策略是：

1. 7B / 8B 这类 GUI 模型单卡或双卡常驻。
2. 35B 模型按业务负载做主服务。
3. 122B / 235B / 106B 级别模型按需启动，或者单独做一类专用服务。
