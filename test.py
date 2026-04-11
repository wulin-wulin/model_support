import requests
import json

base_url = "http://127.0.0.1:8006/v1"
api_key = "EMPTY"   # 如果你设置了 VLLM_API_KEY，就改成真实 key
# model = "qwen3.5-35b-a3b"   # 改成你服务实际 exposed 的 model 名
# model = "gui-owl-1.5-32b-instruct"   # 改成你服务实际 exposed 的 model 名
model = "qwen3.5-122b-a10b"

payload = {
    "model": model,
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "你好，请用三句话介绍你自己。"}
    ],
    "max_tokens": 1500,
    "temperature": 0.2
}

resp = requests.post(
    f"{base_url}/chat/completions",
    headers={
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    },
    json=payload,
    timeout=300,
)

print(resp.status_code)
print(json.dumps(resp.json(), ensure_ascii=False, indent=2))
