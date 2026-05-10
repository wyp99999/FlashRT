# FlashRT SM89 API 使用示例

> **版本**: Session 67
> **更新时间**: 2026-05-06
> **适用硬件**: RTX 4060 Ti (SM89)

---

## 🚀 快速开始

### GROOT N1.7 示例 (推荐)

```python
"""GROOT N1.7 快速推理示例"""
import os
import sys
import numpy as np
import time

# 设置环境
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
sys.path.insert(0, '/data/FlashRT')

import torch
from flash_rt.frontends.torch.groot_n17_sm89 import GrootN17TorchFrontendSm89

def main():
    # 1. 初始化模型 (最优配置: 38ms)
    print("加载GROOT N1.7模型...")
    pipe = GrootN17TorchFrontendSm89(
        '/data/models/groot-n1.7',
        num_views=1,       # 单视角
        num_flow_steps=2   # 2步流匹配
    )
    
    # 2. 设置任务提示
    print("设置任务提示...")
    pipe.set_prompt('pick up the red block')
    
    # 3. 构建推理管线 (包含CUDA Graph捕获)
    print("构建推理管线...")
    pipe.build_pipeline()
    
    # 4. 准备输入图像
    # 实际使用时替换为真实摄像头图像
    image = np.zeros((224, 224, 3), dtype=np.uint8)  # 黑色图像示例
    
    # 5. Warmup
    print("预热...")
    for _ in range(10):
        pipe.infer({'image': image})
    torch.cuda.synchronize()
    
    # 6. 性能测试
    print("性能测试...")
    t0 = time.time()
    for _ in range(50):
        result = pipe.infer({'image': image})
    torch.cuda.synchronize()
    
    latency = (time.time() - t0) / 50 * 1000
    print(f"平均延迟: {latency:.2f}ms")
    print(f"吞吐量: {1000/latency:.1f} req/s")
    
    # 7. 获取输出
    actions = result['actions']  # shape: (10, 7)
    print(f"输出动作序列: {actions.shape}")
    print(f"第一个动作: {actions[0]}")

if __name__ == '__main__':
    main()
```

---

## 🤖 Pi0.5 示例

```python
"""Pi0.5 快速推理示例"""
import os
import sys
import numpy as np
import time

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
sys.path.insert(0, '/data/FlashRT')

import torch
from flash_rt.frontends.torch.pi05_sm89 import Pi05TorchFrontendSm89

def main():
    # 1. 初始化模型 (最优配置: ~59ms)
    print("加载Pi0.5模型...")
    pipe = Pi05TorchFrontendSm89(
        '/data/models/pi05_base',
        num_views=1,   # 单视角 (必须!)
        num_steps=1    # 1步扩散 (必须!)
    )
    
    # 2. 设置提示
    pipe.set_prompt('pick up')
    
    # 3. 构建管线
    pipe.build_pipeline()
    
    # 4. 准备输入
    obs = {
        'image': np.zeros((224, 224, 3), dtype=np.uint8),
        'wrist_image': np.zeros((224, 224, 3), dtype=np.uint8),
        'state': np.zeros((7,), dtype=np.float32)  # 机器人状态
    }
    
    # 5. Warmup
    for _ in range(10):
        pipe.infer(obs)
    torch.cuda.synchronize()
    
    # 6. 性能测试
    t0 = time.time()
    for _ in range(50):
        result = pipe.infer(obs)
    torch.cuda.synchronize()
    
    latency = (time.time() - t0) / 50 * 1000
    print(f"平均延迟: {latency:.2f}ms")
    
    # 注意: 默认参数(num_views=2, num_steps=10)会~138ms

if __name__ == '__main__':
    main()
```

---

## 💬 Qwen2.5 示例

```python
"""Qwen2.5-0.5B 文本生成示例"""
import os
import sys
import time

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
sys.path.insert(0, '/data/FlashRT')

import torch
from flash_rt.frontends.torch.qwen25_sm89 import Qwen25TorchFrontendSm89

def main():
    # 1. 初始化模型
    print("加载Qwen2.5-0.5B模型...")
    pipe = Qwen25TorchFrontendSm89(
        '/data/models/qwen2.5-0.5b',
        max_new_tokens=10  # 生成10个token
    )
    
    # 2. 设置提示
    pipe.set_prompt('Hello, how are you today?')
    
    # 3. 构建管线
    pipe.build_pipeline()
    
    # 4. Warmup
    for _ in range(5):
        pipe.infer({})
    torch.cuda.synchronize()
    
    # 5. 性能测试
    t0 = time.time()
    for _ in range(30):
        result = pipe.infer({})
    torch.cuda.synchronize()
    
    latency = (time.time() - t0) / 30 * 1000
    print(f"平均延迟: {latency:.2f}ms/10tok")
    print(f"吞吐量: {10000/latency:.1f} tokens/s")
    
    # 6. 获取生成文本
    print(f"生成文本: '{result['generated_text']}'")

if __name__ == '__main__':
    main()
```

---

## 🌐 HTTP API 示例

### 启动服务

```bash
cd /data && python3 -m flash_rt.services.async_api_server \
    --model groot_n17 \
    --port 8080 \
    --checkpoint /data/models/groot-n1.7 \
    --num_views 1 \
    --num_flow_steps 2
```

### Python客户端

```python
"""HTTP API 客户端示例"""
import requests
import base64
import numpy as np
from PIL import Image
import io

# 服务地址
API_URL = "http://localhost:8080"

def encode_image(image: np.ndarray) -> str:
    """将numpy图像编码为base64"""
    pil_img = Image.fromarray(image)
    buffer = io.BytesIO()
    pil_img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()

def infer(prompt: str, image: np.ndarray):
    """发送推理请求"""
    response = requests.post(
        f"{API_URL}/infer",
        json={
            "prompt": prompt,
            "image_b64": encode_image(image)
        }
    )
    return response.json()

def batch_infer(requests_list):
    """批量推理"""
    payload = {
        "requests": [
            {"prompt": r["prompt"], "image_b64": encode_image(r["image"])}
            for r in requests_list
        ]
    }
    response = requests.post(f"{API_URL}/batch_infer", json=payload)
    return response.json()

# 使用示例
image = np.zeros((224, 224, 3), dtype=np.uint8)

# 单次推理
result = infer("pick up the red block", image)
print(f"动作: {result['actions']}")
print(f"延迟: {result['latency_ms']}ms")

# 批量推理
batch_result = batch_infer([
    {"prompt": "pick up", "image": image},
    {"prompt": "place down", "image": image}
])
print(f"批量结果: {len(batch_result['results'])}个")

# 健康检查
health = requests.get(f"{API_URL}/health").json()
print(f"服务状态: {health['status']}")
```

---

## 🔌 gRPC 示例

### 启动服务

```bash
cd /data && python3 -m flash_rt.services.grpc_server \
    --model groot_n17 \
    --port 50051 \
    --checkpoint /data/models/groot-n1.7
```

### Python客户端

```python
"""gRPC 客户端示例"""
import grpc
import numpy as np
import base64

# 导入生成的proto模块
sys.path.insert(0, '/data/FlashRT')
from flash_rt.services import flashrt_pb2, flashrt_pb2_grpc

def main():
    # 连接服务
    channel = grpc.insecure_channel('localhost:50051')
    stub = flashrt_pb2_grpc.FlashRTStub(channel)
    
    # 健康检查
    health = stub.Health(flashrt_pb2.HealthRequest())
    print(f"服务状态: {health.status}")
    
    # 推理请求
    image = np.zeros((224, 224, 3), dtype=np.uint8)
    image_b64 = base64.b64encode(image.tobytes()).decode()
    
    request = flashrt_pb2.InferRequest(
        prompt="pick up the red block",
        image_b64=image_b64
    )
    
    response = stub.Infer(request)
    print(f"动作: {list(response.actions)}")
    print(f"延迟: {response.latency_ms}ms")
    
    channel.close()

if __name__ == '__main__':
    main()
```

---

## 📊 性能对比

### 配置参数影响

| 模型 | 参数 | 默认延迟 | 优化延迟 |
|------|------|----------|----------|
| GROOT | num_views=1, num_flow_steps=2 | ~50ms | **38ms** |
| Pi0.5 | num_views=1, num_steps=1 | ~138ms | **~59ms** |

### 吞吐量测试

```python
"""吞吐量测试示例"""
import threading
import time

def throughput_test(pipe, image, num_threads=4, requests_per_thread=25):
    """多线程吞吐量测试"""
    
    def worker(results):
        for _ in range(requests_per_thread):
            pipe.infer({'image': image})
        results.append(requests_per_thread)
    
    # Warmup
    for _ in range(10):
        pipe.infer({'image': image})
    torch.cuda.synchronize()
    
    # 测试
    results = []
    threads = [
        threading.Thread(target=worker, args=(results,))
        for _ in range(num_threads)
    ]
    
    t0 = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    torch.cuda.synchronize()
    
    total = sum(results)
    elapsed = time.time() - t0
    throughput = total / elapsed
    
    print(f"总请求: {total}")
    print(f"耗时: {elapsed:.2f}s")
    print(f"吞吐量: {throughput:.1f} req/s")
```

---

## ⚠️ 注意事项

1. **内存管理**: 不同模型会占用不同GPU内存，无法同时加载多个模型
2. **CUDA Graph**: 首次推理会捕获CUDA Graph，后续推理更快
3. **硬件限制**: RTX 4060 Ti不支持FP8，Pi0.5无法<50ms
4. **精度保证**: MSE=0, Cosine=1，完美精度