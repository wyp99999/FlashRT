# FlashRT SM89 生产部署指南

> **版本**: Session 67
> **更新时间**: 2026-05-06
> **适用硬件**: RTX 4060 Ti (SM89)

---

## 📊 性能概览

| 模型 | 性能 | 状态 |
|------|------|:----:|
| GROOT N1.7 | 38ms | ✅ 达标 (<50ms) |
| Pi0.5 | 59ms | ⚠️ 接近 (SM89限制) |
| Qwen2.5 0.5B | 165ms/10tok | ✅ 支持 |

---

## 🚀 快速部署

### 1. 环境准备

```bash
# 安装依赖
pip install torch numpy transformers safetensors -i https://mirrors.aliyun.com/pypi/simple/

# 安装FlashRT
cd /data/FlashRT
pip install -e ".[torch]"
```

### 2. 模型准备

```bash
# 模型路径
/data/models/groot-n1.7/      # GROOT N1.7
/data/models/pi05_base/       # Pi0.5
/data/models/qwen2.5-0.5b/    # Qwen2.5 0.5B
```

### 3. 启动API服务

#### HTTP REST API

```bash
cd /data && python3 -m flash_rt.services.async_api_server \
    --model groot_n17 \
    --port 8080 \
    --checkpoint /data/models/groot-n1.7 \
    --num_views 1 \
    --num_flow_steps 2
```

#### gRPC服务

```bash
cd /data && python3 -m flash_rt.services.grpc_server \
    --model groot_n17 \
    --port 50051 \
    --checkpoint /data/models/groot-n1.7
```

---

## 📝 API使用示例

### GROOT推理

```python
import os, sys
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
sys.path.insert(0, '/data/FlashRT')

import numpy as np
from flash_rt.frontends.torch.groot_n17_sm89 import GrootN17TorchFrontendSm89

# 初始化
pipe = GrootN17TorchFrontendSm89(
    '/data/models/groot-n1.7',
    num_views=1,
    num_flow_steps=2
)

# 设置prompt并构建
pipe.set_prompt('pick up the red block')
pipe.build_pipeline()

# 准备图像 (224x224 RGB)
image = np.zeros((224, 224, 3), dtype=np.uint8)

# 推理
result = pipe.infer({'image': image})
actions = result['actions']  # shape: (10, 7)
```

### Pi0.5推理

```python
from flash_rt.frontends.torch.pi05_sm89 import Pi05TorchFrontendSm89

pipe = Pi05TorchFrontendSm89(
    '/data/models/pi05_base',
    num_views=1,    # 单视角
    num_steps=1     # 1步扩散
)

pipe.set_prompt('pick up')
pipe.build_pipeline()

obs = {
    'image': np.zeros((224, 224, 3), dtype=np.uint8),
    'wrist_image': np.zeros((224, 224, 3), dtype=np.uint8),
    'state': np.zeros((7,), dtype=np.float32)
}

result = pipe.infer(obs)
actions = result['actions']
```

### Qwen2.5推理

```python
from flash_rt.frontends.torch.qwen25_sm89 import Qwen25TorchFrontendSm89

pipe = Qwen25TorchFrontendSm89(
    '/data/models/qwen2.5-0.5b',
    max_new_tokens=10
)

pipe.set_prompt('Hello, how are you?')
pipe.build_pipeline()

result = pipe.infer({})
print(result['generated_text'])
```

---

## 🌐 HTTP API接口

### 推理请求

```bash
curl -X POST http://localhost:8080/infer \
    -H "Content-Type: application/json" \
    -d '{
        "prompt": "pick up the red block",
        "image_b64": "<base64_encoded_image>"
    }'
```

### Batch推理

```bash
curl -X POST http://localhost:8080/batch_infer \
    -H "Content-Type: application/json" \
    -d '{
        "requests": [
            {"prompt": "pick up", "image_b64": "<img1>"},
            {"prompt": "place down", "image_b64": "<img2>"}
        ]
    }'
```

### 健康检查

```bash
curl http://localhost:8080/health
```

---

## 🔧 性能调优

### GROOT最优配置

```python
# 最快配置 (38ms)
GrootN17TorchFrontendSm89(
    checkpoint_dir,
    num_views=1,       # 单视角，减少视觉编码开销
    num_flow_steps=2   # 2步流匹配，平衡质量和速度
)
```

### Pi0.5最优配置

```python
# 最快配置 (~59ms，SM89硬件限制)
Pi05TorchFrontendSm89(
    checkpoint_dir,
    num_views=1,   # 单视角
    num_steps=1    # 1步扩散 (质量略降)
)

# 注意: 默认参数(num_views=2, num_steps=10)会~138ms
```

---

## ⚠️ 硬件限制说明

### RTX 4060 Ti (SM89)

```
限制:
  ❌ 无FP8 Tensor Core GEMM
  ❌ 不支持CUTLASS SM100 kernel
  ❌ Pi0.5最优59ms，无法<50ms

解决方案:
  升级到SM90+ GPU:
    - RTX 4090 (FP8支持)
    - RTX 5090 (SM100)
    - Jetson AGX Thor (嵌入式)
```

---

## 📊 性能基准

### GROOT N1.7 (100次测试)

| 指标 | 值 |
|------|-----|
| 平均延迟 | 38.08 ms |
| 标准差 | 0.14 ms |
| P99 | 38.55 ms |
| 吞吐量 | 26.3 req/s |

### Batch推理

| Batch Size | 延迟 | 吞吐量 |
|------------|------|--------|
| 5 | 192ms | 26 req/s |
| 10 | 389ms | 26 req/s |
| 20 | 773ms | 26 req/s |

---

## 🔒 精度保证

```
验证结果:
  MSE: 0.000000 (完美)
  Cosine Similarity: 1.000000 (完美)

技术保证:
  ✅ 禁用int8/int4量化
  ✅ FP16精度保持
  ✅ 计算等价性验证
```

---

## 📞 支持

- 文档: `/data/FlashRT/docs/`
- 会话记录: `/data/session/mos0s7yx/`
- 状态报告: `/data/session/mos0s7yx/session-status.md`