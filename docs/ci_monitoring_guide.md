# FlashRT CI/CD and Monitoring Guide

> **Last Updated**: 2026-05-06
> **Session**: Session 69
> **Status**: CI/CD Pipeline + Monitoring Integration Complete

---

## 📋 Overview

This document describes the CI/CD pipeline and monitoring system for FlashRT SM89 deployment.

### Components Added

| Component | File | Purpose |
|-----------|------|---------|
| GitHub Actions CI | `.github/workflows/ci.yml` | Automated build/test/deploy |
| Prometheus Metrics | `flash_rt/monitoring/metrics.py` | Performance metrics export |
| Grafana Dashboard | `monitoring/grafana-dashboard.json` | Visualization |
| Alert Rules | `monitoring/alerts.yml` | Performance alerting |
| Helm Chart | `deploy/helm/flashrt/` | Kubernetes deployment |
| Benchmarks | `tests/benchmarks/run_all_benchmarks.py` | Performance testing |
| Accuracy Tests | `tests/accuracy/test_accuracy.py` | Quality verification |

---

## 🔄 CI/CD Pipeline

### GitHub Actions Workflow

```yaml
# Location: .github/workflows/ci.yml

Jobs:
1. lint          → Code quality (Black, flake8, isort)
2. test          → Unit tests (pytest)
3. gpu-test      → GPU benchmarks (self-hosted runner)
4. docker-build  → Docker image (ghcr.io)
5. deploy-staging → Staging deployment (develop branch)
6. deploy-production → Production deployment (v* tags)
7. benchmark     → Full performance report
8. release       → GitHub release creation
```

### Prerequisites

1. **Self-hosted GPU Runner**
   ```bash
   # Configure runner on GPU machine
   ./config.sh --labels gpu,sm89
   ```

2. **Kubernetes Secrets**
   ```bash
   # Staging kubeconfig
   kubectl create secret generic kube-config-staging --from-file=config=<staging-kubeconfig>
   
   # Production kubeconfig
   kubectl create secret generic kube-config-production --from-file=config=<prod-kubeconfig>
   ```

3. **Container Registry Access**
   - GitHub Packages (ghcr.io) configured automatically

### Trigger Conditions

| Event | Jobs Run |
|-------|----------|
| Push to main/develop | lint → test → docker-build → deploy |
| Push to sm89/** | lint → test → gpu-test |
| Tag v* | Full pipeline → production deploy → release |
| Pull Request | lint → test |
| Manual dispatch | All + optional benchmark |

---

## 📊 Monitoring Stack

### Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│ FlashRT API │────▶│ Prometheus  │────▶│ Grafana     │
│ (metrics)   │     │ (collect)   │     │ (visualize) │
└─────────────┘     └─────────────┘     └─────────────┘
                           │
                           ▼
                    ┌─────────────┐
                    │ Alertmanager│
                    │ (notify)    │
                    └─────────────┘
```

### Quick Start

```bash
# Start monitoring stack
cd /data/FlashRT/monitoring
docker-compose -f monitoring-compose.yml up -d

# Access
# Prometheus: http://localhost:9090
# Grafana: http://localhost:3000 (admin/flashrt123)
```

### Prometheus Metrics

Key metrics exported:

| Metric | Description | Target |
|--------|-------------|--------|
| `flashrt_request_latency_seconds` | Request latency histogram | < 50ms P95 |
| `flashrt_inference_latency_seconds` | GPU inference time | Model-specific |
| `flashrt_gpu_memory_used_bytes` | GPU memory usage | < 90% |
| `flashrt_gpu_utilization_percent` | GPU utilization | > 50% |
| `flashrt_accuracy_mse` | Accuracy MSE | < 0.01 |
| `flashrt_accuracy_cosine` | Cosine similarity | > 0.999 |
| `flashrt_queue_depth` | Request queue | < 100 |
| `flashrt_tokens_generated_total` | Token throughput | Model-specific |

### Alert Rules

| Alert | Condition | Severity |
|-------|-----------|----------|
| GROOTLatencyHigh | P95 > 50ms | Warning |
| GPUMemoryCritical | > 95% | Critical |
| AccuracyDegradation | MSE > 0.01 | Critical |
| ServiceDown | Instance unreachable | Critical |
| HighErrorRate | > 5% errors | Warning |

### Grafana Dashboard

The dashboard includes:
- P95 latency gauges for each model
- Latency distribution charts (P50/P95/P99)
- Request rate time series
- GPU memory and utilization
- Accuracy metrics (MSE/Cosine)
- Token throughput
- Queue depth

---

## 🎯 Helm Chart

### Usage

```bash
# Install with defaults
helm install flashrt deploy/helm/flashrt \
  --namespace flashrt --create-namespace

# Install with custom values
helm install flashrt deploy/helm/flashrt \
  --namespace flashrt \
  -f custom-values.yaml

# Upgrade
helm upgrade flashrt deploy/helm/flashrt \
  --namespace flashrt

# Uninstall
helm uninstall flashrt --namespace flashrt
```

### Key Values

```yaml
# SM89 GPU configuration
gpu:
  smVersion: "89"  # RTX 4060 Ti/4070/4080

# Model enabling
models:
  groot:
    enabled: true
    replicas: 1
  pi05:
    enabled: true
  qwen:
    enabled: true

# Monitoring
monitoring:
  enabled: true

# Ingress (optional)
ingress:
  enabled: true
  className: nginx
```

### Deployment Resources

| Resource | Template | Description |
|----------|----------|-------------|
| Namespace | namespace.yaml | FlashRT namespace |
| ConfigMap | configmap.yaml | Environment config |
| Secret | secret.yaml | HF token, API keys |
| Deployment | deployment.yaml | Model deployments |
| Service | service.yaml | ClusterIP services |
| Ingress | ingress.yaml | External access |
| PVC | pvc.yaml | Model storage |
| HPA | hpa.yaml | Autoscaling |
| PDB | pdb.yaml | Pod disruption budget |

---

## 🧪 Testing

### Run Benchmarks

```bash
cd /data/FlashRT
python3 tests/benchmarks/run_all_benchmarks.py
```

Output saved to: `benchmark_results/benchmark_*.json`

### Run Accuracy Tests

```bash
python3 tests/accuracy/test_accuracy.py
```

Output saved to: `benchmark_results/accuracy_*.json`

### Expected Results

| Model | Performance | Accuracy |
|-------|-------------|----------|
| GROOT N1.7 | < 50ms ✅ | MSE=0, Cos=1 ✅ |
| Pi0.5 | ~59ms ⚠️ | MSE=0, Cos=1 ✅ |
| Qwen2.5 | ~168ms | Output consistent ✅ |

---

## 🔧 Integration Guide

### Add Metrics to API Server

```python
from flash_rt.monitoring import start_metrics_server, FlashRTMetrics

# Start metrics server
metrics = start_metrics_server(port=9090)

# Register model
metrics.register_model('groot_n17', version='0.1.0')

# Track inference
@app.post("/infer")
async def infer(request: InferRequest):
    with metrics.track_inference('groot_n17'):
        result = model.infer(request.obs)
    
    metrics.update_gpu_metrics()
    return result
```

### Add Metrics Endpoint to FastAPI

```python
from flash_rt.monitoring import add_metrics_endpoint

app = FastAPI()
add_metrics_endpoint(app)  # Adds /metrics and /health
```

---

## 📁 File Structure

```
/data/FlashRT/
├── .github/
│   └── workflows/
│       └── ci.yml              # GitHub Actions CI/CD
│
├── flash_rt/
│   └── monitoring/
│       ├── __init__.py
│       └── metrics.py          # Prometheus metrics
│
├── monitoring/
│   ├── prometheus.yml          # Prometheus config
│   ├── alerts.yml              # Alert rules
│   ├── alertmanager.yml        # Alert routing
│   ├── grafana-dashboard.json  # Dashboard
│   ├── grafana-datasource.yml  # Datasource
│   └── monitoring-compose.yml  # Docker compose
│
├── deploy/
│   ├── helm/
│   │   └── flashrt/
│   │       ├── Chart.yaml
│   │       ├── values.yaml
│   │       └── templates/
│   │           ├── _helpers.tpl
│   │           ├── namespace.yaml
│   │           ├── configmap.yaml
│   │           ├── deployment.yaml
│   │           ├── service.yaml
│   │           ├── ingress.yaml
│   │           ├── pvc.yaml
│   │           ├── hpa.yaml
│   │           └ pdb.yaml
│   │           └ networkpolicy.yaml
│   │           ├── serviceaccount.yaml
│   │           └── secret.yaml
│   └── kubernetes/
│       └── flashrt-deployment.yaml
│
├── tests/
│   ├── benchmarks/
│   │   └── run_all_benchmarks.py
│   ├── accuracy/
│   │   └ test_accuracy.py
│   └── unit/
│       └ test_flashrt.py
│
└── docker/
    ├── Dockerfile.sm89
    └── docker-compose.sm89.yml
```

---

## ⚠️ Notes

1. **Self-hosted Runner Required** for GPU tests
2. **Kubernetes Secrets** must be configured before deployment
3. **Prometheus scraping** requires /metrics endpoint exposed
4. **Helm dependencies** (Prometheus, Grafana) optional
5. **Alertmanager** requires notification channel configuration

---

## 📊 Project Status Update

| Component | Previous | Current | Status |
|-----------|----------|---------|--------|
| CI/CD Pipeline | 0% | 100% | ✅ NEW |
| Monitoring | 0% | 100% | ✅ NEW |
| Helm Chart | 0% | 100% | ✅ NEW |
| Tests | 0% | 100% | ✅ NEW |
| **Overall** | **85%** | **95%** | ✅ |

---

**Session 69 Complete ✅**

**CI/CD + Monitoring + Helm Ready ✅**

**Project 95% Complete ✅**