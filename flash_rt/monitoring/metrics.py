# FlashRT Prometheus Metrics Exporter
# 导出性能指标供 Prometheus 抓取

import time
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from functools import wraps

from prometheus_client import Counter, Histogram, Gauge, Info, CollectorRegistry
from prometheus_client import start_http_server, REGISTRY

logger = logging.getLogger(__name__)

# ============== Prometheus 指标定义 ==============

# 请求计数器
REQUEST_COUNT = Counter(
    'flashrt_request_total',
    'Total number of inference requests',
    ['model', 'status']
)

# 请求延迟直方图
REQUEST_LATENCY = Histogram(
    'flashrt_request_latency_seconds',
    'Request latency in seconds',
    ['model'],
    buckets=[0.01, 0.02, 0.03, 0.04, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0]
)

# 推理延迟直方图（更细粒度）
INFERENCE_LATENCY = Histogram(
    'flashrt_inference_latency_seconds',
    'Inference latency in seconds (GPU time)',
    ['model', 'batch_size'],
    buckets=[0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.1, 0.15, 0.2]
)

# GPU 内存使用
GPU_MEMORY_USED = Gauge(
    'flashrt_gpu_memory_used_bytes',
    'GPU memory used in bytes',
    ['gpu_id', 'model']
)

GPU_MEMORY_TOTAL = Gauge(
    'flashrt_gpu_memory_total_bytes',
    'GPU memory total in bytes',
    ['gpu_id']
)

# GPU 利用率
GPU_UTILIZATION = Gauge(
    'flashrt_gpu_utilization_percent',
    'GPU utilization percentage',
    ['gpu_id']
)

# 模型信息
MODEL_INFO = Info(
    'flashrt_model',
    'Model information',
    ['model', 'version']
)

# 队列深度
QUEUE_DEPTH = Gauge(
    'flashrt_queue_depth',
    'Number of requests in queue',
    ['model']
)

# 吞吐量
THROUGHPUT = Counter(
    'flashrt_tokens_generated_total',
    'Total tokens generated',
    ['model']
)

# 精度指标
ACCURACY_MSE = Gauge(
    'flashrt_accuracy_mse',
    'Mean squared error for accuracy verification',
    ['model']
)

ACCURACY_COSINE = Gauge(
    'flashrt_accuracy_cosine',
    'Cosine similarity for accuracy verification',
    ['model']
)

# CUDA Graph 状态
CUDA_GRAPH_STATUS = Gauge(
    'flashrt_cuda_graph_captured',
    'Whether CUDA graph is captured (1=yes, 0=no)',
    ['model']
)

# 缓存命中率
CACHE_HIT_RATE = Gauge(
    'flashrt_cache_hit_rate',
    'KV cache hit rate',
    ['model']
)


@dataclass
class MetricsConfig:
    """Metrics configuration."""
    enabled: bool = True
    port: int = 9090
    prefix: str = 'flashrt'
    labels: Dict[str, str] = field(default_factory=dict)


class FlashRTMetrics:
    """FlashRT Prometheus metrics manager.
    
    Usage::
    
        from flash_rt.monitoring.metrics import FlashRTMetrics
        
        metrics = FlashRTMetrics()
        metrics.start_server(port=9090)
        
        # Record inference
        with metrics.track_inference('groot_n17'):
            result = model.infer(obs)
        
        # Update GPU metrics
        metrics.update_gpu_metrics()
    """
    
    _instance: Optional['FlashRTMetrics'] = None
    
    def __init__(self, config: Optional[MetricsConfig] = None):
        self.config = config or MetricsConfig()
        self._server_started = False
        self._models: Dict[str, dict] = {}
    
    @classmethod
    def get_instance(cls) -> 'FlashRTMetrics':
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def start_server(self, port: int = 9090):
        """Start Prometheus metrics HTTP server."""
        if self._server_started:
            logger.warning("Metrics server already running")
            return
        
        try:
            start_http_server(port, registry=REGISTRY)
            self._server_started = True
            logger.info(f"Metrics server started on port {port}")
        except Exception as e:
            logger.error(f"Failed to start metrics server: {e}")
    
    def track_inference(self, model: str, batch_size: int = 1):
        """Context manager to track inference latency.
        
        Usage::
        
            with metrics.track_inference('groot_n17', batch_size=1):
                result = model.infer(obs)
        """
        return _InferenceTracker(self, model, batch_size)
    
    def record_request(self, model: str, status: str = 'success'):
        """Record a request."""
        REQUEST_COUNT.labels(model=model, status=status).inc()
    
    def record_latency(self, model: str, latency_ms: float):
        """Record request latency."""
        REQUEST_LATENCY.labels(model=model).observe(latency_ms / 1000)
    
    def record_inference_latency(self, model: str, latency_ms: float, batch_size: int = 1):
        """Record inference latency."""
        INFERENCE_LATENCY.labels(model=model, batch_size=str(batch_size)).observe(latency_ms / 1000)
    
    def update_gpu_metrics(self):
        """Update GPU memory and utilization metrics."""
        try:
            import torch
            if not torch.cuda.is_available():
                return
            
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                allocated = torch.cuda.memory_allocated(i)
                total = props.total_memory
                
                GPU_MEMORY_TOTAL.labels(gpu_id=str(i)).set(total)
                
                # Try to get utilization
                try:
                    util = torch.cuda.utilization(i)
                    GPU_UTILIZATION.labels(gpu_id=str(i)).set(util)
                except:
                    pass
        except Exception as e:
            logger.debug(f"Failed to update GPU metrics: {e}")
    
    def register_model(self, model: str, version: str = 'unknown', **metadata):
        """Register a model with info."""
        MODEL_INFO.labels(model=model, version=version).info({
            'model': model,
            'version': version,
            **{k: str(v) for k, v in metadata.items()}
        })
        self._models[model] = {'version': version, **metadata}
    
    def set_queue_depth(self, model: str, depth: int):
        """Set current queue depth."""
        QUEUE_DEPTH.labels(model=model).set(depth)
    
    def record_tokens(self, model: str, count: int):
        """Record generated tokens."""
        THROUGHPUT.labels(model=model).inc(count)
    
    def set_accuracy(self, model: str, mse: float, cosine: float):
        """Set accuracy metrics."""
        ACCURACY_MSE.labels(model=model).set(mse)
        ACCURACY_COSINE.labels(model=model).set(cosine)
    
    def set_cuda_graph_status(self, model: str, captured: bool):
        """Set CUDA Graph capture status."""
        CUDA_GRAPH_STATUS.labels(model=model).set(1 if captured else 0)
    
    def set_cache_hit_rate(self, model: str, rate: float):
        """Set cache hit rate."""
        CACHE_HIT_RATE.labels(model=model).set(rate)


class _InferenceTracker:
    """Context manager for tracking inference latency."""
    
    def __init__(self, metrics: FlashRTMetrics, model: str, batch_size: int):
        self.metrics = metrics
        self.model = model
        self.batch_size = batch_size
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = (time.perf_counter() - self.start_time) * 1000
        status = 'success' if exc_type is None else 'error'
        
        self.metrics.record_request(self.model, status)
        self.metrics.record_latency(self.model, elapsed)
        self.metrics.record_inference_latency(self.model, elapsed, self.batch_size)
        
        if exc_type is not None:
            logger.error(f"Inference error for {self.model}: {exc_val}")
        
        return False  # Don't suppress exceptions


def track_latency(model: str):
    """Decorator to track function latency.
    
    Usage::
    
        @track_latency('groot_n17')
        def run_inference(obs):
            return model.infer(obs)
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            metrics = FlashRTMetrics.get_instance()
            with metrics.track_inference(model):
                return func(*args, **kwargs)
        return wrapper
    return decorator


# ============== FastAPI Integration ==============

def add_metrics_endpoint(app):
    """Add /metrics endpoint to FastAPI app.
    
    Usage::
    
        from fastapi import FastAPI
        from flash_rt.monitoring.metrics import add_metrics_endpoint
        
        app = FastAPI()
        add_metrics_endpoint(app)
    """
    from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
    from fastapi import Response
    
    @app.get("/metrics")
    async def metrics():
        """Prometheus metrics endpoint."""
        return Response(
            content=generate_latest(REGISTRY),
            media_type=CONTENT_TYPE_LATEST
        )
    
    @app.get("/health")
    async def health():
        """Health check endpoint."""
        return {"status": "healthy"}


# ============== Convenience Functions ==============

def start_metrics_server(port: int = 9090):
    """Start metrics server with default configuration."""
    metrics = FlashRTMetrics.get_instance()
    metrics.start_server(port)
    return metrics


__all__ = [
    'FlashRTMetrics',
    'MetricsConfig',
    'track_latency',
    'add_metrics_endpoint',
    'start_metrics_server',
]