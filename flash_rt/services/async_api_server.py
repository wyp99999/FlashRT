"""FlashRT Async Inference API Service

High-performance async HTTP API for FlashRT inference using FastAPI + uvicorn.

Key improvements over Flask version:
1. Async request handling - better concurrency
2. Request queue management - controlled processing
3. Pydantic models - automatic validation
4. OpenAPI docs - /docs endpoint
5. Streaming responses - optional batch processing

Usage:
    # Start server
    python -m flash_rt.services.async_api_server --model groot_n17 --port 8080
    
    # Send inference request
    curl -X POST http://localhost:8080/infer \
        -H "Content-Type: application/json" \
        -d '{"prompt": "pick up", "image_b64": "base64_encoded_image"}'

Supported models:
    - groot_n17: GROOT N1.7 (38ms, recommended)
    - pi05: Pi0.5 (139ms)
    - pi0: Pi0 (137ms)
"""

from __future__ import annotations

import argparse
import base64
import io
import logging
import time
import asyncio
from typing import Optional, List
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ============ Pydantic Models ============

class InferRequest(BaseModel):
    """Inference request model."""
    prompt: Optional[str] = Field(default="pick up", description="Task prompt")
    image_b64: Optional[str] = Field(default=None, description="Base64 encoded image")
    wrist_image_b64: Optional[str] = Field(default=None, description="Base64 encoded wrist image")
    state: Optional[List[float]] = Field(default=None, description="State vector")
    noise_seed: Optional[int] = Field(default=None, description="Random seed for noise")


class InferResponse(BaseModel):
    """Inference response model."""
    actions: Optional[List[List[float]]] = None  # VLA models
    generated_text: Optional[str] = None  # LLM models
    tokens_generated: Optional[int] = None  # LLM models
    latency_ms: float
    queue_position: Optional[int] = None


class StatsResponse(BaseModel):
    """Statistics response model."""
    model: str
    checkpoint: str
    pipeline_built: bool
    total_inferences: Optional[int] = None
    mean_latency_ms: Optional[float] = None
    std_latency_ms: Optional[float] = None
    min_latency_ms: Optional[float] = None
    max_latency_ms: Optional[float] = None
    queue_size: Optional[int] = None
    queue_pending: Optional[int] = None


class HealthResponse(BaseModel):
    """Health check response model."""
    status: str
    model: str
    gpu_available: bool
    gpu_name: Optional[str] = None


class PromptRequest(BaseModel):
    """Set prompt request model."""
    prompt: str = Field(default="pick up", description="Task prompt")


class BatchInferRequest(BaseModel):
    """Batch inference request model."""
    requests: List[InferRequest] = Field(default_factory=list, description="List of inference requests")
    max_batch_size: Optional[int] = Field(default=10, description="Maximum batch size")


class BatchInferResponse(BaseModel):
    """Batch inference response model."""
    results: List[InferResponse]
    total_latency_ms: float
    batch_size: int
    throughput_req_per_sec: float


# ============ Service Class ============

class AsyncFlashRTService:
    """Async FlashRT inference service with queue management."""
    
    def __init__(self, model_name: str, checkpoint_dir: str, **kwargs):
        """Initialize the async inference service.
        
        Args:
            model_name: Model name (groot_n17, pi05, pi0)
            checkpoint_dir: Path to model checkpoint
            **kwargs: Additional model-specific arguments
        """
        self.model_name = model_name
        self.checkpoint_dir = checkpoint_dir
        
        # Import appropriate frontend
        if model_name == "groot_n17":
            from flash_rt.frontends.torch.groot_n17_sm89 import GrootN17TorchFrontendSm89
            num_views = kwargs.get("num_views", 1)
            num_flow_steps = kwargs.get("num_flow_steps", 2)
            self._pipe = GrootN17TorchFrontendSm89(
                checkpoint_dir, num_views=num_views, num_flow_steps=num_flow_steps)
        elif model_name == "pi05":
            from flash_rt.frontends.torch.pi05_sm89 import Pi05TorchFrontendSm89
            num_views = kwargs.get("num_views", 2)
            num_steps = kwargs.get("num_steps", 1)
            self._pipe = Pi05TorchFrontendSm89(checkpoint_dir, num_views=num_views, num_steps=num_steps)
        elif model_name == "pi0":
            from flash_rt.frontends.torch.pi0_sm89 import Pi0TorchFrontendSm89
            num_views = kwargs.get("num_views", 2)
            num_steps = kwargs.get("num_steps", 1)
            self._pipe = Pi0TorchFrontendSm89(checkpoint_dir, num_views=num_views, num_steps=num_steps)
        elif model_name == "qwen25":
            from flash_rt.frontends.torch.qwen25_sm89 import Qwen25TorchFrontendSm89
            max_new_tokens = kwargs.get("max_new_tokens", 50)
            temperature = kwargs.get("temperature", 0.7)
            self._pipe = Qwen25TorchFrontendSm89(
                checkpoint_dir, max_new_tokens=max_new_tokens, temperature=temperature)
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        self._built = False
        
        # Request queue management
        self._request_queue: asyncio.Queue = asyncio.Queue()
        self._queue_size = 0
        self._pending_count = 0
        self._lock = asyncio.Lock()
        
        # Thread pool for blocking GPU operations
        self._executor = ThreadPoolExecutor(max_workers=1)
        
        logger.info(f"AsyncFlashRTService initialized: {model_name}")
    
    def set_prompt(self, prompt: str) -> None:
        """Set the task prompt."""
        self._pipe.set_prompt(prompt)
    
    def build_pipeline(self) -> None:
        """Build the inference pipeline (CUDA Graph capture)."""
        if not self._built:
            self._pipe.build_pipeline()
            self._built = True
            logger.info("Pipeline built with CUDA Graph")
    
    def _decode_image(self, b64_str: str) -> np.ndarray:
        """Decode base64 image to numpy array."""
        if b64_str is None:
            return np.zeros((224, 224, 3), dtype=np.uint8)
        
        img_bytes = base64.b64decode(b64_str)
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        
        if img_array.shape[0] == 224 * 224 * 3:
            return img_array.reshape(224, 224, 3)
        
        # Try to decode as JPEG/PNG
        from PIL import Image
        img = Image.open(io.BytesIO(img_bytes))
        img = img.resize((224, 224))
        return np.array(img)
    
    def _run_inference_sync(self, observation: dict) -> dict:
        """Run inference synchronously (for thread pool)."""
        result = self._pipe.infer(observation)
        
        if hasattr(self._pipe, 'latency_records') and self._pipe.latency_records:
            # latency_records单位因模型不同:
            # - GROOT/Qwen: 秒 (需要乘1000转毫秒)
            # - Pi0/Pi0.5: 毫秒 (不需要乘)
            last_latency = self._pipe.latency_records[-1]
            if self.model_name in ['pi05', 'pi0']:
                # 已经是毫秒
                result['latency_ms'] = last_latency
            else:
                # 秒转毫秒
                result['latency_ms'] = last_latency * 1000
        
        return result
    
    async def infer(self, request: InferRequest) -> dict:
        """Run inference asynchronously with queue management.
        
        Args:
            request: InferRequest with image and prompt
            
        Returns:
            Dict with 'actions' (VLA models) or 'generated_text' (LLM models) and 'latency_ms'
        """
        if not self._built:
            self.build_pipeline()
        
        # Build observation based on model type
        obs = {}
        
        # LLM models (qwen25) don't need image input
        if self.model_name == "qwen25":
            # For LLM, observation can be empty - prompt is set separately
            pass
        else:
            # VLA models need image input
            obs['image'] = self._decode_image(request.image_b64)
            
            if request.wrist_image_b64:
                obs['wrist_image'] = self._decode_image(request.wrist_image_b64)
            
            if request.state:
                obs['state'] = np.array(request.state, dtype=np.float32)
        
        # Set prompt if different
        if request.prompt:
            self._pipe.set_prompt(request.prompt)
        
        # Set noise seed if provided (for reproducibility)
        if request.noise_seed is not None:
            if hasattr(self._pipe, '_noise_seed'):
                self._pipe._noise_seed = request.noise_seed
        
        # Queue management
        async with self._lock:
            queue_position = self._pending_count
            self._pending_count += 1
        
        # Run inference in thread pool (GPU ops are blocking)
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            self._executor, 
            self._run_inference_sync, 
            obs
        )
        
        # Update queue stats
        async with self._lock:
            self._pending_count -= 1
            self._queue_size += 1
        
        result['queue_position'] = queue_position
        return result
    
    async def batch_infer(self, requests: List[InferRequest]) -> dict:
        """Run batch inference - processes multiple requests sequentially.
        
        Args:
            requests: List of InferRequest
            
        Returns:
            Dict with 'results', 'total_latency_ms', 'batch_size', 'throughput'
        """
        if not self._built:
            self.build_pipeline()
        
        results = []
        start_time = time.perf_counter()
        
        # Process each request sequentially (CUDA Graph replay)
        for req in requests:
            obs = {'image': self._decode_image(req.image_b64)}
            
            if req.wrist_image_b64:
                obs['wrist_image'] = self._decode_image(req.wrist_image_b64)
            
            if req.state:
                obs['state'] = np.array(req.state, dtype=np.float32)
            
            if req.prompt:
                self._pipe.set_prompt(req.prompt)
            
            if req.noise_seed is not None:
                if hasattr(self._pipe, '_noise_seed'):
                    self._pipe._noise_seed = req.noise_seed
            
            # Run in thread pool
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self._executor,
                self._run_inference_sync,
                obs
            )
            
            results.append({
                'actions': result['actions'].tolist(),
                'latency_ms': result.get('latency_ms', 0.0),
            })
        
        total_latency = (time.perf_counter() - start_time) * 1000
        batch_size = len(requests)
        throughput = batch_size / (total_latency / 1000) if total_latency > 0 else 0
        
        # Update queue stats
        async with self._lock:
            self._queue_size += batch_size
        
        return {
            'results': results,
            'total_latency_ms': total_latency,
            'batch_size': batch_size,
            'throughput_req_per_sec': throughput,
        }
    
    def get_stats(self) -> dict:
        """Get performance statistics."""
        stats = {
            "model": self.model_name,
            "checkpoint": self.checkpoint_dir,
            "pipeline_built": self._built,
            "queue_size": self._queue_size,
            "queue_pending": self._pending_count,
        }
        
        if hasattr(self._pipe, 'latency_records') and self._pipe.latency_records:
            records = np.array(self._pipe.latency_records)
            # latency_records单位因模型不同:
            # - GROOT/Qwen: 秒 (需要乘1000转毫秒)
            # - Pi0/Pi0.5: 毫秒 (不需要乘)
            if self.model_name not in ['pi05', 'pi0']:
                records = records * 1000
            stats.update({
                "total_inferences": len(records),
                "mean_latency_ms": records.mean(),
                "std_latency_ms": records.std(),
                "min_latency_ms": records.min(),
                "max_latency_ms": records.max(),
            })
        
        return stats
    
    def get_health(self) -> dict:
        """Get health status."""
        return {
            "status": "healthy" if self._built else "initializing",
            "model": self.model_name,
            "gpu_available": torch.cuda.is_available(),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }


# ============ FastAPI App Factory ============

def create_async_app(service: AsyncFlashRTService) -> FastAPI:
    """Create FastAPI application for the async service."""
    
    app = FastAPI(
        title="FlashRT Inference API",
        description="High-performance async inference API for FlashRT",
        version="2.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )
    
    @app.get("/health", response_model=HealthResponse)
    async def health():
        """Health check endpoint."""
        return service.get_health()
    
    @app.get("/stats", response_model=StatsResponse)
    async def stats():
        """Get performance statistics."""
        return service.get_stats()
    
    @app.post("/set_prompt")
    async def set_prompt(req: PromptRequest):
        """Set the task prompt."""
        service.set_prompt(req.prompt)
        return {"status": "ok", "prompt": req.prompt}
    
    @app.post("/build")
    async def build():
        """Build the pipeline (warmup)."""
        # Build in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(service._executor, service.build_pipeline)
        return {"status": "ok", "pipeline_built": True}
    
    @app.post("/infer", response_model=InferResponse)
    async def infer(req: InferRequest):
        """Run inference.
        
        Request body:
            {
                "prompt": "pick up",
                "image_b64": "base64_encoded_image",
                "wrist_image_b64": "base64_encoded_image (optional)",
                "state": [0.0, 0.0, ...] (optional),
                "noise_seed": 42 (optional)
            }
        """
        try:
            result = await service.infer(req)
            # 根据模型类型返回不同字段
            if service.model_name == 'qwen25':
                # LLM模型
                return InferResponse(
                    actions=None,
                    generated_text=result.get('generated_text', ''),
                    tokens_generated=result.get('tokens_generated', 0),
                    latency_ms=result.get('latency_ms', 0.0),
                    queue_position=result.get('queue_position'),
                )
            else:
                # VLA模型
                return InferResponse(
                    actions=result['actions'].tolist(),
                    generated_text=None,
                    tokens_generated=None,
                    latency_ms=result.get('latency_ms', 0.0),
                    queue_position=result.get('queue_position'),
                )
        except Exception as e:
            import traceback
            logger.error(f"Inference error: {e}\n{traceback.format_exc()}")
            raise HTTPException(status_code=500, detail=str(e) or "Unknown error")
    
    @app.post("/infer_sync")
    async def infer_sync(req: InferRequest):
        """Run inference synchronously (bypasses async queue)."""
        if not service._built:
            service.build_pipeline()
        
        obs = {'image': service._decode_image(req.image_b64)}
        if req.wrist_image_b64:
            obs['wrist_image'] = service._decode_image(req.wrist_image_b64)
        if req.state:
            obs['state'] = np.array(req.state, dtype=np.float32)
        
        if req.prompt:
            service._pipe.set_prompt(req.prompt)
        
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            service._executor,
            service._run_inference_sync,
            obs
        )
        
        return {
            "actions": result['actions'].tolist(),
            "latency_ms": result.get('latency_ms', 0.0),
        }
    
    @app.get("/queue")
    async def queue_status():
        """Get queue status."""
        return {
            "pending": service._pending_count,
            "total_processed": service._queue_size,
        }
    
    @app.post("/batch_infer", response_model=BatchInferResponse)
    async def batch_infer(req: BatchInferRequest):
        """Run batch inference - process multiple requests efficiently.
        
        Request body:
            {
                "requests": [
                    {"prompt": "pick up", "image_b64": "..."},
                    {"prompt": "place down", "image_b64": "..."},
                    ...
                ],
                "max_batch_size": 10
            }
        
        Note: CUDA Graph requires fixed input shapes, so batch processing
        is sequential internally. Throughput gain comes from reduced
        network overhead and request handling.
        """
        if len(req.requests) == 0:
            raise HTTPException(status_code=400, detail="Empty batch")
        
        if len(req.requests) > req.max_batch_size:
            raise HTTPException(
                status_code=400, 
                detail=f"Batch size {len(req.requests)} exceeds max {req.max_batch_size}"
            )
        
        try:
            result = await service.batch_infer(req.requests)
            
            infer_results = [
                InferResponse(
                    actions=r['actions'],
                    latency_ms=r['latency_ms'],
                    queue_position=None,
                ) for r in result['results']
            ]
            
            return BatchInferResponse(
                results=infer_results,
                total_latency_ms=result['total_latency_ms'],
                batch_size=result['batch_size'],
                throughput_req_per_sec=result['throughput_req_per_sec'],
            )
        except Exception as e:
            logger.error(f"Batch inference error: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    return app


# ============ Main Entry Point ============

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="FlashRT Async Inference API Server")
    parser.add_argument('--model', type=str, default='groot_n17',
                        choices=['groot_n17', 'pi05', 'pi0', 'qwen25'],
                        help='Model name')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Checkpoint directory (default: /data/models/<model>)')
    parser.add_argument('--port', type=int, default=8080,
                        help='Server port')
    parser.add_argument('--num_views', type=int, default=1,
                        help='Number of camera views')
    parser.add_argument('--num_flow_steps', type=int, default=2,
                        help='Number of flow steps (GROOT only)')
    parser.add_argument('--prompt', type=str, default='pick up the object from the table and place it in the container',
                        help='Default prompt (use longer prompt for larger Se_max)')
    parser.add_argument('--workers', type=int, default=1,
                        help='Number of uvicorn workers (default: 1 for CUDA Graph)')
    
    args = parser.parse_args()
    
    # Default checkpoint paths
    if args.checkpoint is None:
        if args.model == 'groot_n17':
            args.checkpoint = '/data/models/groot-n1.7'
        elif args.model == 'pi05':
            args.checkpoint = '/data/models/pi05_base'
        elif args.model == 'pi0':
            args.checkpoint = '/data/models/pi0'
        elif args.model == 'qwen25':
            args.checkpoint = '/data/models/qwen2.5-0.5b'
    
    # Import FlashRT
    import sys
    sys.path.insert(0, '/data/FlashRT')
    
    # Create service
    kwargs = {'num_views': args.num_views}
    if args.model == 'groot_n17':
        kwargs['num_flow_steps'] = args.num_flow_steps
    
    service = AsyncFlashRTService(args.model, args.checkpoint, **kwargs)
    service.set_prompt(args.prompt)
    
    # Warmup
    logger.info("Building pipeline (warmup)...")
    service.build_pipeline()
    
    logger.info("Running warmup inference...")
    for _ in range(5):
        obs = {'image': np.zeros((224, 224, 3), dtype=np.uint8)}
        service._run_inference_sync(obs)
    
    stats = service.get_stats()
    logger.info(f"Warmup complete: mean latency = {stats.get('mean_latency_ms', 0):.2f}ms")
    
    # Create FastAPI app
    app = create_async_app(service)
    
    # Start uvicorn
    import uvicorn
    logger.info(f"Starting async server on port {args.port}")
    logger.info(f"API docs available at http://localhost:{args.port}/docs")
    
    # Run with single worker (CUDA Graph requires single process)
    uvicorn.run(
        app,
        host='0.0.0.0',
        port=args.port,
        workers=args.workers,
        log_level="info",
    )


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()