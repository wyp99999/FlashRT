"""FlashRT gRPC Inference Service

High-performance gRPC API for FlashRT inference.

Key advantages over HTTP REST:
1. Binary protocol - lower serialization overhead
2. HTTP/2 multiplexing - better connection handling
3. Streaming support - efficient batch processing
4. Strong typing - protobuf schema validation

Usage:
    # Generate proto bindings
    python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. flashrt.proto
    
    # Start server
    python -m flash_rt.services.grpc_server --model groot_n17 --port 50051
    
    # Client example:
    import grpc
    import flashrt_pb2, flashrt_pb2_grpc
    channel = grpc.insecure_channel('localhost:50051')
    stub = flashrt_pb2_grpc.FlashRTServiceStub(channel)
    response = stub.Infer(flashrt_pb2.InferRequest(prompt='pick up'))
"""

from __future__ import annotations

import argparse
import logging
import time
import asyncio
import numpy as np
import torch
from concurrent import futures
from typing import Optional, List

# gRPC imports
import grpc
from google.protobuf import empty_pb2

logger = logging.getLogger(__name__)


class FlashRTGrpcService:
    """gRPC FlashRT inference service implementation."""
    
    def __init__(self, model_name: str, checkpoint_dir: str, **kwargs):
        """Initialize gRPC service."""
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
            self._pipe = Pi05TorchFrontendSm89(checkpoint_dir, num_views=num_views)
        elif model_name == "pi0":
            from flash_rt.frontends.torch.pi0_sm89 import Pi0TorchFrontendSm89
            num_views = kwargs.get("num_views", 2)
            self._pipe = Pi0TorchFrontendSm89(checkpoint_dir, num_views=num_views)
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        self._built = False
        self._total_inferences = 0
        self._latency_records = []
        
        logger.info(f"FlashRTGrpcService initialized: {model_name}")
    
    def set_prompt(self, prompt: str) -> None:
        """Set the task prompt."""
        self._pipe.set_prompt(prompt)
    
    def build_pipeline(self) -> None:
        """Build the inference pipeline."""
        if not self._built:
            self._pipe.build_pipeline()
            self._built = True
    
    def _decode_image(self, image_data: bytes) -> np.ndarray:
        """Decode image bytes to numpy array."""
        if image_data is None or len(image_data) == 0:
            return np.zeros((224, 224, 3), dtype=np.uint8)
        
        # Assume raw bytes are 224x224x3
        if len(image_data) == 224 * 224 * 3:
            return np.frombuffer(image_data, dtype=np.uint8).reshape(224, 224, 3)
        
        # Otherwise try PIL decode
        import io
        from PIL import Image
        img = Image.open(io.BytesIO(image_data))
        img = img.resize((224, 224))
        return np.array(img)
    
    def infer(self, request) -> dict:
        """Run single inference."""
        if not self._built:
            self.build_pipeline()
        
        obs = {'image': self._decode_image(request.image_data)}
        
        if request.wrist_image_data:
            obs['wrist_image'] = self._decode_image(request.wrist_image_data)
        
        if request.state:
            obs['state'] = np.array(request.state, dtype=np.float32)
        
        if request.prompt:
            self._pipe.set_prompt(request.prompt)
        
        if request.noise_seed:
            if hasattr(self._pipe, '_noise_seed'):
                self._pipe._noise_seed = request.noise_seed
        
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        result = self._pipe.infer(obs)
        torch.cuda.synchronize()
        latency = (time.perf_counter() - t0) * 1000
        
        self._total_inferences += 1
        self._latency_records.append(latency)
        
        return {
            'actions': result['actions'],
            'latency_ms': latency,
        }
    
    def batch_infer(self, requests: List) -> dict:
        """Run batch inference."""
        if not self._built:
            self.build_pipeline()
        
        results = []
        start_time = time.perf_counter()
        
        for req in requests:
            result = self.infer(req)
            results.append(result)
        
        total_latency = (time.perf_counter() - start_time) * 1000
        batch_size = len(requests)
        throughput = batch_size / (total_latency / 1000) if total_latency > 0 else 0
        
        return {
            'results': results,
            'total_latency_ms': total_latency,
            'batch_size': batch_size,
            'throughput': throughput,
        }
    
    def get_stats(self) -> dict:
        """Get performance statistics."""
        stats = {
            'model': self.model_name,
            'checkpoint': self.checkpoint_dir,
            'pipeline_built': self._built,
            'total_inferences': self._total_inferences,
            'queue_size': self._total_inferences,
            'queue_pending': 0,
        }
        
        if self._latency_records:
            records = np.array(self._latency_records)
            stats.update({
                'mean_latency_ms': records.mean(),
                'std_latency_ms': records.std(),
                'min_latency_ms': records.min(),
                'max_latency_ms': records.max(),
            })
        
        return stats
    
    def get_health(self) -> dict:
        """Get health status."""
        return {
            'status': 'healthy' if self._built else 'initializing',
            'model': self.model_name,
            'gpu_available': torch.cuda.is_available(),
            'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }


def create_grpc_server(service: FlashRTGrpcService, port: int) -> grpc.Server:
    """Create gRPC server with service implementation."""
    
    # Import generated proto files
    try:
        import flash_rt.services.flashrt_pb2 as pb2
        import flash_rt.services.flashrt_pb2_grpc as pb2_grpc
    except ImportError:
        # Generate proto files if not exist
        import subprocess
        import os
        proto_dir = os.path.dirname(__file__)
        proto_file = os.path.join(proto_dir, 'flashrt.proto')
        
        subprocess.run([
            'python', '-m', 'grpc_tools.protoc',
            '-I', proto_dir,
            '--python_out', proto_dir,
            '--grpc_python_out', proto_dir,
            proto_file
        ], check=True)
        
        import flash_rt.services.flashrt_pb2 as pb2
        import flash_rt.services.flashrt_pb2_grpc as pb2_grpc
    
    # Create service implementation
    class FlashRTServiceServicer(pb2_grpc.FlashRTServiceServicer):
        """gRPC service implementation."""
        
        def Health(self, request, context):
            health = service.get_health()
            return pb2.HealthResponse(
                status=health['status'],
                model=health['model'],
                gpu_available=health['gpu_available'],
                gpu_name=health.get('gpu_name', ''),
            )
        
        def GetStats(self, request, context):
            stats = service.get_stats()
            return pb2.StatsResponse(
                model=stats['model'],
                checkpoint=stats['checkpoint'],
                pipeline_built=stats['pipeline_built'],
                total_inferences=stats.get('total_inferences', 0),
                mean_latency_ms=stats.get('mean_latency_ms', 0.0),
                std_latency_ms=stats.get('std_latency_ms', 0.0),
                min_latency_ms=stats.get('min_latency_ms', 0.0),
                max_latency_ms=stats.get('max_latency_ms', 0.0),
                queue_size=stats.get('queue_size', 0),
                queue_pending=stats.get('queue_pending', 0),
            )
        
        def Infer(self, request, context):
            result = service.infer(request)
            
            actions = []
            for act in result['actions']:
                actions.append(pb2.Action(values=act.tolist()))
            
            return pb2.InferResponse(
                actions=actions,
                latency_ms=result['latency_ms'],
                queue_position=0,
            )
        
        def BatchInfer(self, request, context):
            result = service.batch_infer(request.requests)
            
            results = []
            for r in result['results']:
                actions = []
                for act in r['actions']:
                    actions.append(pb2.Action(values=act.tolist()))
                results.append(pb2.InferResponse(
                    actions=actions,
                    latency_ms=r['latency_ms'],
                    queue_position=0,
                ))
            
            return pb2.BatchInferResponse(
                results=results,
                total_latency_ms=result['total_latency_ms'],
                batch_size=result['batch_size'],
                throughput_req_per_sec=result['throughput'],
            )
        
        def SetPrompt(self, request, context):
            service.set_prompt(request.prompt)
            return pb2.PromptResponse(
                status='ok',
                prompt=request.prompt,
            )
        
        def BuildPipeline(self, request, context):
            service.build_pipeline()
            return pb2.BuildResponse(
                status='ok',
                pipeline_built=True,
            )
    
    # Create server
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
    pb2_grpc.add_FlashRTServiceServicer_to_server(FlashRTServiceServicer(), server)
    server.add_insecure_port(f'[::]:{port}')
    
    return server


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="FlashRT gRPC Inference Server")
    parser.add_argument('--model', type=str, default='groot_n17',
                        choices=['groot_n17', 'pi05', 'pi0'],
                        help='Model name')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Checkpoint directory')
    parser.add_argument('--port', type=int, default=50051,
                        help='gRPC server port')
    parser.add_argument('--num_views', type=int, default=1,
                        help='Number of camera views')
    parser.add_argument('--num_flow_steps', type=int, default=2,
                        help='Number of flow steps (GROOT)')
    parser.add_argument('--prompt', type=str, default='pick up',
                        help='Default prompt')
    
    args = parser.parse_args()
    
    # Default checkpoint paths
    if args.checkpoint is None:
        if args.model == 'groot_n17':
            args.checkpoint = '/data/models/groot-n1.7'
        elif args.model == 'pi05':
            args.checkpoint = '/data/models/pi05_base'
        elif args.model == 'pi0':
            args.checkpoint = '/data/models/pi0'
    
    # Import FlashRT
    import sys
    sys.path.insert(0, '/data/FlashRT')
    
    # Create service
    kwargs = {'num_views': args.num_views}
    if args.model == 'groot_n17':
        kwargs['num_flow_steps'] = args.num_flow_steps
    
    service = FlashRTGrpcService(args.model, args.checkpoint, **kwargs)
    service.set_prompt(args.prompt)
    
    # Warmup
    logger.info("Building pipeline...")
    service.build_pipeline()
    
    logger.info("Running warmup...")
    for _ in range(5):
        obs = {'image': np.zeros((224, 224, 3), dtype=np.uint8)}
        service._pipe.infer(obs)
    
    stats = service.get_stats()
    logger.info(f"Warmup complete: mean latency = {stats.get('mean_latency_ms', 0):.2f}ms")
    
    # Create and start gRPC server
    server = create_grpc_server(service, args.port)
    server.start()
    
    logger.info(f"gRPC server started on port {args.port}")
    logger.info(f"Model: {args.model}")
    logger.info("Press Ctrl+C to stop...")
    
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        server.stop(0)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()