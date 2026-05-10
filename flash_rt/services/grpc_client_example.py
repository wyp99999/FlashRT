"""FlashRT gRPC Client Example

Example client for FlashRT gRPC inference service.

Usage:
    python -m flash_rt.services.grpc_client_example --server localhost:50051
"""

import argparse
import time
import numpy as np
import grpc
import sys
sys.path.insert(0, '/data/FlashRT')

import flash_rt.services.flashrt_pb2 as pb2
import flash_rt.services.flashrt_pb2_grpc as pb2_grpc


def create_dummy_image():
    """Create dummy 224x224x3 image."""
    return np.zeros((224, 224, 3), dtype=np.uint8).tobytes()


def main():
    parser = argparse.ArgumentParser(description="FlashRT gRPC Client Example")
    parser.add_argument('--server', type=str, default='localhost:50051',
                        help='gRPC server address')
    parser.add_argument('--model', type=str, default='groot_n17',
                        help='Model to test')
    parser.add_argument('--iterations', type=int, default=10,
                        help='Number of test iterations')
    parser.add_argument('--batch_size', type=int, default=5,
                        help='Batch size for batch test')
    
    args = parser.parse_args()
    
    # Connect to server
    channel = grpc.insecure_channel(args.server)
    stub = pb2_grpc.FlashRTServiceStub(channel)
    
    print(f"Connecting to gRPC server at {args.server}")
    
    # Health check
    health = stub.Health(pb2.HealthRequest())
    print(f"\nHealth Check:")
    print(f"  Status: {health.status}")
    print(f"  Model: {health.model}")
    print(f"  GPU: {health.gpu_name}")
    print(f"  GPU Available: {health.gpu_available}")
    
    # Set prompt
    prompt_resp = stub.SetPrompt(pb2.PromptRequest(prompt='pick up'))
    print(f"\nSet Prompt: {prompt_resp.prompt}")
    
    # Build pipeline (warmup)
    build_resp = stub.BuildPipeline(pb2.BuildRequest())
    print(f"Build Pipeline: {build_resp.pipeline_built}")
    
    # Single inference test
    print(f"\n=== Single Inference Test ({args.iterations} iterations) ===")
    latencies = []
    
    for i in range(args.iterations):
        request = pb2.InferRequest(
            prompt='pick up',
            image_data=create_dummy_image(),
        )
        
        response = stub.Infer(request)
        latencies.append(response.latency_ms)
        
        if i == 0:
            print(f"  First inference: {response.latency_ms:.2f}ms")
            print(f"  Actions count: {len(response.actions)}")
            if response.actions:
                print(f"  First action dims: {len(response.actions[0].values)}")
    
    print(f"\nSingle Inference Results:")
    print(f"  Mean latency: {np.mean(latencies):.2f}ms")
    print(f"  Std latency: {np.std(latencies):.2f}ms")
    print(f"  Min latency: {np.min(latencies):.2f}ms")
    print(f"  Max latency: {np.max(latencies):.2f}ms")
    
    # Batch inference test
    print(f"\n=== Batch Inference Test (batch_size={args.batch_size}) ===")
    
    batch_request = pb2.BatchInferRequest(
        requests=[pb2.InferRequest(prompt='pick up', image_data=create_dummy_image()) 
                  for _ in range(args.batch_size)],
        max_batch_size=10,
    )
    
    batch_response = stub.BatchInfer(batch_request)
    
    print(f"  Batch size: {batch_response.batch_size}")
    print(f"  Total latency: {batch_response.total_latency_ms:.2f}ms")
    print(f"  Per-request: {batch_response.total_latency_ms/batch_response.batch_size:.2f}ms")
    print(f"  Throughput: {batch_response.throughput_req_per_sec:.2f} req/s")
    
    # Get stats
    stats = stub.GetStats(pb2.StatsRequest())
    print(f"\n=== Server Statistics ===")
    print(f"  Model: {stats.model}")
    print(f"  Checkpoint: {stats.checkpoint}")
    print(f"  Pipeline built: {stats.pipeline_built}")
    print(f"  Total inferences: {stats.total_inferences}")
    print(f"  Mean latency: {stats.mean_latency_ms:.2f}ms")
    print(f"  Std latency: {stats.std_latency_ms:.2f}ms")
    
    print("\n✅ gRPC client test completed successfully!")
    
    channel.close()


if __name__ == '__main__':
    main()