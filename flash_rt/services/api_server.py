"""FlashRT Inference API Service

A simple HTTP API for FlashRT inference using Flask.

Usage:
    # Start server
    python -m flash_rt.services.api_server --model groot_n17 --port 8080
    
    # Send inference request
    curl -X POST http://localhost:8080/infer \
        -H "Content-Type: application/json" \
        -d '{"prompt": "pick up", "image": "base64_encoded_image"}'

Supported models:
    - groot_n17: GROOT N1.7 (38ms, recommended)
    - pi05: Pi0.5 (138ms)
    - pi0: Pi0 (137ms)
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import logging
import time
from typing import Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


class FlashRTService:
    """FlashRT inference service wrapper."""
    
    def __init__(self, model_name: str, checkpoint_dir: str, **kwargs):
        """Initialize the inference service.
        
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
            self._pipe = Pi05TorchFrontendSm89(checkpoint_dir, num_views=num_views)
        elif model_name == "pi0":
            from flash_rt.frontends.torch.pi0_sm89 import Pi0TorchFrontendSm89
            num_views = kwargs.get("num_views", 2)
            self._pipe = Pi0TorchFrontendSm89(checkpoint_dir, num_views=num_views)
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        self._built = False
        logger.info(f"FlashRTService initialized: {model_name}")
    
    def set_prompt(self, prompt: str) -> None:
        """Set the task prompt."""
        self._pipe.set_prompt(prompt)
    
    def build_pipeline(self) -> None:
        """Build the inference pipeline (CUDA Graph capture)."""
        if not self._built:
            self._pipe.build_pipeline()
            self._built = True
            logger.info("Pipeline built with CUDA Graph")
    
    def infer(self, observation: dict) -> dict:
        """Run inference on observation.
        
        Args:
            observation: Dict with 'image', optional 'wrist_image', optional 'state'
            
        Returns:
            Dict with 'actions' and 'latency_ms'
        """
        if not self._built:
            self.build_pipeline()
        
        result = self._pipe.infer(observation)
        
        # Add latency info
        if hasattr(self._pipe, 'latency_records') and self._pipe.latency_records:
            result['latency_ms'] = self._pipe.latency_records[-1] * 1000
        
        return result
    
    def get_stats(self) -> dict:
        """Get performance statistics."""
        stats = {
            "model": self.model_name,
            "checkpoint": self.checkpoint_dir,
            "pipeline_built": self._built,
        }
        
        if hasattr(self._pipe, 'latency_records') and self._pipe.latency_records:
            records = np.array(self._pipe.latency_records) * 1000
            stats.update({
                "total_inferences": len(records),
                "mean_latency_ms": records.mean(),
                "std_latency_ms": records.std(),
                "min_latency_ms": records.min(),
                "max_latency_ms": records.max(),
            })
        
        return stats


def create_flask_app(service: FlashRTService):
    """Create Flask application for the service."""
    from flask import Flask, request, jsonify
    
    app = Flask(__name__)
    
    @app.route('/health', methods=['GET'])
    def health():
        """Health check endpoint."""
        return jsonify({"status": "healthy", "model": service.model_name})
    
    @app.route('/stats', methods=['GET'])
    def stats():
        """Get performance statistics."""
        return jsonify(service.get_stats())
    
    @app.route('/set_prompt', methods=['POST'])
    def set_prompt():
        """Set the task prompt."""
        data = request.get_json()
        prompt = data.get('prompt', 'pick up')
        service.set_prompt(prompt)
        return jsonify({"status": "ok", "prompt": prompt})
    
    @app.route('/build', methods=['POST'])
    def build():
        """Build the pipeline (warmup)."""
        service.build_pipeline()
        return jsonify({"status": "ok", "pipeline_built": True})
    
    @app.route('/infer', methods=['POST'])
    def infer():
        """Run inference.
        
        Request body:
            {
                "prompt": "pick up",
                "image": "base64_encoded_image",
                "wrist_image": "base64_encoded_image (optional)",
                "state": [0.0, 0.0, ...] (optional)
            }
        """
        data = request.get_json()
        
        # Set prompt if provided
        if 'prompt' in data:
            service.set_prompt(data['prompt'])
        
        # Decode images
        def decode_image(b64_str):
            if b64_str is None:
                return None
            img_bytes = base64.b64decode(b64_str)
            # Assume image is already correct size (224x224x3)
            img_array = np.frombuffer(img_bytes, dtype=np.uint8)
            if img_array.shape[0] == 224 * 224 * 3:
                return img_array.reshape(224, 224, 3)
            # Try to decode as JPEG/PNG
            from PIL import Image
            img = Image.open(io.BytesIO(img_bytes))
            img = img.resize((224, 224))
            return np.array(img)
        
        # Build observation
        obs = {}
        if 'image' in data:
            obs['image'] = decode_image(data['image'])
        else:
            # Use dummy image if not provided
            obs['image'] = np.zeros((224, 224, 3), dtype=np.uint8)
        
        if 'wrist_image' in data:
            obs['wrist_image'] = decode_image(data['wrist_image'])
        
        if 'state' in data:
            obs['state'] = np.array(data['state'], dtype=np.float32)
        
        # Run inference
        result = service.infer(obs)
        
        return jsonify({
            "actions": result['actions'].tolist(),
            "latency_ms": result.get('latency_ms', 0.0),
        })
    
    return app


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="FlashRT Inference API Server")
    parser.add_argument('--model', type=str, default='groot_n17',
                        choices=['groot_n17', 'pi05', 'pi0'],
                        help='Model name')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Checkpoint directory (default: /data/models/<model>)')
    parser.add_argument('--port', type=int, default=8080,
                        help='Server port')
    parser.add_argument('--num_views', type=int, default=1,
                        help='Number of camera views')
    parser.add_argument('--num_flow_steps', type=int, default=2,
                        help='Number of flow steps (GROOT only)')
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
    
    # Create service
    kwargs = {'num_views': args.num_views}
    if args.model == 'groot_n17':
        kwargs['num_flow_steps'] = args.num_flow_steps
    
    service = FlashRTService(args.model, args.checkpoint, **kwargs)
    service.set_prompt(args.prompt)
    
    # Warmup
    logger.info("Building pipeline (warmup)...")
    service.build_pipeline()
    
    # Run warmup inference
    logger.info("Running warmup inference...")
    for _ in range(5):
        service.infer({'image': np.zeros((224, 224, 3), dtype=np.uint8)})
    
    stats = service.get_stats()
    logger.info(f"Warmup complete: mean latency = {stats.get('mean_latency_ms', 0):.2f}ms")
    
    # Start Flask server
    app = create_flask_app(service)
    logger.info(f"Starting server on port {args.port}")
    app.run(host='0.0.0.0', port=args.port, threaded=False)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()