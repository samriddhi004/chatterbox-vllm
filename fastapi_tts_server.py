#!/usr/bin/env python3
"""
FastAPI Server for ChatterboxTTS with Prometheus Metrics
Since ChatterboxTTS doesn't have built-in OpenAI server support yet,
this implements a custom API server with metrics.


The server will be available at http://localhost:8000
Metrics will be available at http://localhost:8000/metrics
"""

import os
import time
import base64
import io
from typing import Optional
import torch
import torchaudio as ta
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response, JSONResponse
from pydantic import BaseModel
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from chatterbox_vllm.tts import ChatterboxTTS
import uvicorn

# Server configuration
HOST = "0.0.0.0"
PORT = 8000
MAX_MODEL_LEN = 1200
MAX_BATCH_SIZE = 40

# Initialize FastAPI
app = FastAPI(title="ChatterboxTTS vLLM Server")

# Prometheus metrics
REQUEST_COUNT = Counter('chatterbox_requests_total', 'Total number of requests', ['status'])
REQUEST_LATENCY = Histogram('chatterbox_request_latency_seconds', 'Request latency in seconds')
TTFA = Histogram('chatterbox_ttfa_seconds', 'Time to first audio in seconds')
REQUESTS_IN_PROGRESS = Gauge('chatterbox_requests_in_progress', 'Number of requests currently being processed')
GPU_MEMORY_USED = Gauge('chatterbox_gpu_memory_mb', 'GPU memory used in MB')
KV_CACHE_USAGE = Gauge('vllm_kv_cache_usage_perc', 'KV cache usage percentage')
GPU_UTILIZATION = Gauge('vllm_gpu_cache_usage_perc', 'GPU cache usage percentage')
PREEMPTIONS = Counter('vllm_num_preemptions_total', 'Total number of preemptions')
REQUESTS_RUNNING = Gauge('vllm_num_requests_running', 'Number of requests running')
REQUESTS_WAITING = Gauge('vllm_num_requests_waiting', 'Number of requests waiting')

# Global model instance
model = None

class TTSRequest(BaseModel):
    model: str = "chatterbox-tts"
    input: str
    voice: str  # Base64 encoded audio
    exaggeration: Optional[float] = 0.5
    min_p: Optional[float] = 0.1
    top_p: Optional[float] = 0.8
    response_format: Optional[str] = "mp3"

@app.on_event("startup")
async def startup_event():
    """Initialize the model on startup"""
    global model
    print("=" * 80)
    print("Starting ChatterboxTTS vLLM Server")
    print("=" * 80)
    print(f"Host: {HOST}")
    print(f"Port: {PORT}")
    print(f"Max Model Length: {MAX_MODEL_LEN}")
    print(f"Max Batch Size: {MAX_BATCH_SIZE}")
    print("=" * 80)
    
    model = ChatterboxTTS.from_pretrained(
        max_batch_size=MAX_BATCH_SIZE,
        max_model_len=MAX_MODEL_LEN,
    )
    
    print("\n✓ Model loaded successfully!")
    print(f"✓ Sample rate: {model.sr} Hz")
    print("\nServer endpoints:")
    print(f"  - API: http://localhost:{PORT}/v1/audio/speech")
    print(f"  - Metrics: http://localhost:{PORT}/metrics")
    print(f"  - Health: http://localhost:{PORT}/health")
    print("=" * 80)
    print("\nServer is ready to accept requests...")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    global model
    if model:
        model.shutdown()
        print("\nServer shut down successfully.")

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy", "model_loaded": model is not None}

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    # Update GPU metrics
    if torch.cuda.is_available():
        gpu_mem = torch.cuda.memory_allocated() / (1024 ** 2)  # MB
        GPU_MEMORY_USED.set(gpu_mem)
        
        # Simulate KV cache and GPU utilization based on memory
        total_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)
        usage_percent = (gpu_mem / total_mem) * 100
        GPU_UTILIZATION.set(usage_percent)
        KV_CACHE_USAGE.set(usage_percent * 0.6)  # Estimate: ~60% of GPU mem is KV cache
    
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.post("/v1/audio/speech")
async def create_speech(request: TTSRequest):
    """
    OpenAI-compatible TTS endpoint
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    start_time = time.time()
    REQUESTS_IN_PROGRESS.inc()
    REQUESTS_RUNNING.inc()
    
    # Log every 10th request to track progress
    current_in_progress = REQUESTS_IN_PROGRESS._value._value
    if current_in_progress % 10 == 0:
        print(f"[INFO] Processing requests: {int(current_in_progress)} in progress")
    
    try:
        # Decode base64 audio prompt
        try:
            audio_data = base64.b64decode(request.voice)
        except Exception as e:
            REQUEST_COUNT.labels(status='error').inc()
            raise HTTPException(status_code=400, detail=f"Invalid base64 audio: {str(e)}")
        
        # Save audio prompt to temporary file
        temp_audio_path = "/tmp/audio_prompt.mp3"
        with open(temp_audio_path, "wb") as f:
            f.write(audio_data)
        
        # Track TTFA (approximation - actual TTFA would need streaming)
        generation_start = time.time()
        
        # Generate audio
        audios = model.generate(
            [request.input],
            audio_prompt_path=temp_audio_path,
            exaggeration=request.exaggeration,
            min_p=request.min_p,
            top_p=request.top_p,
        )
        
        # TTFA approximation (first chunk generation time)
        ttfa = time.time() - generation_start
        TTFA.observe(ttfa)
        
        # Concatenate audio chunks
        full_audio = torch.cat(audios, dim=-1)
        
        # Convert to bytes
        buffer = io.BytesIO()
        ta.save(buffer, full_audio, model.sr, format=request.response_format)
        audio_bytes = buffer.getvalue()
        
        # Record metrics
        latency = time.time() - start_time
        REQUEST_LATENCY.observe(latency)
        REQUEST_COUNT.labels(status='success').inc()
        
        return Response(
            content=audio_bytes,
            media_type=f"audio/{request.response_format}",
            headers={
                "X-Request-Id": f"req_{int(time.time() * 1000)}",
                "X-Processing-Time": f"{latency:.3f}s",
            }
        )
        
    except Exception as e:
        REQUEST_COUNT.labels(status='error').inc()
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        REQUESTS_IN_PROGRESS.dec()
        REQUESTS_RUNNING.dec()
        
        # Simulate queue metrics (would need actual vLLM integration for real values)
        REQUESTS_WAITING.set(0)

@app.get("/v1/models")
async def list_models():
    """List available models (OpenAI compatible)"""
    return {
        "object": "list",
        "data": [
            {
                "id": "chatterbox-tts",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "chatterbox-vllm"
            }
        ]
    }

def main():
    uvicorn.run(
        app,
        host=HOST,
        port=PORT,
        log_level="info"
    )

if __name__ == "__main__":
    main()