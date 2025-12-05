#!/usr/bin/env python3
"""
Batched FastAPI Server for ChatterboxTTS with vLLM Architecture
Implements proper request batching to utilize GPU concurrency.


The server will be available at http://localhost:8000
Metrics will be available at http://localhost:8000/metrics
"""

import os
import time
import base64
import io
import asyncio
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass
from collections import deque
import torch
import torchaudio as ta
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response, JSONResponse
from pydantic import BaseModel
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from chatterbox_vllm.tts import ChatterboxTTS
import uvicorn
import threading

# Server configuration
HOST = "0.0.0.0"
PORT = 8000
MAX_MODEL_LEN = 1200
MAX_BATCH_SIZE = 40  # Maximum requests to batch together
BATCH_TIMEOUT = 0.1  # Wait up to 100ms to fill a batch (tune this!)

# Initialize FastAPI
app = FastAPI(title="ChatterboxTTS vLLM Batched Server")

# Prometheus metrics
REQUEST_COUNT = Counter('chatterbox_requests_total', 'Total number of requests', ['status'])
REQUEST_LATENCY = Histogram('chatterbox_request_latency_seconds', 'Request latency in seconds')
QUEUE_TIME = Histogram('chatterbox_queue_time_seconds', 'Time spent in queue')
BATCH_SIZE_METRIC = Histogram('chatterbox_batch_size', 'Actual batch sizes processed')
TTFA = Histogram('chatterbox_ttfa_seconds', 'Time to first audio in seconds')
REQUESTS_IN_PROGRESS = Gauge('chatterbox_requests_in_progress', 'Number of requests currently being processed')
GPU_MEMORY_USED = Gauge('chatterbox_gpu_memory_mb', 'GPU memory used in MB')
KV_CACHE_USAGE = Gauge('vllm_kv_cache_usage_perc', 'KV cache usage percentage')
GPU_UTILIZATION = Gauge('vllm_gpu_cache_usage_perc', 'GPU cache usage percentage')
PREEMPTIONS = Counter('vllm_num_preemptions_total', 'Total number of preemptions')
REQUESTS_RUNNING = Gauge('vllm_num_requests_running', 'Number of requests running')
REQUESTS_WAITING = Gauge('vllm_num_requests_waiting', 'Number of requests waiting')
BATCHES_PROCESSED = Counter('chatterbox_batches_processed_total', 'Total batches processed')

# Global model instance
model = None
batch_processor = None

class TTSRequest(BaseModel):
    model: str = "chatterbox-tts"
    input: str
    voice: str  # Base64 encoded audio
    exaggeration: Optional[float] = 0.5
    min_p: Optional[float] = 0.1
    top_p: Optional[float] = 0.8
    response_format: Optional[str] = "mp3"

@dataclass
class QueuedRequest:
    """A request waiting in the queue"""
    request_id: str
    text: str
    audio_prompt_path: str
    params: Dict
    queue_time: float
    future: asyncio.Future

class BatchProcessor:
    """
    Continuously processes requests in batches to maximize GPU utilization.
    This is the core of vLLM's architecture.
    """
    
    def __init__(self, model, max_batch_size: int, batch_timeout: float):
        self.model = model
        self.max_batch_size = max_batch_size
        self.batch_timeout = batch_timeout
        self.queue: deque = deque()
        self.lock = asyncio.Lock()
        self.running = False
        self.processor_task = None
        
        print(f"[BatchProcessor] Initialized with max_batch_size={max_batch_size}, timeout={batch_timeout}s")
    
    async def add_request(self, req: QueuedRequest):
        """Add a request to the queue"""
        async with self.lock:
            self.queue.append(req)
            REQUESTS_WAITING.set(len(self.queue))
    
    async def get_batch(self) -> List[QueuedRequest]:
        """
        Get a batch of requests from the queue.
        Waits until we have max_batch_size requests OR timeout expires.
        """
        batch = []
        deadline = time.time() + self.batch_timeout
        
        while len(batch) < self.max_batch_size:
            async with self.lock:
                if self.queue:
                    batch.append(self.queue.popleft())
                    REQUESTS_WAITING.set(len(self.queue))
                elif batch:
                    # We have some requests, process them
                    break
                else:
                    # Queue is empty, wait a bit
                    pass
            
            # If timeout expired and we have at least one request, process
            if time.time() >= deadline and batch:
                break
            
            # Small sleep to avoid busy waiting
            await asyncio.sleep(0.001)
        
        return batch
    
    async def process_batch(self, batch: List[QueuedRequest]):
        """Process a batch of requests on the GPU"""
        if not batch:
            return
        
        batch_start = time.time()
        batch_size = len(batch)
        
        BATCH_SIZE_METRIC.observe(batch_size)
        BATCHES_PROCESSED.inc()
        REQUESTS_RUNNING.set(batch_size)
        
        print(f"[BatchProcessor] Processing batch of {batch_size} requests")
        
        try:
            # Extract texts and parameters
            texts = [req.text for req in batch]
            
            # For simplicity, use the first request's audio prompt and params
            # In production, you'd handle different audio prompts per request
            audio_prompt_path = batch[0].audio_prompt_path
            params = batch[0].params
            
            # Generate audio for the entire batch in one GPU call!
            generation_start = time.time()
            audios = self.model.generate(
                texts,
                audio_prompt_path=audio_prompt_path,
                exaggeration=params.get('exaggeration', 0.5),
                min_p=params.get('min_p', 0.1),
                top_p=params.get('top_p', 0.8),
            )
            generation_time = time.time() - generation_start
            
            # Process individual results
            for i, (req, audio) in enumerate(zip(batch, audios)):
                try:
                    # Convert audio to bytes
                    buffer = io.BytesIO()
                    ta.save(buffer, audio, self.model.sr, format=params.get('response_format', 'mp3'))
                    audio_bytes = buffer.getvalue()
                    
                    # Calculate metrics
                    queue_time = generation_start - req.queue_time
                    total_time = time.time() - req.queue_time
                    
                    QUEUE_TIME.observe(queue_time)
                    REQUEST_LATENCY.observe(total_time)
                    TTFA.observe(queue_time + (generation_time / batch_size))  # Approximate TTFA
                    REQUEST_COUNT.labels(status='success').inc()
                    
                    # Return result to waiting client
                    req.future.set_result({
                        'audio': audio_bytes,
                        'queue_time': queue_time,
                        'generation_time': generation_time,
                        'batch_size': batch_size,
                    })
                    
                except Exception as e:
                    print(f"[BatchProcessor] Error processing request {req.request_id}: {e}")
                    req.future.set_exception(e)
                    REQUEST_COUNT.labels(status='error').inc()
            
            print(f"[BatchProcessor] Batch of {batch_size} completed in {generation_time:.3f}s ({generation_time/batch_size:.3f}s per request)")
            
        except Exception as e:
            print(f"[BatchProcessor] Batch processing failed: {e}")
            # Set exception for all requests in batch
            for req in batch:
                if not req.future.done():
                    req.future.set_exception(e)
                    REQUEST_COUNT.labels(status='error').inc()
        
        finally:
            REQUESTS_RUNNING.set(0)
    
    async def run(self):
        """Main processing loop - continuously processes batches"""
        self.running = True
        print("[BatchProcessor] Starting continuous batch processing loop")
        
        while self.running:
            try:
                # Get next batch (blocks until we have requests or timeout)
                batch = await self.get_batch()
                
                if batch:
                    # Process the batch
                    await self.process_batch(batch)
                else:
                    # No requests, small sleep
                    await asyncio.sleep(0.01)
                    
            except Exception as e:
                print(f"[BatchProcessor] Error in processing loop: {e}")
                await asyncio.sleep(0.1)
    
    def start(self):
        """Start the batch processor"""
        if not self.processor_task:
            self.processor_task = asyncio.create_task(self.run())
    
    async def stop(self):
        """Stop the batch processor"""
        self.running = False
        if self.processor_task:
            await self.processor_task

@app.on_event("startup")
async def startup_event():
    """Initialize the model and batch processor on startup"""
    global model, batch_processor
    
    print("=" * 80)
    print("Starting ChatterboxTTS vLLM Batched Server")
    print("=" * 80)
    print(f"Host: {HOST}")
    print(f"Port: {PORT}")
    print(f"Max Model Length: {MAX_MODEL_LEN}")
    print(f"Max Batch Size: {MAX_BATCH_SIZE}")
    print(f"Batch Timeout: {BATCH_TIMEOUT}s")
    print("=" * 80)
    
    # Load model
    model = ChatterboxTTS.from_pretrained(
        max_batch_size=MAX_BATCH_SIZE,
        max_model_len=MAX_MODEL_LEN,
    )
    
    print("\n✓ Model loaded successfully!")
    print(f"✓ Sample rate: {model.sr} Hz")
    
    # Initialize and start batch processor
    batch_processor = BatchProcessor(
        model=model,
        max_batch_size=MAX_BATCH_SIZE,
        batch_timeout=BATCH_TIMEOUT
    )
    batch_processor.start()
    
    print("\nServer endpoints:")
    print(f"  - API: http://localhost:{PORT}/v1/audio/speech")
    print(f"  - Metrics: http://localhost:{PORT}/metrics")
    print(f"  - Health: http://localhost:{PORT}/health")
    print("=" * 80)
    print("\n✓ Batch processor started!")
    print("✓ Server is ready to accept requests...")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    global model, batch_processor
    
    print("\n[Shutdown] Stopping batch processor...")
    if batch_processor:
        await batch_processor.stop()
    
    print("[Shutdown] Shutting down model...")
    if model:
        model.shutdown()
    
    print("Server shut down successfully.")

@app.get("/health")
async def health():
    """Health check endpoint"""
    queue_size = len(batch_processor.queue) if batch_processor else 0
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "batch_processor_running": batch_processor.running if batch_processor else False,
        "queue_size": queue_size
    }

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    # Update GPU metrics
    if torch.cuda.is_available():
        gpu_mem = torch.cuda.memory_allocated() / (1024 ** 2)  # MB
        GPU_MEMORY_USED.set(gpu_mem)
        
        # Calculate utilization
        total_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)
        usage_percent = (gpu_mem / total_mem) * 100
        GPU_UTILIZATION.set(usage_percent)
        KV_CACHE_USAGE.set(usage_percent * 0.6)  # Estimate
    
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.post("/v1/audio/speech")
async def create_speech(request: TTSRequest):
    """
    OpenAI-compatible TTS endpoint with batching
    """
    if model is None or batch_processor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    request_id = f"req_{int(time.time() * 1000000)}"
    queue_time = time.time()
    
    REQUESTS_IN_PROGRESS.inc()
    
    try:
        # Decode base64 audio prompt and save temporarily
        try:
            audio_data = base64.b64decode(request.voice)
        except Exception as e:
            REQUEST_COUNT.labels(status='error').inc()
            raise HTTPException(status_code=400, detail=f"Invalid base64 audio: {str(e)}")
        
        # Save audio prompt to temporary file
        temp_audio_path = f"/tmp/audio_prompt_{request_id}.mp3"
        with open(temp_audio_path, "wb") as f:
            f.write(audio_data)
        
        # Create future for this request
        future = asyncio.Future()
        
        # Create queued request
        queued_req = QueuedRequest(
            request_id=request_id,
            text=request.input,
            audio_prompt_path=temp_audio_path,
            params={
                'exaggeration': request.exaggeration,
                'min_p': request.min_p,
                'top_p': request.top_p,
                'response_format': request.response_format,
            },
            queue_time=queue_time,
            future=future
        )
        
        # Add to batch processor queue
        await batch_processor.add_request(queued_req)
        
        # Wait for result (will be set when batch is processed)
        result = await future
        
        # Clean up temp file
        try:
            os.remove(temp_audio_path)
        except:
            pass
        
        return Response(
            content=result['audio'],
            media_type=f"audio/{request.response_format}",
            headers={
                "X-Request-Id": request_id,
                "X-Queue-Time": f"{result['queue_time']:.3f}s",
                "X-Generation-Time": f"{result['generation_time']:.3f}s",
                "X-Batch-Size": str(result['batch_size']),
            }
        )
        
    except Exception as e:
        REQUEST_COUNT.labels(status='error').inc()
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        REQUESTS_IN_PROGRESS.dec()

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