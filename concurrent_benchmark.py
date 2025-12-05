#!/usr/bin/env python3
"""
Concurrent Load Benchmark for ChatterboxTTS (In-Process)
Simulates multiple simultaneous generation requests using threading/batching
"""
import time
import threading
import queue
import json
import psutil
import numpy as np
import torch
import torchaudio as ta
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict, field
from concurrent.futures import ThreadPoolExecutor, as_completed
from chatterbox_vllm.tts import ChatterboxTTS

# Configuration
AUDIO_PROMPT_PATH = "docs/audio-sample-sampu.wav"
TEXT_PATH = "docs/benchmark-text-1.txt"
MAX_CHUNK_SIZE = 400
BATCH_SIZE = 40
NUM_CONCURRENT_REQUESTS = 100  # Number of simulated concurrent requests
NUM_WORKER_THREADS = 4  # Number of threads processing requests

@dataclass
class RequestMetrics:
    """Metrics for a single generation request"""
    request_id: int
    text: str
    text_length: int
    start_time: float
    end_time: float
    duration: float
    queue_time: float  # Time spent waiting in queue
    processing_time: float  # Actual generation time
    success: bool
    error_message: Optional[str] = None
    audio_duration: Optional[float] = None
    num_chunks: int = 0

@dataclass
class ConcurrentBenchmarkMetrics:
    # Request metrics
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    success_rate: float = 0.0
    
    # Timing metrics
    total_benchmark_time: float = 0.0
    avg_request_duration: float = 0.0
    min_request_duration: float = 0.0
    max_request_duration: float = 0.0
    median_request_duration: float = 0.0
    p95_request_duration: float = 0.0
    p99_request_duration: float = 0.0
    
    # Queue metrics
    avg_queue_time: float = 0.0
    max_queue_time: float = 0.0
    avg_processing_time: float = 0.0
    
    # Throughput metrics
    requests_per_second: float = 0.0
    avg_audio_seconds_per_second: float = 0.0
    total_audio_generated_seconds: float = 0.0
    
    # Resource metrics
    peak_cpu_percent: float = 0.0
    avg_cpu_percent: float = 0.0
    peak_ram_mb: float = 0.0
    avg_ram_mb: float = 0.0
    peak_gpu_memory_mb: float = 0.0
    avg_gpu_memory_mb: float = 0.0
    
    # Concurrency metrics
    avg_concurrent_requests: float = 0.0
    max_concurrent_requests: int = 0
    max_queue_depth: int = 0
    
    # Model load time
    model_load_time: float = 0.0
    
    # Detailed request data
    request_details: List[Dict] = field(default_factory=list)

class ResourceMonitor:
    """Monitor system resources during load test"""
    def __init__(self):
        self.monitoring = False
        self.cpu_samples = []
        self.ram_samples = []
        self.gpu_samples = []
        self.thread = None
        
    def start(self):
        self.monitoring = True
        self.cpu_samples = []
        self.ram_samples = []
        self.gpu_samples = []
        self.thread = threading.Thread(target=self._monitor, daemon=True)
        self.thread.start()
        
    def stop(self):
        self.monitoring = False
        if self.thread:
            self.thread.join()
    
    def _monitor(self):
        while self.monitoring:
            self.cpu_samples.append(psutil.cpu_percent(interval=0.1))
            self.ram_samples.append(psutil.Process().memory_info().rss / 1024**2)
            
            if torch.cuda.is_available():
                self.gpu_samples.append(
                    torch.cuda.memory_allocated() / 1024**2
                )
            time.sleep(0.1)
    
    def get_stats(self) -> Dict[str, float]:
        return {
            'peak_cpu': max(self.cpu_samples) if self.cpu_samples else 0,
            'avg_cpu': np.mean(self.cpu_samples) if self.cpu_samples else 0,
            'peak_ram': max(self.ram_samples) if self.ram_samples else 0,
            'avg_ram': np.mean(self.ram_samples) if self.ram_samples else 0,
            'peak_gpu': max(self.gpu_samples) if self.gpu_samples else 0,
            'avg_gpu': np.mean(self.gpu_samples) if self.gpu_samples else 0,
        }

class ConcurrencyTracker:
    """Track concurrent requests and queue depth"""
    def __init__(self):
        self.lock = threading.Lock()
        self.current_active = 0
        self.max_active = 0
        self.current_queued = 0
        self.max_queued = 0
        self.active_samples = []
        
    def queue_request(self):
        with self.lock:
            self.current_queued += 1
            self.max_queued = max(self.max_queued, self.current_queued)
    
    def start_processing(self):
        with self.lock:
            self.current_queued -= 1
            self.current_active += 1
            self.max_active = max(self.max_active, self.current_active)
            self.active_samples.append(self.current_active)
    
    def finish_processing(self):
        with self.lock:
            self.current_active -= 1
            self.active_samples.append(self.current_active)
    
    def get_stats(self):
        with self.lock:
            return {
                'max_active': self.max_active,
                'avg_active': np.mean(self.active_samples) if self.active_samples else 0,
                'max_queued': self.max_queued
            }

class TTSRequestQueue:
    """Manages TTS generation requests with thread-safe queueing"""
    def __init__(self, model: ChatterboxTTS, num_workers: int, concurrency_tracker: ConcurrencyTracker):
        self.model = model
        self.request_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.num_workers = num_workers
        self.concurrency_tracker = concurrency_tracker
        self.workers = []
        self.stop_flag = threading.Event()
        
    def start_workers(self):
        """Start worker threads"""
        for i in range(self.num_workers):
            worker = threading.Thread(
                target=self._worker,
                args=(i,),
                daemon=True
            )
            worker.start()
            self.workers.append(worker)
    
    def _worker(self, worker_id: int):
        """Worker thread that processes requests"""
        while not self.stop_flag.is_set():
            try:
                # Get request from queue with timeout
                request_data = self.request_queue.get(timeout=1.0)
                if request_data is None:  # Poison pill
                    break
                
                request_id, text, queue_start_time = request_data
                
                # Record queue time
                queue_time = time.time() - queue_start_time
                
                # Start processing
                self.concurrency_tracker.start_processing()
                processing_start = time.time()
                
                metrics = RequestMetrics(
                    request_id=request_id,
                    text=text,
                    text_length=len(text),
                    start_time=queue_start_time,
                    end_time=0,
                    duration=0,
                    queue_time=queue_time,
                    processing_time=0,
                    success=False
                )
                
                try:
                    # Split text into chunks if needed
                    chunks = self._split_text(text)
                    metrics.num_chunks = len(chunks)
                    
                    # Generate audio
                    audios = self.model.generate(
                        chunks,
                        audio_prompt_path=AUDIO_PROMPT_PATH,
                        exaggeration=0.5,
                        min_p=0.1,
                        top_p=0.8,
                    )
                    
                    # Calculate audio duration
                    full_audio = torch.cat(audios, dim=-1)
                    audio_duration = full_audio.shape[-1] / self.model.sr
                    
                    metrics.audio_duration = audio_duration
                    metrics.success = True
                    
                except Exception as e:
                    metrics.error_message = str(e)
                finally:
                    # Record timing
                    metrics.end_time = time.time()
                    metrics.processing_time = metrics.end_time - processing_start
                    metrics.duration = metrics.end_time - metrics.start_time
                    
                    # Mark processing complete
                    self.concurrency_tracker.finish_processing()
                    self.request_queue.task_done()
                    self.result_queue.put(metrics)
                    
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[ERROR] Worker {worker_id} error: {e}")
    
    def _split_text(self, text: str) -> List[str]:
        """Split text into chunks"""
        if len(text) <= MAX_CHUNK_SIZE:
            return [text]
        
        # Simple word-based splitting
        words = text.split()
        chunks = []
        current_chunk = []
        current_length = 0
        
        for word in words:
            word_len = len(word) + 1  # +1 for space
            if current_length + word_len > MAX_CHUNK_SIZE and current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = [word]
                current_length = word_len
            else:
                current_chunk.append(word)
                current_length += word_len
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks
    
    def submit_request(self, request_id: int, text: str):
        """Submit a request to the queue"""
        self.concurrency_tracker.queue_request()
        self.request_queue.put((request_id, text, time.time()))
    
    def shutdown(self):
        """Shutdown workers"""
        self.stop_flag.set()
        # Send poison pills
        for _ in range(self.num_workers):
            self.request_queue.put(None)
        # Wait for workers
        for worker in self.workers:
            worker.join(timeout=5.0)
    
    def get_all_results(self, expected_count: int) -> List[RequestMetrics]:
        """Collect all results"""
        results = []
        while len(results) < expected_count:
            try:
                result = self.result_queue.get(timeout=1.0)
                results.append(result)
            except queue.Empty:
                if self.request_queue.empty() and all(not w.is_alive() for w in self.workers):
                    break
        return results

def load_and_prepare_texts(text_path: str) -> List[str]:
    """Load and prepare texts for requests"""
    with open(text_path, "r") as f:
        text = f.read()
    
    # Clean text
    lines = [line.strip() for line in text.split("\n") 
             if not line.startswith("#") and line.strip()]
    
    # Create varied length texts
    texts = []
    for line in lines:
        # Original line
        texts.append(line)
        
        # Short version (first ~200 chars)
        if len(line) > 200:
            words = line.split()
            short = []
            length = 0
            for word in words:
                if length + len(word) > 200:
                    break
                short.append(word)
                length += len(word) + 1
            texts.append(" ".join(short))
        
        # Long version (duplicate for longer generation)
        if len(line) < 300:
            texts.append(line + " " + line)
    
    return texts

def calculate_metrics(
    request_results: List[RequestMetrics],
    resource_stats: Dict,
    concurrency_stats: Dict,
    total_time: float,
    model_load_time: float
) -> ConcurrentBenchmarkMetrics:
    """Calculate aggregate metrics"""
    
    metrics = ConcurrentBenchmarkMetrics()
    
    # Basic counts
    metrics.total_requests = len(request_results)
    metrics.successful_requests = sum(1 for r in request_results if r.success)
    metrics.failed_requests = metrics.total_requests - metrics.successful_requests
    metrics.success_rate = metrics.successful_requests / metrics.total_requests if metrics.total_requests > 0 else 0
    
    # Timing metrics
    metrics.total_benchmark_time = total_time
    metrics.model_load_time = model_load_time
    
    successful_durations = [r.duration for r in request_results if r.success]
    if successful_durations:
        metrics.avg_request_duration = np.mean(successful_durations)
        metrics.min_request_duration = np.min(successful_durations)
        metrics.max_request_duration = np.max(successful_durations)
        metrics.median_request_duration = np.median(successful_durations)
        metrics.p95_request_duration = np.percentile(successful_durations, 95)
        metrics.p99_request_duration = np.percentile(successful_durations, 99)
    
    # Queue metrics
    queue_times = [r.queue_time for r in request_results if r.success]
    processing_times = [r.processing_time for r in request_results if r.success]
    if queue_times:
        metrics.avg_queue_time = np.mean(queue_times)
        metrics.max_queue_time = np.max(queue_times)
    if processing_times:
        metrics.avg_processing_time = np.mean(processing_times)
    
    # Throughput
    metrics.requests_per_second = metrics.successful_requests / total_time if total_time > 0 else 0
    
    successful_audio = [r.audio_duration for r in request_results if r.success and r.audio_duration]
    if successful_audio:
        metrics.total_audio_generated_seconds = sum(successful_audio)
        metrics.avg_audio_seconds_per_second = metrics.total_audio_generated_seconds / total_time if total_time > 0 else 0
    
    # Resource metrics
    metrics.peak_cpu_percent = resource_stats['peak_cpu']
    metrics.avg_cpu_percent = resource_stats['avg_cpu']
    metrics.peak_ram_mb = resource_stats['peak_ram']
    metrics.avg_ram_mb = resource_stats['avg_ram']
    metrics.peak_gpu_memory_mb = resource_stats['peak_gpu']
    metrics.avg_gpu_memory_mb = resource_stats['avg_gpu']
    
    # Concurrency
    metrics.max_concurrent_requests = concurrency_stats['max_active']
    metrics.avg_concurrent_requests = concurrency_stats['avg_active']
    metrics.max_queue_depth = concurrency_stats['max_queued']
    
    # Store detailed request data
    metrics.request_details = [
        {
            'request_id': r.request_id,
            'duration': r.duration,
            'queue_time': r.queue_time,
            'processing_time': r.processing_time,
            'success': r.success,
            'error': r.error_message,
            'text_length': r.text_length,
            'num_chunks': r.num_chunks,
            'audio_duration': r.audio_duration,
        }
        for r in request_results
    ]
    
    return metrics

def print_metrics_report(metrics: ConcurrentBenchmarkMetrics):
    """Print comprehensive metrics report"""
    print("\n" + "="*80)
    print("CONCURRENT LOAD BENCHMARK RESULTS (IN-PROCESS)")
    print("="*80)
    
    print("\n📊 REQUEST SUMMARY")
    print(f"  Total Requests:           {metrics.total_requests}")
    print(f"  Successful:               {metrics.successful_requests} ({metrics.success_rate*100:.1f}%)")
    print(f"  Failed:                   {metrics.failed_requests}")
    print(f"  Total Time:               {metrics.total_benchmark_time:.2f}s")
    print(f"  Model Load Time:          {metrics.model_load_time:.2f}s")
    
    print("\n⚡ THROUGHPUT")
    print(f"  Requests/Second:          {metrics.requests_per_second:.2f} req/s")
    print(f"  Audio Generated:          {metrics.total_audio_generated_seconds:.2f}s")
    print(f"  Audio Seconds/Second:     {metrics.avg_audio_seconds_per_second:.2f}s/s")
    
    print("\n⏱️  LATENCY DISTRIBUTION (End-to-End)")
    print(f"  Average Duration:         {metrics.avg_request_duration:.3f}s")
    print(f"  Median Duration:          {metrics.median_request_duration:.3f}s")
    print(f"  Min Duration:             {metrics.min_request_duration:.3f}s")
    print(f"  Max Duration:             {metrics.max_request_duration:.3f}s")
    print(f"  P95 Duration:             {metrics.p95_request_duration:.3f}s")
    print(f"  P99 Duration:             {metrics.p99_request_duration:.3f}s")
    
    print("\n⏳ QUEUE & PROCESSING TIME")
    print(f"  Avg Queue Time:           {metrics.avg_queue_time:.3f}s")
    print(f"  Max Queue Time:           {metrics.max_queue_time:.3f}s")
    print(f"  Avg Processing Time:      {metrics.avg_processing_time:.3f}s")
    
    print("\n🔄 CONCURRENCY")
    print(f"  Max Concurrent Active:    {metrics.max_concurrent_requests}")
    print(f"  Avg Concurrent Active:    {metrics.avg_concurrent_requests:.1f}")
    print(f"  Max Queue Depth:          {metrics.max_queue_depth}")
    
    print("\n💾 RESOURCE UTILIZATION")
    print(f"  Peak CPU:                 {metrics.peak_cpu_percent:.1f}%")
    print(f"  Avg CPU:                  {metrics.avg_cpu_percent:.1f}%")
    print(f"  Peak RAM:                 {metrics.peak_ram_mb:.1f} MB")
    print(f"  Avg RAM:                  {metrics.avg_ram_mb:.1f} MB")
    print(f"  Peak GPU Memory:          {metrics.peak_gpu_memory_mb:.1f} MB")
    print(f"  Avg GPU Memory:           {metrics.avg_gpu_memory_mb:.1f} MB")
    
    print("\n" + "="*80 + "\n")

def main():
    """Main benchmark execution"""
    print("[BENCHMARK] Concurrent Load Test Starting (In-Process)...")
    print(f"[BENCHMARK] Target: {NUM_CONCURRENT_REQUESTS} concurrent requests")
    print(f"[BENCHMARK] Workers: {NUM_WORKER_THREADS} threads")
    print(f"[BENCHMARK] Batch Size: {BATCH_SIZE}")
    
    # Load texts
    print(f"[BENCHMARK] Loading text from {TEXT_PATH}...")
    texts = load_and_prepare_texts(TEXT_PATH)
    print(f"[BENCHMARK] Prepared {len(texts)} text variants")
    
    # Load model
    print("[BENCHMARK] Loading ChatterboxTTS model...")
    model_load_start = time.time()
    model = ChatterboxTTS.from_pretrained(
        max_batch_size=BATCH_SIZE,
        max_model_len=MAX_CHUNK_SIZE * 3,
    )
    model_load_time = time.time() - model_load_start
    print(f"[BENCHMARK] Model loaded in {model_load_time:.2f}s")
    
    # Warmup
    print("[BENCHMARK] Running warmup...")
    _ = model.generate(
        ["The quick brown fox jumps over the lazy dog."],
        audio_prompt_path=AUDIO_PROMPT_PATH,
        exaggeration=0.5,
        min_p=0.1,
        top_p=0.8,
    )
    print("[BENCHMARK] Warmup complete")
    
    # Reset GPU stats
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    
    # Setup monitoring
    monitor = ResourceMonitor()
    concurrency_tracker = ConcurrencyTracker()
    
    # Create request queue
    request_queue = TTSRequestQueue(model, NUM_WORKER_THREADS, concurrency_tracker)
    request_queue.start_workers()
    
    # Start monitoring
    monitor.start()
    
    # Submit all requests
    print(f"[BENCHMARK] Submitting {NUM_CONCURRENT_REQUESTS} requests...")
    benchmark_start = time.time()
    
    for i in range(NUM_CONCURRENT_REQUESTS):
        text = texts[i % len(texts)]
        request_queue.submit_request(i, text)
    
    print("[BENCHMARK] All requests submitted, waiting for completion...")
    
    # Wait for all requests to complete
    request_queue.request_queue.join()
    
    # Collect results
    results = request_queue.get_all_results(NUM_CONCURRENT_REQUESTS)
    
    benchmark_time = time.time() - benchmark_start
    
    # Shutdown
    request_queue.shutdown()
    monitor.stop()
    
    print(f"[BENCHMARK] All requests completed in {benchmark_time:.2f}s")
    
    # Get stats
    resource_stats = monitor.get_stats()
    concurrency_stats = concurrency_tracker.get_stats()
    
    # Calculate metrics
    metrics = calculate_metrics(
        results,
        resource_stats,
        concurrency_stats,
        benchmark_time,
        model_load_time
    )
    
    # Print report
    print_metrics_report(metrics)
    
    # Save results
    output_file = f"concurrent_benchmark_inprocess_{NUM_CONCURRENT_REQUESTS}req.json"
    with open(output_file, "w") as f:
        json.dump(asdict(metrics), f, indent=2)
    print(f"[BENCHMARK] Detailed metrics saved to {output_file}")
    
    # Save failed requests
    failed_requests = [r for r in results if not r.success]
    if failed_requests:
        failed_file = f"failed_requests_inprocess_{NUM_CONCURRENT_REQUESTS}req.json"
        with open(failed_file, "w") as f:
            json.dump([
                {
                    'request_id': r.request_id,
                    'error': r.error_message,
                    'text_length': r.text_length,
                }
                for r in failed_requests
            ], f, indent=2)
        print(f"[BENCHMARK] Failed request details saved to {failed_file}")
    
    # Cleanup
    model.shutdown()
    print("[BENCHMARK] Benchmark complete!")

if __name__ == "__main__":
    main()