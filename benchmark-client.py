#!/usr/bin/env python3
"""
Benchmark Client for ChatterboxTTS vLLM Server
Sends 100 simultaneous requests and collects comprehensive metrics.

Usage:
    python benchmark_client.py

Requirements:
    pip install aiohttp numpy psutil requests
"""

import asyncio
import aiohttp
import json
import time
import base64
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from dataclasses import dataclass, asdict
import statistics
import requests

# Configuration
SERVER_URL = "http://localhost:8000"
METRICS_URL = f"{SERVER_URL}/metrics"
NUM_REQUESTS = 100
DOCS_DIR = "docs"
OUTPUT_FILE = "BATCHED_benchmark_results.json"

# TTS parameters
TTS_PARAMS = {
    "exaggeration": 0.5,
    "min_p": 0.1,
    "top_p": 0.8,
}

@dataclass
class RequestMetrics:
    request_id: int
    text: str
    text_length: int
    audio_prompt: str
    
    # Timing metrics
    start_time: float
    end_time: float
    ttfa: float  # Time to First Audio
    e2e_latency: float  # End-to-End Latency
    
    # Result
    success: bool
    error: str = None
    audio_duration: float = 0.0
    rtfx: float = 0.0  # Real-Time Factor

@dataclass
class SystemMetrics:
    timestamp: float
    gpu_utilization: float
    kv_cache_utilization: float
    total_preemptions: int
    requests_running: int
    requests_waiting: int

class PrometheusMetricsParser:
    """Parse Prometheus metrics from vLLM"""
    
    @staticmethod
    def parse_metrics(metrics_text: str) -> Dict:
        """Parse Prometheus text format metrics"""
        metrics = {}
        for line in metrics_text.split('\n'):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            try:
                # Simple parsing for metrics without labels
                if '{' not in line:
                    parts = line.split()
                    if len(parts) == 2:
                        key, value = parts
                        metrics[key] = float(value)
                else:
                    # Parse metrics with labels
                    metric_name = line.split('{')[0]
                    value = float(line.split()[-1])
                    metrics[metric_name] = value
            except:
                continue
        
        return metrics

async def load_test_data() -> Tuple[List[str], List[str]]:
    """Load text files and audio prompts from docs directory"""
    docs_path = Path(DOCS_DIR)
    
    # Find all text files
    text_files = sorted(docs_path.glob("benchmark-text-*.txt"))
    if not text_files:
        text_files = sorted(docs_path.glob("*.txt"))
    
    # Find all audio files
    audio_files = sorted(docs_path.glob("audio-sample-*.mp3"))
    if not audio_files:
        audio_files = sorted(docs_path.glob("*.mp3"))
    
    if not text_files or not audio_files:
        raise FileNotFoundError(f"No text or audio files found in {DOCS_DIR}")
    
    # Load texts
    texts = []
    for text_file in text_files:
        with open(text_file, 'r') as f:
            content = f.read()
            # Remove lines starting with #
            content = "\n".join([line for line in content.split("\n") 
                               if not line.startswith("#")])
            # Split into sentences
            sentences = [s.strip() for s in content.split('.') if s.strip()]
            texts.extend(sentences)
    
    # Get audio file paths
    audio_paths = [str(f) for f in audio_files]
    
    print(f"✓ Loaded {len(texts)} text samples from {len(text_files)} files")
    print(f"✓ Loaded {len(audio_paths)} audio prompts")
    
    return texts, audio_paths

def encode_audio_to_base64(audio_path: str) -> str:
    """Encode audio file to base64"""
    with open(audio_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')

async def fetch_system_metrics() -> SystemMetrics:
    """Fetch current system metrics from Prometheus endpoint"""
    try:
        response = requests.get(METRICS_URL, timeout=10)  # Increased timeout
        if response.status_code == 200:
            parser = PrometheusMetricsParser()
            metrics = parser.parse_metrics(response.text)
            
            return SystemMetrics(
                timestamp=time.time(),
                gpu_utilization=metrics.get('vllm_gpu_cache_usage_perc', 0.0),
                kv_cache_utilization=metrics.get('vllm_kv_cache_usage_perc', 0.0),
                total_preemptions=int(metrics.get('vllm_num_preemptions_total', 0)),
                requests_running=int(metrics.get('vllm_num_requests_running', 0)),
                requests_waiting=int(metrics.get('vllm_num_requests_waiting', 0)),
            )
    except requests.exceptions.Timeout:
        # Server is too busy, return empty metrics (expected during heavy load)
        pass
    except Exception as e:
        # Only print non-timeout errors
        if "timed out" not in str(e).lower():
            print(f"Warning: Could not fetch metrics: {e}")
    
    return SystemMetrics(
        timestamp=time.time(),
        gpu_utilization=0.0,
        kv_cache_utilization=0.0,
        total_preemptions=0,
        requests_running=0,
        requests_waiting=0,
    )

async def send_tts_request(
    session: aiohttp.ClientSession,
    request_id: int,
    text: str,
    audio_prompt_path: str,
    audio_prompt_b64: str,
) -> RequestMetrics:
    """Send a single TTS request and measure metrics"""
    
    start_time = time.time()
    ttfa = None
    success = False
    error = None
    audio_duration = 0.0
    
    try:
        # Prepare request payload (matching our FastAPI server format)
        payload = {
            "model": "chatterbox-tts",
            "input": text,
            "voice": audio_prompt_b64,  # Base64 encoded audio
            "exaggeration": TTS_PARAMS["exaggeration"],
            "min_p": TTS_PARAMS["min_p"],
            "top_p": TTS_PARAMS["top_p"],
            "response_format": "mp3"
        }
        
        # Send request
        async with session.post(
            f"{SERVER_URL}/v1/audio/speech",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=300)  # 5 minute timeout
        ) as response:
            
            # Time to first byte (approximates TTFA)
            chunk_count = 0
            audio_chunks = []
            
            async for chunk in response.content.iter_any():
                if chunk:
                    if ttfa is None:
                        ttfa = time.time() - start_time
                    audio_chunks.append(chunk)
                    chunk_count += 1
            
            if response.status == 200:
                success = True
                # Estimate audio duration (rough calculation)
                total_bytes = sum(len(chunk) for chunk in audio_chunks)
                # Assuming MP3 at 128kbps: bytes * 8 / (128000) = seconds
                audio_duration = (total_bytes * 8) / 128000
            else:
                error = f"HTTP {response.status}: {await response.text()}"
    
    except asyncio.TimeoutError:
        error = "Request timeout"
    except Exception as e:
        error = str(e)
    
    end_time = time.time()
    e2e_latency = end_time - start_time
    
    if ttfa is None:
        ttfa = e2e_latency  # If no streaming, TTFA = E2E
    
    # Calculate RTFX (Real-Time Factor)
    rtfx = audio_duration / e2e_latency if e2e_latency > 0 and audio_duration > 0 else 0.0
    
    return RequestMetrics(
        request_id=request_id,
        text=text[:100] + "..." if len(text) > 100 else text,
        text_length=len(text),
        audio_prompt=Path(audio_prompt_path).name,
        start_time=start_time,
        end_time=end_time,
        ttfa=ttfa,
        e2e_latency=e2e_latency,
        success=success,
        error=error,
        audio_duration=audio_duration,
        rtfx=rtfx,
    )

async def monitor_metrics(metrics_list: List[SystemMetrics], stop_event: asyncio.Event):
    """Continuously monitor system metrics during benchmark"""
    consecutive_failures = 0
    while not stop_event.is_set():
        metrics = await fetch_system_metrics()
        
        # Only add non-empty metrics (server responded)
        if metrics.gpu_utilization > 0 or metrics.requests_running > 0:
            metrics_list.append(metrics)
            consecutive_failures = 0
        else:
            consecutive_failures += 1
            # If server is too busy, add placeholder to maintain timeline
            if consecutive_failures > 3:
                metrics_list.append(metrics)  # Add zeros to show server was unresponsive
        
        await asyncio.sleep(2)  # Sample every 2 seconds to reduce load

async def run_benchmark():
    """Run the complete benchmark"""
    
    print("=" * 80)
    print("ChatterboxTTS vLLM Benchmark - 100 Simultaneous Requests")
    print("=" * 80)
    print(f"Server: {SERVER_URL}")
    print(f"Number of requests: {NUM_REQUESTS}")
    print(f"Parameters: {TTS_PARAMS}")
    print("=" * 80)
    
    # Load test data
    print("\nLoading test data...")
    texts, audio_paths = await load_test_data()
    
    # Prepare audio prompts (encode to base64)
    print("Encoding audio prompts...")
    audio_prompts_b64 = {path: encode_audio_to_base64(path) for path in audio_paths}
    
    # Prepare requests (cycle through texts and audio)
    print(f"Preparing {NUM_REQUESTS} requests...")
    requests_data = []
    for i in range(NUM_REQUESTS):
        text = texts[i % len(texts)]
        audio_path = audio_paths[i % len(audio_paths)]
        requests_data.append((i, text, audio_path, audio_prompts_b64[audio_path]))
    
    # Get baseline metrics before benchmark
    print("\nFetching baseline metrics...")
    baseline_metrics = await fetch_system_metrics()
    
    # Start metrics monitoring
    system_metrics = []
    stop_monitoring = asyncio.Event()
    monitor_task = asyncio.create_task(monitor_metrics(system_metrics, stop_monitoring))
    
    # Run benchmark
    print(f"\n{'=' * 80}")
    print("Starting benchmark - sending all requests simultaneously...")
    print(f"{'=' * 80}\n")
    
    benchmark_start = time.time()
    completed = 0
    
    # Create session and send all requests simultaneously
    async with aiohttp.ClientSession() as session:
        tasks = [
            send_tts_request(session, req_id, text, audio_path, audio_b64)
            for req_id, text, audio_path, audio_b64 in requests_data
        ]
        
        # Wait for all requests to complete with progress updates
        print("Progress: ", end="", flush=True)
        results = []
        for task in asyncio.as_completed(tasks):
            result = await task
            results.append(result)
            completed += 1
            if completed % 10 == 0 or completed == NUM_REQUESTS:
                elapsed = time.time() - benchmark_start
                print(f"{completed}/{NUM_REQUESTS} ({elapsed:.1f}s) ", end="", flush=True)
        print()  # New line after progress
    
    benchmark_end = time.time()
    total_benchmark_time = benchmark_end - benchmark_start
    
    # Stop metrics monitoring
    stop_monitoring.set()
    await monitor_task
    
    # Get final metrics
    final_metrics = await fetch_system_metrics()
    
    # Process results
    print(f"\n{'=' * 80}")
    print("Benchmark Complete - Processing Results...")
    print(f"{'=' * 80}\n")
    
    successful_results = [r for r in results if isinstance(r, RequestMetrics) and r.success]
    failed_results = [r for r in results if isinstance(r, RequestMetrics) and not r.success]
    exception_results = [r for r in results if not isinstance(r, RequestMetrics)]
    
    # Calculate statistics
    success_rate = len(successful_results) / NUM_REQUESTS * 100
    
    if successful_results:
        ttfa_values = [r.ttfa for r in successful_results]
        e2e_values = [r.e2e_latency for r in successful_results]
        rtfx_values = [r.rtfx for r in successful_results if r.rtfx > 0]
        audio_durations = [r.audio_duration for r in successful_results if r.audio_duration > 0]
        
        def percentile(values, p):
            return np.percentile(values, p) if values else 0.0
        
        # Latency metrics
        latency_stats = {
            "ttfa": {
                "mean": statistics.mean(ttfa_values),
                "median": statistics.median(ttfa_values),
                "min": min(ttfa_values),
                "max": max(ttfa_values),
                "p50": percentile(ttfa_values, 50),
                "p95": percentile(ttfa_values, 95),
                "p99": percentile(ttfa_values, 99),
                "stddev": statistics.stdev(ttfa_values) if len(ttfa_values) > 1 else 0.0,
            },
            "e2e_latency": {
                "mean": statistics.mean(e2e_values),
                "median": statistics.median(e2e_values),
                "min": min(e2e_values),
                "max": max(e2e_values),
                "p50": percentile(e2e_values, 50),
                "p95": percentile(e2e_values, 95),
                "p99": percentile(e2e_values, 99),
                "stddev": statistics.stdev(e2e_values) if len(e2e_values) > 1 else 0.0,
            },
        }
        
        # Throughput metrics
        total_audio_duration = sum(audio_durations)
        requests_per_second = len(successful_results) / total_benchmark_time
        
        throughput_stats = {
            "rtfx": {
                "mean": statistics.mean(rtfx_values) if rtfx_values else 0.0,
                "median": statistics.median(rtfx_values) if rtfx_values else 0.0,
                "min": min(rtfx_values) if rtfx_values else 0.0,
                "max": max(rtfx_values) if rtfx_values else 0.0,
            },
            "requests_per_second": requests_per_second,
            "total_audio_generated_seconds": total_audio_duration,
            "audio_throughput_rtfx": total_audio_duration / total_benchmark_time if total_benchmark_time > 0 else 0.0,
        }
        
        # Resource utilization
        if system_metrics:
            gpu_utils = [m.gpu_utilization for m in system_metrics]
            kv_cache_utils = [m.kv_cache_utilization for m in system_metrics]
            
            resource_stats = {
                "gpu_utilization": {
                    "mean": statistics.mean(gpu_utils),
                    "max": max(gpu_utils),
                    "min": min(gpu_utils),
                },
                "kv_cache_utilization": {
                    "mean": statistics.mean(kv_cache_utils),
                    "max": max(kv_cache_utils),
                    "min": min(kv_cache_utils),
                },
                "total_preemptions": final_metrics.total_preemptions - baseline_metrics.total_preemptions,
                "peak_requests_running": max([m.requests_running for m in system_metrics]),
                "peak_requests_waiting": max([m.requests_waiting for m in system_metrics]),
            }
        else:
            resource_stats = {
                "gpu_utilization": {"mean": 0, "max": 0, "min": 0},
                "kv_cache_utilization": {"mean": 0, "max": 0, "min": 0},
                "total_preemptions": 0,
                "peak_requests_running": 0,
                "peak_requests_waiting": 0,
            }
    else:
        latency_stats = {}
        throughput_stats = {}
        resource_stats = {}
    
    # Compile final results
    final_results = {
        "benchmark_config": {
            "server_url": SERVER_URL,
            "num_requests": NUM_REQUESTS,
            "tts_params": TTS_PARAMS,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "summary": {
            "total_benchmark_time_seconds": total_benchmark_time,
            "successful_requests": len(successful_results),
            "failed_requests": len(failed_results),
            "exception_requests": len(exception_results),
            "success_rate_percent": success_rate,
        },
        "latency_metrics": latency_stats,
        "throughput_metrics": throughput_stats,
        "resource_utilization": resource_stats,
        "individual_requests": [asdict(r) for r in results if isinstance(r, RequestMetrics)],
        "system_metrics_timeline": [asdict(m) for m in system_metrics],
        "failed_requests_details": [
            {
                "request_id": r.request_id,
                "error": r.error,
                "text_length": r.text_length,
            }
            for r in failed_results
        ],
    }
    
    # Save to JSON
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(final_results, f, indent=2)
    
    # Print summary
    print(f"\n{'=' * 80}")
    print("BENCHMARK RESULTS SUMMARY")
    print(f"{'=' * 80}\n")
    
    print(f"Total Time: {total_benchmark_time:.2f}s")
    print(f"Success Rate: {success_rate:.1f}% ({len(successful_results)}/{NUM_REQUESTS})")
    print(f"Failed Requests: {len(failed_results)}")
    print(f"Exception Requests: {len(exception_results)}")
    
    if successful_results:
        print(f"\n--- LATENCY METRICS ---")
        print(f"Time to First Audio (TTFA):")
        print(f"  Mean: {latency_stats['ttfa']['mean']:.3f}s")
        print(f"  P50:  {latency_stats['ttfa']['p50']:.3f}s")
        print(f"  P95:  {latency_stats['ttfa']['p95']:.3f}s")
        print(f"  P99:  {latency_stats['ttfa']['p99']:.3f}s")
        print(f"  Max:  {latency_stats['ttfa']['max']:.3f}s")
        
        print(f"\nEnd-to-End Latency (E2E):")
        print(f"  Mean: {latency_stats['e2e_latency']['mean']:.3f}s")
        print(f"  P50:  {latency_stats['e2e_latency']['p50']:.3f}s")
        print(f"  P95:  {latency_stats['e2e_latency']['p95']:.3f}s")
        print(f"  P99:  {latency_stats['e2e_latency']['p99']:.3f}s")
        print(f"  Max:  {latency_stats['e2e_latency']['max']:.3f}s")
        
        print(f"\n--- THROUGHPUT METRICS ---")
        print(f"Requests Per Second: {throughput_stats['requests_per_second']:.2f}")
        print(f"Audio Throughput (RTFX): {throughput_stats['audio_throughput_rtfx']:.2f}x")
        print(f"Mean RTFX per request: {throughput_stats['rtfx']['mean']:.2f}x")
        print(f"Total Audio Generated: {throughput_stats['total_audio_generated_seconds']:.1f}s")
        
        print(f"\n--- RESOURCE UTILIZATION ---")
        print(f"GPU Utilization: {resource_stats['gpu_utilization']['mean']:.1f}% (avg), {resource_stats['gpu_utilization']['max']:.1f}% (max)")
        print(f"KV Cache Utilization: {resource_stats['kv_cache_utilization']['mean']:.1f}% (avg), {resource_stats['kv_cache_utilization']['max']:.1f}% (max)")
        print(f"Total Preemptions: {resource_stats['total_preemptions']}")
        print(f"Peak Requests Running: {resource_stats['peak_requests_running']}")
        print(f"Peak Requests Waiting: {resource_stats['peak_requests_waiting']}")
    
    print(f"\n{'=' * 80}")
    print(f"✓ Detailed results saved to: {OUTPUT_FILE}")
    print(f"{'=' * 80}\n")

def main():
    try:
        asyncio.run(run_benchmark())
    except KeyboardInterrupt:
        print("\n\nBenchmark interrupted by user.")
    except Exception as e:
        print(f"\n❌ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()