#!/usr/bin/env python3
import time
import re
from typing import List, Dict, Any
import torch
import torchaudio as ta
import numpy as np
import psutil
import threading
from dataclasses import dataclass, asdict
from chatterbox_vllm.tts import ChatterboxTTS

AUDIO_PROMPT_PATH = "docs/audio-sample-sampu.wav"
TEXT_PATH = "docs/benchmark-text-2.txt"
MAX_CHUNK_SIZE = 400  # characters
BATCH_SIZE = 40

@dataclass
class BenchmarkMetrics:
    # Timing metrics
    model_load_time: float = 0.0
    time_to_first_audio: float = 0.0
    total_inference_time: float = 0.0
    total_time: float = 0.0
    warmup_time: float = 0.0
    
    # Performance metrics
    real_time_factor: float = 0.0
    audio_seconds_per_second: float = 0.0
    chunks_per_second: float = 0.0
    
    # Resource metrics
    peak_gpu_memory_mb: float = 0.0
    peak_cpu_percent: float = 0.0
    peak_ram_mb: float = 0.0
    avg_gpu_memory_mb: float = 0.0
    avg_cpu_percent: float = 0.0
    
    # Audio metrics
    total_audio_duration_seconds: float = 0.0
    num_chunks: int = 0
    total_characters: int = 0
    
    # Quality metrics (basic)
    audio_samples: int = 0
    sample_rate: int = 0
    has_clipping: bool = False
    max_amplitude: float = 0.0
    rms_level: float = 0.0

class ResourceMonitor:
    """Monitor CPU, RAM, and GPU usage in background thread"""
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
            # CPU and RAM
            self.cpu_samples.append(psutil.cpu_percent(interval=0.1))
            self.ram_samples.append(psutil.Process().memory_info().rss / 1024**2)
            
            # GPU memory (if available)
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
            'peak_gpu': max(self.gpu_samples) if self.gpu_samples else 0,
            'avg_gpu': np.mean(self.gpu_samples) if self.gpu_samples else 0,
        }

def analyze_audio_quality(audio_tensor: torch.Tensor, sample_rate: int) -> Dict[str, Any]:
    """Analyze audio for basic quality metrics"""
    audio_np = audio_tensor.cpu().numpy()
    
    # Flatten if multi-channel
    if audio_np.ndim > 1:
        audio_np = audio_np.flatten()
    
    max_amp = np.abs(audio_np).max()
    rms = np.sqrt(np.mean(audio_np**2))
    
    # Check for clipping (values at or near ±1.0)
    clipping_threshold = 0.99
    has_clipping = np.any(np.abs(audio_np) >= clipping_threshold)
    
    # Check for abrupt cutoffs (sudden drop to near-zero at end)
    end_samples = min(int(0.05 * sample_rate), len(audio_np))  # Last 50ms
    if end_samples > 0:
        end_rms = np.sqrt(np.mean(audio_np[-end_samples:]**2))
        abrupt_cutoff = end_rms < 0.01 * rms if rms > 0 else False
    else:
        abrupt_cutoff = False
    
    # Detect silence gaps (potential glitches)
    silence_threshold = 0.01
    silence_mask = np.abs(audio_np) < silence_threshold
    
    # Find consecutive silence regions
    silence_changes = np.diff(silence_mask.astype(int))
    silence_starts = np.where(silence_changes == 1)[0]
    silence_ends = np.where(silence_changes == -1)[0]
    
    # Count significant silence gaps (> 100ms)
    min_gap_samples = int(0.1 * sample_rate)
    if len(silence_starts) > 0 and len(silence_ends) > 0:
        if silence_starts[0] > silence_ends[0]:
            silence_ends = silence_ends[1:]
        if len(silence_starts) > len(silence_ends):
            silence_starts = silence_starts[:-1]
        
        gap_lengths = silence_ends - silence_starts
        significant_gaps = np.sum(gap_lengths > min_gap_samples)
    else:
        significant_gaps = 0
    
    return {
        'max_amplitude': float(max_amp),
        'rms_level': float(rms),
        'has_clipping': bool(has_clipping),
        'abrupt_cutoff': bool(abrupt_cutoff),
        'silence_gaps': int(significant_gaps),
        'duration_seconds': len(audio_np) / sample_rate
    }

def split_text_by_sentence(text: str) -> List[str]:
    sentences = text.split(". ")
    n_chunks_needed = len(text) // MAX_CHUNK_SIZE + 1
    approx_chunk_size = len(text) // n_chunks_needed
    
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        sentence = " ".join(sentence.split())
        sentence = sentence.strip()
        
        if current_length + len(sentence) > approx_chunk_size:
            chunks.append(". ".join(current_chunk))
            current_chunk = [sentence]
            current_length = len(sentence)
        else:
            current_chunk.append(sentence)
            current_length += len(sentence)
    
    if current_chunk:
        chunks.append(". ".join(current_chunk))
    
    chunks = [chunk + "." if re.match(r"[a-zA-Z0-9]", chunk[-1]) else chunk 
              for chunk in chunks if len(chunk) > 0]
    
    return chunks

def print_metrics_report(metrics: BenchmarkMetrics):
    """Print formatted metrics report"""
    print("\n" + "="*70)
    print("BENCHMARK RESULTS")
    print("="*70)
    
    print("\n📊 TIMING METRICS")
    print(f"  Model Load Time:          {metrics.model_load_time:.2f}s")
    print(f"  Warmup Time:              {metrics.warmup_time:.2f}s")
    print(f"  Time-To-First-Audio:      {metrics.time_to_first_audio:.2f}s")
    print(f"  Total Inference Time:     {metrics.total_inference_time:.2f}s")
    print(f"  Total Time:               {metrics.total_time:.2f}s")
    
    print("\n⚡ PERFORMANCE METRICS")
    print(f"  Real-Time Factor (RTF):   {metrics.real_time_factor:.3f}x")
    print(f"  Audio Seconds/Second:     {metrics.audio_seconds_per_second:.2f}s/s")
    print(f"  Chunks Processed/Second:  {metrics.chunks_per_second:.2f}")
    
    print("\n💾 RESOURCE UTILIZATION")
    print(f"  Peak GPU Memory:          {metrics.peak_gpu_memory_mb:.1f} MB")
    print(f"  Avg GPU Memory:           {metrics.avg_gpu_memory_mb:.1f} MB")
    print(f"  Peak CPU Usage:           {metrics.peak_cpu_percent:.1f}%")
    print(f"  Avg CPU Usage:            {metrics.avg_cpu_percent:.1f}%")
    print(f"  Peak RAM Usage:           {metrics.peak_ram_mb:.1f} MB")
    
    print("\n🎵 AUDIO METRICS")
    print(f"  Total Audio Duration:     {metrics.total_audio_duration_seconds:.2f}s")
    print(f"  Number of Chunks:         {metrics.num_chunks}")
    print(f"  Total Characters:         {metrics.total_characters}")
    print(f"  Sample Rate:              {metrics.sample_rate} Hz")
    print(f"  Total Samples:            {metrics.audio_samples:,}")
    
    print("\n🔍 QUALITY INDICATORS")
    print(f"  Max Amplitude:            {metrics.max_amplitude:.3f}")
    print(f"  RMS Level:                {metrics.rms_level:.3f}")
    print(f"  Clipping Detected:        {'YES' if metrics.has_clipping else 'NO'}")
    
    print("\n" + "="*70 + "\n")

if __name__ == "__main__":
    metrics = BenchmarkMetrics()
    monitor = ResourceMonitor()
    
    # Start overall timing
    start_time = time.time()
    
    # Load and prepare text
    with open(TEXT_PATH, "r") as f:
        text = f.read()
    
    text = "\n".join([line for line in text.split("\n") if not line.startswith("#")])
    text = [i.strip() for i in text.split("\n") if len(i.strip()) > 0]
    text = [split_text_by_sentence(line) for line in text]
    text = [item for sublist in text for item in sublist]
    
    metrics.num_chunks = len(text)
    metrics.total_characters = sum(len(chunk) for chunk in text)
    
    print(f"[BENCHMARK] Text chunked into {metrics.num_chunks} chunks")
    print(f"[BENCHMARK] Total characters: {metrics.total_characters}")
    
    # Start resource monitoring
    monitor.start()
    
    # Load model
    model_load_start = time.time()
    model = ChatterboxTTS.from_pretrained(
        max_batch_size=BATCH_SIZE,
        max_model_len=MAX_CHUNK_SIZE * 3,
    )
    metrics.model_load_time = time.time() - model_load_start
    print(f"[BENCHMARK] Model loaded in {metrics.model_load_time:.2f}s")
    
    # Warmup run (single small chunk)
    WARMUP_TEXT = "The quick brown fox jumps over the lazy dog. " * 10 
    warmup_start = time.time()
    _ = model.generate(
        [WARMUP_TEXT[:MAX_CHUNK_SIZE]],  # Small warmup chunk
        audio_prompt_path=AUDIO_PROMPT_PATH,
        exaggeration=0.5,
        min_p=0.1,
        top_p=0.8,
    )
    metrics.warmup_time = time.time() - warmup_start
    print(f"[BENCHMARK] Warmup completed in {metrics.warmup_time:.2f}s")
    
    # Reset GPU memory stats after warmup
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    
    # Main generation
    inference_start = time.time()
    first_audio_received = False
    
    audios = model.generate(
        text,
        audio_prompt_path=AUDIO_PROMPT_PATH,
        exaggeration=0.5,
        min_p=0.1,
        top_p=0.8,
    )
    
    # Record TTFA (approximate - when first audio chunk is available)
    if not first_audio_received:
        metrics.time_to_first_audio = time.time() - inference_start
        first_audio_received = True
    
    metrics.total_inference_time = time.time() - inference_start
    print(f"[BENCHMARK] Generation completed in {metrics.total_inference_time:.2f}s")
    
    # Stop monitoring
    monitor.stop()
    resource_stats = monitor.get_stats()
    
    # Stitch audio and analyze
    full_audio = torch.cat(audios, dim=-1)
    metrics.sample_rate = model.sr
    metrics.audio_samples = full_audio.shape[-1]
    metrics.total_audio_duration_seconds = metrics.audio_samples / metrics.sample_rate
    
    # Calculate performance metrics
    if metrics.total_inference_time > 0:
        metrics.real_time_factor = metrics.total_inference_time / metrics.total_audio_duration_seconds
        metrics.audio_seconds_per_second = metrics.total_audio_duration_seconds / metrics.total_inference_time
        metrics.chunks_per_second = metrics.num_chunks / metrics.total_inference_time
    
    # Resource metrics
    metrics.peak_cpu_percent = resource_stats['peak_cpu']
    metrics.avg_cpu_percent = resource_stats['avg_cpu']
    metrics.peak_ram_mb = resource_stats['peak_ram']
    metrics.peak_gpu_memory_mb = resource_stats['peak_gpu']
    metrics.avg_gpu_memory_mb = resource_stats['avg_gpu']
    
    # Audio quality analysis
    # Can't be trusted; needs manual testing
    quality_metrics = analyze_audio_quality(full_audio, metrics.sample_rate)
    metrics.max_amplitude = quality_metrics['max_amplitude']
    metrics.rms_level = quality_metrics['rms_level']
    metrics.has_clipping = quality_metrics['has_clipping']
    
    # Save audio
    ta.save(f"benchmark-sampu-overall.mp3", full_audio, model.sr)
    print(f"[BENCHMARK] Audio saved to benchmark-sampu-overall.mp3")
    
    # Additional quality warnings
    if quality_metrics['abrupt_cutoff']:
        print("!!![WARNING] Abrupt cutoff detected at end of audio")
    if quality_metrics['silence_gaps'] > 0:
        print(f"!!![WARNING] {quality_metrics['silence_gaps']} significant silence gaps detected")
    
    # Total time
    metrics.total_time = time.time() - start_time
    
    # Print comprehensive report
    print_metrics_report(metrics)
    
    # Save metrics to JSON
    import json
    with open("benchmark_metrics.json", "w") as f:
        json.dump(asdict(metrics), f, indent=2)
    print("Detailed metrics saved to benchmark_metrics.json")
    
    # Cleanup
    model.shutdown()