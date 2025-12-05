"""
Chatterbox-VLLM TTS Cold + Warm Startup Benchmark
==================================================
Measures both:
- 5 Cold starts (with disk cache clearing)
- 5 Warm starts (cached in RAM)

Adapted for chatterbox-vllm which uses vLLM backend.
"""

import time
import torch
import psutil
import json
import subprocess
import sys
import os
from datetime import datetime
from pathlib import Path
from chatterbox_vllm.tts import ChatterboxTTS

try:
    import GPUtil
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

class StartupBenchmark:
    """Comprehensive cold + warm startup time measurement for chatterbox-vllm"""
    
    def __init__(self, output_dir="vllm_startup_benchmark_results", max_batch_size=15, max_model_len=1200):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.max_batch_size = max_batch_size
        self.max_model_len = max_model_len
        
        print("="*70)
        print("CHATTERBOX-VLLM TTS COLD + WARM STARTUP BENCHMARK")
        print("="*70)
        print(f"Device: {self.device}")
        print(f"Max Batch Size: {self.max_batch_size}")
        print(f"Max Model Length: {self.max_model_len}")
        print(f"Output: {self.output_dir}")
        print("="*70)
    
    def get_system_state(self):
        """Capture current system state"""
        state = {
            "timestamp": datetime.now().isoformat(),
            "cpu_count": psutil.cpu_count(),
            "ram_total_gb": psutil.virtual_memory().total / (1024**3),
            "ram_available_gb": psutil.virtual_memory().available / (1024**3),
            "ram_used_percent": psutil.virtual_memory().percent,
            "cpu_percent": psutil.cpu_percent(interval=1),
        }
        
        if torch.cuda.is_available():
            state["cuda_available"] = True
            state["gpu_name"] = torch.cuda.get_device_name(0)
            state["gpu_memory_total_gb"] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            state["gpu_memory_allocated_gb"] = torch.cuda.memory_allocated() / (1024**3)
            state["gpu_memory_reserved_gb"] = torch.cuda.memory_reserved() / (1024**3)
            state["cuda_version"] = torch.version.cuda
            state["pytorch_version"] = torch.__version__
            
            if GPU_AVAILABLE:
                try:
                    gpus = GPUtil.getGPUs()
                    if gpus:
                        gpu = gpus[0]
                        state["gpu_utilization_percent"] = gpu.load * 100
                        state["gpu_temperature_c"] = gpu.temperature
                except:
                    pass
        else:
            state["cuda_available"] = False
        
        # Check disk I/O
        try:
            disk = psutil.disk_io_counters()
            state["disk_read_mb"] = disk.read_bytes / (1024**2)
            state["disk_write_mb"] = disk.write_bytes / (1024**2)
        except:
            pass
        
        return state
    
    def clear_disk_cache(self):
        """
        Clear Linux page cache to force cold disk read
        Requires sudo privileges
        """
        print("  Attempting to clear disk cache (requires sudo)...")
        try:
            # Sync to flush buffers
            subprocess.run(["sync"], check=False, capture_output=True)
            
            # Clear page cache
            result = subprocess.run(
                ["sudo", "sh", "-c", "echo 3 > /proc/sys/vm/drop_caches"],
                check=False,
                capture_output=True,
                timeout=5
            )
            
            if result.returncode == 0:
                print("  ✓ Disk cache cleared successfully")
                return True
            else:
                print("  ✗ Could not clear disk cache (may need sudo privileges)")
                print(f"    Run: sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'")
                return False
        except subprocess.TimeoutExpired:
            print("  ✗ Disk cache clear timed out")
            return False
        except Exception as e:
            print(f"  ✗ Error clearing disk cache: {e}")
            return False
    
    def clear_gpu_cache(self):
        """Clear CUDA cache only"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            print("  ✓ Cleared CUDA cache")
    
    def measure_startup(self, run_number, start_type="cold", 
                       warmup_text="Hello world, this is a test for Chatterbox TTS. It's to measure startup time in both cold state and cached state. Let's see how this goes."):
        """
        Measure a single startup (cold or warm)
        
        Args:
            run_number: Run identifier
            start_type: "cold" or "warm"
        """
        print(f"\n{'─'*70}")
        print(f"{start_type.upper()} START - RUN #{run_number}")
        print(f"{'─'*70}")
        
        # Capture pre-load state
        print("Capturing pre-load state...")
        pre_state = self.get_system_state()
        disk_read_before = pre_state.get("disk_read_mb", 0)
        
        # Clear caches based on start type
        if start_type == "cold":
            print("Performing COLD start (clearing all caches)...")
            self.clear_gpu_cache()
            cache_cleared = self.clear_disk_cache()
            if not cache_cleared:
                print("  ⚠️  WARNING: Disk cache NOT cleared - this may not be a true cold start!")
        else:
            print("Performing WARM start (GPU cache only)...")
            self.clear_gpu_cache()
        
        # Small delay to let system settle
        time.sleep(1)
        
        # Measure model load time
        print("Loading model...")
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        load_start = time.time()
        model = ChatterboxTTS.from_pretrained(
            max_batch_size=self.max_batch_size,
            max_model_len=self.max_model_len,
        )
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        load_time = time.time() - load_start
        print(f"  ✓ Model loaded: {load_time:.3f}s")
        
        # Capture post-load state
        post_load_state = self.get_system_state()
        disk_read_after_load = post_load_state.get("disk_read_mb", 0)
        disk_read_during_load = disk_read_after_load - disk_read_before
        
        # Measure warmup time
        print("Running warmup inference...")
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        warmup_start = time.time()
        # chatterbox-vllm expects a list of texts
        _ = model.generate([warmup_text])
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        warmup_time = time.time() - warmup_start
        print(f"  ✓ Warmup completed: {warmup_time:.3f}s")
        
        # Capture post-warmup state
        post_warmup_state = self.get_system_state()
        
        # Calculate deltas
        gpu_memory_increase_load = 0
        gpu_memory_increase_warmup = 0
        if torch.cuda.is_available():
            gpu_memory_increase_load = (
                post_load_state["gpu_memory_allocated_gb"] - 
                pre_state["gpu_memory_allocated_gb"]
            )
            gpu_memory_increase_warmup = (
                post_warmup_state["gpu_memory_allocated_gb"] - 
                post_load_state["gpu_memory_allocated_gb"]
            )
        
        result = {
            "run_number": run_number,
            "start_type": start_type,
            "model_load_time": load_time,
            "warmup_time": warmup_time,
            "total_startup_time": load_time + warmup_time,
            "disk_read_mb": disk_read_during_load,
            "gpu_memory_increase_load_gb": gpu_memory_increase_load,
            "gpu_memory_increase_warmup_gb": gpu_memory_increase_warmup,
            "pre_state": pre_state,
            "post_load_state": post_load_state,
            "post_warmup_state": post_warmup_state,
        }
        
        # Cleanup
        print("  Shutting down model...")
        model.shutdown()
        del model
        self.clear_gpu_cache()
        
        print(f"\n  Summary:")
        print(f"    Type:        {start_type.upper()}")
        print(f"    Model Load:  {load_time:.3f}s")
        print(f"    Warmup:      {warmup_time:.3f}s")
        print(f"    Total:       {load_time + warmup_time:.3f}s")
        print(f"    Disk Read:   {disk_read_during_load:.1f} MB")
        if torch.cuda.is_available():
            print(f"    GPU Δ Load:  {gpu_memory_increase_load:.2f}GB")
            print(f"    GPU Δ Warm:  {gpu_memory_increase_warmup:.2f}GB")
        
        return result
    
    def run_benchmark_suite(self, num_cold=5, num_warm=5, delay_between_runs=5):
        """
        Run complete benchmark suite:
        - num_cold cold starts
        - num_warm warm starts
        """
        print(f"\nBenchmark Plan:")
        print(f"  - {num_cold} COLD starts (with disk cache clearing)")
        print(f"  - {num_warm} WARM starts (cached in RAM)")
        print(f"  - {delay_between_runs}s delay between runs")
        print(f"  - Total runs: {num_cold + num_warm}")
        
        all_results = {
            "cold_starts": [],
            "warm_starts": [],
            "config": {
                "max_batch_size": self.max_batch_size,
                "max_model_len": self.max_model_len,
            }
        }
        
        # Phase 1: Cold starts
        print(f"\n{'='*70}")
        print(f"PHASE 1: COLD STARTS ({num_cold} runs)")
        print(f"{'='*70}")
        
        for i in range(1, num_cold + 1):
            try:
                result = self.measure_startup(i, start_type="cold")
                all_results["cold_starts"].append(result)
                
                if i < num_cold:
                    print(f"\n⏳ Waiting {delay_between_runs}s before next run...")
                    time.sleep(delay_between_runs)
            
            except Exception as e:
                print(f"\n✗ Cold start run {i} failed: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Phase 2: Warm starts
        print(f"\n{'='*70}")
        print(f"PHASE 2: WARM STARTS ({num_warm} runs)")
        print(f"{'='*70}")
        print("Note: Model files should now be cached in RAM")
        
        for i in range(1, num_warm + 1):
            try:
                result = self.measure_startup(i, start_type="warm")
                all_results["warm_starts"].append(result)
                
                if i < num_warm:
                    print(f"\n⏳ Waiting {delay_between_runs}s before next run...")
                    time.sleep(delay_between_runs)
            
            except Exception as e:
                print(f"\n✗ Warm start run {i} failed: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        return all_results
    
    def analyze_results(self, results):
        """Calculate statistics for both cold and warm starts"""
        import numpy as np
        
        analysis = {}
        
        for start_type in ["cold_starts", "warm_starts"]:
            runs = results[start_type]
            if not runs:
                continue
            
            load_times = [r["model_load_time"] for r in runs]
            warmup_times = [r["warmup_time"] for r in runs]
            total_times = [r["total_startup_time"] for r in runs]
            disk_reads = [r.get("disk_read_mb", 0) for r in runs]
            
            analysis[start_type] = {
                "num_runs": len(runs),
                "model_load_time": {
                    "mean": float(np.mean(load_times)),
                    "median": float(np.median(load_times)),
                    "std": float(np.std(load_times)),
                    "min": float(np.min(load_times)),
                    "max": float(np.max(load_times)),
                    "p75": float(np.percentile(load_times, 75)),
                    "p95": float(np.percentile(load_times, 95)),
                    "cv": float(np.std(load_times) / np.mean(load_times)) if np.mean(load_times) > 0 else 0,
                },
                "warmup_time": {
                    "mean": float(np.mean(warmup_times)),
                    "median": float(np.median(warmup_times)),
                    "std": float(np.std(warmup_times)),
                    "min": float(np.min(warmup_times)),
                    "max": float(np.max(warmup_times)),
                    "p75": float(np.percentile(warmup_times, 75)),
                    "p95": float(np.percentile(warmup_times, 95)),
                    "cv": float(np.std(warmup_times) / np.mean(warmup_times)) if np.mean(warmup_times) > 0 else 0,
                },
                "total_startup_time": {
                    "mean": float(np.mean(total_times)),
                    "median": float(np.median(total_times)),
                    "std": float(np.std(total_times)),
                    "min": float(np.min(total_times)),
                    "max": float(np.max(total_times)),
                    "p75": float(np.percentile(total_times, 75)),
                    "p95": float(np.percentile(total_times, 95)),
                },
                "disk_read_mb": {
                    "mean": float(np.mean(disk_reads)),
                    "median": float(np.median(disk_reads)),
                    "min": float(np.min(disk_reads)),
                    "max": float(np.max(disk_reads)),
                }
            }
        
        return analysis
    
    def recommend_thresholds(self, analysis):
        """Recommend thresholds for both cold and warm starts"""
        recommendations = {}
        
        for start_type in ["cold_starts", "warm_starts"]:
            if start_type not in analysis:
                continue
            
            data = analysis[start_type]
            recommendations[start_type] = {
                "model_load_time": {
                    "expected_max": data["model_load_time"]["median"] * 1.5,
                    "warning": data["model_load_time"]["p75"],
                    "critical": data["model_load_time"]["p95"],
                },
                "warmup_time": {
                    "expected_max": data["warmup_time"]["median"] * 1.5,
                    "warning": data["warmup_time"]["p75"],
                    "critical": data["warmup_time"]["p95"],
                },
                "total_startup_time": {
                    "expected_max": data["total_startup_time"]["median"] * 1.5,
                    "warning": data["total_startup_time"]["p75"],
                    "critical": data["total_startup_time"]["p95"],
                },
                "consistency": {
                    "model_load_cv": data["model_load_time"]["cv"],
                    "warmup_cv": data["warmup_time"]["cv"],
                    "assessment": "Consistent" if data["model_load_time"]["cv"] < 0.3 else "Variable"
                }
            }
        
        return recommendations
    
    def print_analysis(self, analysis, recommendations):
        """Pretty print analysis for both cold and warm starts"""
        print("\n" + "="*70)
        print("ANALYSIS RESULTS")
        print("="*70)
        
        for start_type in ["cold_starts", "warm_starts"]:
            if start_type not in analysis:
                continue
            
            data = analysis[start_type]
            rec = recommendations[start_type]
            
            print(f"\n{'='*70}")
            print(f"{start_type.replace('_', ' ').upper()} (n={data['num_runs']})")
            print(f"{'='*70}")
            
            # Model Load Time
            print("\nMODEL LOAD TIME")
            print("-"*70)
            ml = data["model_load_time"]
            print(f"  Mean:      {ml['mean']:.3f}s")
            print(f"  Median:    {ml['median']:.3f}s  ← Baseline")
            print(f"  Std Dev:   {ml['std']:.3f}s")
            print(f"  Range:     {ml['min']:.3f}s - {ml['max']:.3f}s")
            print(f"  P75:       {ml['p75']:.3f}s")
            print(f"  P95:       {ml['p95']:.3f}s")
            print(f"  CV:        {ml['cv']:.2%} ({'Good' if ml['cv'] < 0.2 else 'High' if ml['cv'] < 0.3 else 'Very High'})")
            
            # Warmup Time
            print("\nWARMUP TIME")
            print("-"*70)
            wt = data["warmup_time"]
            print(f"  Mean:      {wt['mean']:.3f}s")
            print(f"  Median:    {wt['median']:.3f}s  ← Baseline")
            print(f"  Std Dev:   {wt['std']:.3f}s")
            print(f"  Range:     {wt['min']:.3f}s - {wt['max']:.3f}s")
            print(f"  P75:       {wt['p75']:.3f}s")
            print(f"  P95:       {wt['p95']:.3f}s")
            print(f"  CV:        {wt['cv']:.2%}")
            
            # Total
            print("\nTOTAL STARTUP TIME")
            print("-"*70)
            tt = data["total_startup_time"]
            print(f"  Mean:      {tt['mean']:.3f}s")
            print(f"  Median:    {tt['median']:.3f}s  ← Baseline")
            print(f"  Range:     {tt['min']:.3f}s - {tt['max']:.3f}s")
            print(f"  P95:       {tt['p95']:.3f}s")
            
            # Disk I/O
            print("\nDISK I/O")
            print("-"*70)
            dr = data["disk_read_mb"]
            print(f"  Mean:      {dr['mean']:.1f} MB")
            print(f"  Median:    {dr['median']:.1f} MB")
            print(f"  Range:     {dr['min']:.1f} - {dr['max']:.1f} MB")
            
            # Thresholds
            print("\nRECOMMENDED THRESHOLDS")
            print("-"*70)
            print(f"Model Load:  < {rec['model_load_time']['expected_max']:.1f}s (expected)")
            print(f"             > {rec['model_load_time']['warning']:.1f}s (warning)")
            print(f"             > {rec['model_load_time']['critical']:.1f}s (critical)")
            print(f"Warmup:      < {rec['warmup_time']['expected_max']:.1f}s (expected)")
            print(f"Total:       < {rec['total_startup_time']['expected_max']:.1f}s (expected)")
            print(f"Consistency: {rec['consistency']['assessment']}")
        
        # Comparison
        if "cold_starts" in analysis and "warm_starts" in analysis:
            print(f"\n{'='*70}")
            print("COLD vs WARM COMPARISON")
            print(f"{'='*70}")
            cold = analysis["cold_starts"]["model_load_time"]["median"]
            warm = analysis["warm_starts"]["model_load_time"]["median"]
            speedup = cold / warm if warm > 0 else 0
            print(f"  Cold Load:    {cold:.2f}s")
            print(f"  Warm Load:    {warm:.2f}s")
            print(f"  Speedup:      {speedup:.2f}x faster when cached")
            print(f"  Difference:   {cold - warm:.2f}s saved with caching")
    
    def save_results(self, results, analysis, recommendations):
        """Save all results to files"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save raw data
        raw_file = self.output_dir / f"raw_results_{timestamp}.json"
        with open(raw_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save analysis
        analysis_file = self.output_dir / f"analysis_{timestamp}.json"
        with open(analysis_file, 'w') as f:
            json.dump({
                "analysis": analysis,
                "recommendations": recommendations
            }, f, indent=2)
        
        # Save human-readable report
        report_file = self.output_dir / f"report_{timestamp}.txt"
        with open(report_file, 'w') as f:
            f.write("="*70 + "\n")
            f.write("CHATTERBOX-VLLM TTS COLD + WARM STARTUP BENCHMARK\n")
            f.write("="*70 + "\n\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Max Batch Size: {results['config']['max_batch_size']}\n")
            f.write(f"Max Model Length: {results['config']['max_model_len']}\n\n")
            
            for start_type in ["cold_starts", "warm_starts"]:
                if start_type not in analysis:
                    continue
                
                f.write(f"\n{start_type.replace('_', ' ').upper()}\n")
                f.write("="*70 + "\n")
                f.write(f"Number of runs: {analysis[start_type]['num_runs']}\n\n")
                
                for section in ["model_load_time", "warmup_time", "total_startup_time"]:
                    f.write(f"\n{section.replace('_', ' ').title()}\n")
                    f.write("-"*70 + "\n")
                    for metric, value in analysis[start_type][section].items():
                        if metric != 'cv':
                            f.write(f"  {metric}: {value:.3f}s\n")
                
                f.write("\nRecommended Thresholds\n")
                f.write("-"*70 + "\n")
                rec = recommendations[start_type]
                for component in ["model_load_time", "warmup_time", "total_startup_time"]:
                    f.write(f"\n{component.replace('_', ' ').title()}\n")
                    for level, value in rec[component].items():
                        f.write(f"  {level}: {value:.1f}s\n")
        
        print(f"\n✓ Results saved:")
        print(f"  Raw data:  {raw_file}")
        print(f"  Analysis:  {analysis_file}")
        print(f"  Report:    {report_file}")
        
        return raw_file, analysis_file, report_file


def main():
    """Main benchmark execution"""
    print("\n" + "="*70)
    print("CHATTERBOX-VLLM COLD + WARM STARTUP TIME BENCHMARK")
    print("="*70)
    print("\nThis benchmark will measure:")
    print("  1. Cold starts (5 runs) - with disk cache clearing")
    print("  2. Warm starts (5 runs) - with files cached in RAM")
    print("\nNote: Cold starts require sudo to clear disk cache.")
    print("      If sudo is not available, results may not be accurate.")
    print("\n" + "="*70)
    
    # Configuration
    num_cold = 5
    num_warm = 5
    delay_between_runs = 5
    
    # vLLM-specific settings (adjust based on  GPU)
    max_batch_size = 15  # Adjust for your VRAM (15 for 8GB, 40 for 16GB, 80 for 24GB)
    max_model_len = 1200  # Rough heuristic: MAX_CHUNK_SIZE * 3vb 
    
    try:
        # Initialize benchmark
        benchmark = StartupBenchmark(
            max_batch_size=max_batch_size,
            max_model_len=max_model_len
        )
        
        # Run benchmark suite
        results = benchmark.run_benchmark_suite(
            num_cold=num_cold,
            num_warm=num_warm,
            delay_between_runs=delay_between_runs
        )
        
        if not results["cold_starts"] and not results["warm_starts"]:
            print("\n✗ No successful runs. Cannot generate analysis.")
            return
        
        # Analyze results
        analysis = benchmark.analyze_results(results)
        recommendations = benchmark.recommend_thresholds(analysis)
        
        # Print analysis
        benchmark.print_analysis(analysis, recommendations)
        
        # Save results
        benchmark.save_results(results, analysis, recommendations)
        
        print("\n" + "="*70)
        print("✓ BENCHMARK COMPLETE!")
        print("="*70)
        
    except KeyboardInterrupt:
        print("\n\nBenchmark interrupted by user.")
    except Exception as e:
        print(f"\n\nError during benchmark: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()