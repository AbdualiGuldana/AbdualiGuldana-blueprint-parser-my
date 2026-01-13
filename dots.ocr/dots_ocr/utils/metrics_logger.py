"""
Minimal metrics logger for dots.ocr inference tracking.

This module provides a simple way to track:
- Inference time (wall-clock)
- GPU memory usage (peak and current)
- System memory usage
- Custom metadata

Usage:
    from dots_ocr.utils.metrics_logger import track_inference
    
    with track_inference('metrics.jsonl', metadata={'page': 0}):
        result = inference_function()
"""

import time
import json
from pathlib import Path
from datetime import datetime
from contextlib import contextmanager

# Try to import torch for GPU tracking
try:
    import torch
    TORCH_AVAILABLE = torch.cuda.is_available()
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available. GPU metrics will not be collected.")

# Try to import psutil for system memory tracking
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("Warning: psutil not available. System memory metrics will not be collected.")
    print("Install with: pip install psutil")


@contextmanager
def track_inference(log_file, metadata=None):
    """
    Context manager to track inference metrics.
    
    Args:
        log_file (str): Path to JSONL file where metrics will be appended
        metadata (dict, optional): Custom metadata to include in the log
    
    Yields:
        dict: Record dictionary that you can add custom fields to
    
    Example:
        with track_inference('metrics.jsonl', {'page': 0, 'model': 'dots.ocr'}) as record:
            response = model.generate(image, prompt)
            record['response_length'] = len(response)
            record['num_tokens'] = count_tokens(response)
    """
    # Initialize the record dictionary
    record = {
        'timestamp': datetime.now().isoformat(),
        'metadata': metadata or {},
    }
    
    # === SETUP PHASE ===
    # This happens BEFORE the code inside 'with' block runs
    
    # 1. Reset GPU peak memory statistics
    #    Why? So we only measure THIS inference, not previous ones
    if TORCH_AVAILABLE:
        torch.cuda.reset_peak_memory_stats()
        # CRITICAL: Wait for all GPU operations to finish before starting timer
        # Without this, we might start the timer while previous GPU ops are still running
        torch.cuda.synchronize()
    
    # 2. Record starting system memory
    #    We'll compare this to ending memory to see how much memory this inference used
    if PSUTIL_AVAILABLE:
        start_mem = psutil.Process().memory_info().rss / 1024**3  # Convert to GB
    else:
        start_mem = None
    
    # 3. Start the timer
    #    time.perf_counter() is better than time.time() because:
    #    - It's monotonic (doesn't go backwards if system clock changes)
    #    - It's high resolution (nanosecond precision)
    start_time = time.perf_counter()
    
    # === EXECUTION PHASE ===
    # Now the code inside 'with' block runs
    # We use try/except/finally to ensure cleanup happens even if there's an error
    
    error = None
    try:
        # Yield control back to the caller
        # The record dict is passed so they can add custom fields
        yield record
        
    except Exception as e:
        # If an error occurs in the 'with' block, record it
        error = str(e)
        # Re-raise the error so the caller knows something went wrong
        raise
        
    finally:
        # === CLEANUP PHASE ===
        # This ALWAYS runs, even if there was an error
        
        # 1. Stop the timer
        #    CRITICAL: Wait for GPU operations to finish before stopping timer
        #    Without this, we'd stop the timer while GPU is still processing
        if TORCH_AVAILABLE:
            torch.cuda.synchronize()
        
        end_time = time.perf_counter()
        duration = end_time - start_time
        
        # 2. Record timing
        record['inference_time_seconds'] = duration
        record['success'] = error is None
        if error:
            record['error'] = error
        
        # 3. Collect GPU metrics (if available)
        if TORCH_AVAILABLE:
            # Peak memory: Highest memory used during inference
            record['gpu_memory_peak_mb'] = torch.cuda.max_memory_allocated() / 1024**2
            # Current memory: Memory still allocated after inference
            record['gpu_memory_current_mb'] = torch.cuda.memory_allocated() / 1024**2
            # Which GPU was used
            record['gpu_device'] = torch.cuda.current_device()
        
        # 4. Collect system memory metrics (if available)
        if PSUTIL_AVAILABLE:
            end_mem = psutil.Process().memory_info().rss / 1024**3
            if start_mem is not None:
                # How much memory did this inference use?
                record['system_memory_delta_gb'] = end_mem - start_mem
        
        # 5. Save to file (append mode)
        #    We use 'a' (append) so we don't overwrite previous logs
        #    Each call adds one line to the file
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)  # Create directory if needed
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')


# === HELPER FUNCTIONS ===
# These are useful but not required for basic logging

def load_metrics(log_file):
    """
    Load all metrics from a JSONL file.
    
    Args:
        log_file (str): Path to JSONL metrics file
    
    Returns:
        list: List of metric dictionaries
    
    Example:
        metrics = load_metrics('metrics.jsonl')
        for m in metrics:
            print(f"Page {m['metadata']['page']}: {m['inference_time_seconds']:.2f}s")
    """
    if not Path(log_file).exists():
        return []
    
    metrics = []
    with open(log_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:  # Skip empty lines
                try:
                    metrics.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Warning: Skipping invalid JSON line: {e}")
    
    return metrics


def print_summary(log_file):
    """
    Print a summary of metrics from a JSONL file.
    
    Args:
        log_file (str): Path to JSONL metrics file
    
    Example:
        print_summary('metrics.jsonl')
    """
    metrics = load_metrics(log_file)
    
    if not metrics:
        print(f"No metrics found in {log_file}")
        return
    
    # Calculate statistics
    successful = [m for m in metrics if m.get('success', False)]
    failed = [m for m in metrics if not m.get('success', False)]
    
    durations = [m['inference_time_seconds'] for m in successful]
    
    print(f"\n{'='*60}")
    print(f"METRICS SUMMARY: {log_file}")
    print(f"{'='*60}")
    print(f"\nTotal inferences: {len(metrics)}")
    print(f"  Successful: {len(successful)}")
    print(f"  Failed: {len(failed)}")
    
    if durations:
        print(f"\nTiming:")
        print(f"  Total: {sum(durations):.2f}s")
        print(f"  Average: {sum(durations)/len(durations):.3f}s")
        print(f"  Min: {min(durations):.3f}s")
        print(f"  Max: {max(durations):.3f}s")
    
    # GPU metrics (if available)
    gpu_metrics = [m for m in successful if 'gpu_memory_peak_mb' in m]
    if gpu_metrics:
        peak_mems = [m['gpu_memory_peak_mb'] for m in gpu_metrics]
        print(f"\nGPU Memory:")
        print(f"  Average peak: {sum(peak_mems)/len(peak_mems):.1f}MB")
        print(f"  Max peak: {max(peak_mems):.1f}MB")
    
    print(f"{'='*60}\n")


if __name__ == "__main__":
    # Test the logger
    print("Testing metrics logger...")
    
    # Simulate an inference
    with track_inference('test_metrics.jsonl', metadata={'test': 'example', 'page': 0}) as record:
        import timepython -m dots_ocr.utils.metrics_logger
        time.sleep(0.5)  # Simulate inference time
        record['response_length'] = 1000
        record['custom_field'] = 'test_value'
    
    print("✓ Test completed")
    print_summary('test_metrics.jsonl')