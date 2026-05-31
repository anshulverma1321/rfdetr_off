import time
import torch
import psutil
from collections import deque
import numpy as np

class PerformanceTracker:
    def __init__(self, history_size=100):
        """
        Tracks performance metrics such as FPS, latency, and GPU memory usage.
        """
        self.times = deque(maxlen=history_size)
        self.start_time = None
        self.history_size = history_size
        
        # Determine if CUDA is available for memory tracking
        self.has_cuda = torch.cuda.is_available()

    def start(self):
        """Start tracking a frame."""
        # Sync CUDA before timing if available
        if self.has_cuda:
            torch.cuda.synchronize()
        self.start_time = time.time()

    def stop(self):
        """Stop tracking a frame and record latency."""
        if self.has_cuda:
            torch.cuda.synchronize()
        end_time = time.time()
        
        if self.start_time is not None:
            latency = end_time - self.start_time
            self.times.append(latency)
            self.start_time = None
            
    def get_metrics(self):
        """
        Compute current metrics.
        Returns:
            dict: Latency (ms), FPS, and GPU/System memory usage (MB).
        """
        metrics = {
            "avg_latency_ms": 0.0,
            "fps": 0.0,
            "gpu_mem_allocated_mb": 0.0,
            "gpu_mem_reserved_mb": 0.0,
            "sys_ram_usage_percent": 0.0
        }
        
        if len(self.times) > 0:
            avg_latency_sec = np.mean(self.times)
            metrics["avg_latency_ms"] = avg_latency_sec * 1000
            metrics["fps"] = 1.0 / avg_latency_sec if avg_latency_sec > 0 else 0.0
            
        if self.has_cuda:
            metrics["gpu_mem_allocated_mb"] = torch.cuda.memory_allocated() / (1024 ** 2)
            metrics["gpu_mem_reserved_mb"] = torch.cuda.memory_reserved() / (1024 ** 2)
            
        metrics["sys_ram_usage_percent"] = psutil.virtual_memory().percent
        
        return metrics

    def log_metrics(self):
        """Prints metrics to console."""
        metrics = self.get_metrics()
        log_str = f"FPS: {metrics['fps']:.2f} | Latency: {metrics['avg_latency_ms']:.2f}ms"
        if self.has_cuda:
            log_str += f" | GPU Mem (Alloc): {metrics['gpu_mem_allocated_mb']:.1f}MB"
        print(log_str)
