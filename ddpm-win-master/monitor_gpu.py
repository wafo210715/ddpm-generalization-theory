"""
Real-time GPU monitoring script for dual RTX 3090 setup.
Run this in a separate terminal while training to see GPU stats.

Usage:
    python monitor_gpu.py
"""

import subprocess
import time
import re
from datetime import datetime


def get_gpu_stats():
    """Get GPU stats from nvidia-smi."""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw,pstate',
             '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True
        )

        lines = result.stdout.strip().split('\n')
        stats = []

        for line in lines:
            if not line.strip():
                continue
            parts = [x.strip() for x in line.split(',')]
            if len(parts) >= 9:
                stats.append({
                    'index': parts[0],
                    'name': parts[1],
                    'gpu_util': parts[2] + '%',
                    'mem_util': parts[3] + '%',
                    'mem_used': parts[4] + ' MiB',
                    'mem_total': parts[5] + ' MiB',
                    'temp': parts[6] + 'C',
                    'power': parts[7] + 'W',
                    'pstate': parts[8]
                })

        return stats
    except Exception as e:
        print(f"Error getting GPU stats: {e}")
        return []


def print_header():
    """Print header for monitoring display."""
    print("\n" + "=" * 100)
    print(f"{'GPU MONITOR':^100}")
    print("=" * 100)
    print(f"{'GPU':<4} {'Name':<20} {'GPU%':<6} {'MEM%':<6} {'Memory':<15} {'Temp':<6} {'Power':<8} {'PState':<6} {'Time':<12}")
    print("-" * 100)


def print_stats(stats, timestamp):
    """Print GPU stats."""
    for stat in stats:
        mem_str = f"{stat['mem_used']}/{stat['mem_total']}"
        print(f"{stat['index']:<4} {stat['name'][:20]:<20} {stat['gpu_util']:<6} {stat['mem_util']:<6} {mem_str:<15} {stat['temp']:<6} {stat['power']:<8} {stat['pstate']:<6} {timestamp:<12}")


def monitor(interval=1):
    """Monitor GPUs continuously."""
    print("\nStarting GPU monitoring... Press Ctrl+C to stop.\n")

    try:
        while True:
            timestamp = datetime.now().strftime("%H:%M:%S")
            stats = get_gpu_stats()

            if stats:
                print_header()
                print_stats(stats, timestamp)
                print("=" * 100)

            time.sleep(interval)

    except KeyboardInterrupt:
        print("\n\nMonitoring stopped.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Monitor GPU stats in real-time")
    parser.add_argument("-i", "--interval", type=float, default=1.0, help="Update interval in seconds (default: 1.0)")
    args = parser.parse_args()

    monitor(interval=args.interval)
