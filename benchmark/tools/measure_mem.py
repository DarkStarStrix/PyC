import subprocess
import sys
import psutil
import os
import time

def measure_memory(command):
    process = subprocess.Popen(command, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    ps_process = psutil.Process(process.pid)
    max_mem = 0
    try:
        while process.poll() is None:
            mem = ps_process.memory_info().rss
            max_mem = max(max_mem, mem)
            time.sleep(0.01)
    except psutil.NoSuchProcess:
        pass
    process.wait()
    return max_mem / (1024 * 1024)  # Convert to MB

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 measure_mem.py <command>")
        sys.exit(1)
    command = " ".join(sys.argv[1:])
    mem_mb = measure_memory(command)
    print(mem_mb)
    sys.exit(0)