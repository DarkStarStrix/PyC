import time
import subprocess
import sys

def measure_time(command):
    start = time.time()
    subprocess.run(command, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    end = time.time()
    return (end - start) * 1000  # Return time in milliseconds

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 measure_time.py <command>")
        sys.exit(1)
    command = " ".join(sys.argv[1:])
    elapsed_ms = measure_time(command)
    print(elapsed_ms)
    sys.exit(0)
