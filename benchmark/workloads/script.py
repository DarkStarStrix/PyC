# General-purpose Python script for benchmarking
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

def main():
    result = 0
    for i in range(35):
        result += fibonacci(i)
    with open("output.txt", "w") as f:
        f.write(str(result))

if __name__ == "__main__":
    main()
