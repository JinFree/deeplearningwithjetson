import timeit
import numpy as np


def vector_add(a, b, c, n):
    for i in range(n):
        c[i] = a[i] + b[i]
    return c


n = 4096 * 1024
a = np.random.randn(n).astype(np.float32)
b = np.random.randn(n).astype(np.float32)
c = np.zeros_like(a)    
    
timer = timeit.Timer(lambda: vector_add(a, b, c, n))
duration = timer.timeit(number=10)
print(f"10회 반복 평균: {duration/10*1000:.6f}ms")