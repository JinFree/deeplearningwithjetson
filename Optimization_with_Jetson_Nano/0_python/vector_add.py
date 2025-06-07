import timeit
import numpy as np


def vector_add(a, b, c, n):
    for i in range(n):
        c[i] = a[i] + b[i]
    return c


n = 4096
a = np.random.randn(n).astype(np.float32)
b = np.random.randn(n).astype(np.float32)
c = np.zeros_like(a)    
    
timer = timeit.Timer(vector_add(a, b, c, n).copy)
duration = timer.timeit(number=100)
print(f"100회 반복 평균: {duration/100:.6f}초")