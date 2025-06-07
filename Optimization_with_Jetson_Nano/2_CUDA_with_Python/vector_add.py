import timeit
import numpy as np
import ctypes

def vector_add_python(a, b, c, n):
    for i in range(n):
        c[i] = a[i] + b[i]
    return c

def vector_add_numpy(a, b, c, n):
    c = a + b
    return c

# C 라이브러리 로드
# 먼저 C 코드를 컴파일하여 shared library (.so 파일)로 만들어야 합니다.
c_lib = ctypes.CDLL('./build/vector_add.so')

# 벡터 합 연산을 수행하는 C 함수 정의
# void vector_add(double* a, double* b, double* result, int n)
c_lib.vector_add.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_int]

n = 4096 * 4096
a = np.random.randn(n).astype(np.float32)
b = np.random.randn(n).astype(np.float32)
c = np.zeros_like(a)

timer = timeit.Timer(lambda: vector_add_python(a, b, c, n))
duration = timer.timeit(number=2)
print(f"vector_add_python 2회 반복 평균: {duration/2*1000:.6f}ms")

timer = timeit.Timer(lambda: vector_add_numpy(a, b, c, n))
duration = timer.timeit(number=10)
print(f"vector_add_numpy 10회 반복 평균: {duration/10*1000:.6f}ms")

# C 함수 호출
c_lib.vector_add(a.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                 b.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                 c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                 len(a))

timer = timeit.Timer(lambda: c_lib.vector_add(a.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                                                b.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                                                c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                                                len(a)))
duration = timer.timeit(number=10)
print(f"vector_add 10회 반복 평균: {duration/10*1000:.6f}ms")