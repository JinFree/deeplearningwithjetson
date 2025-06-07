import numpy as np
import ctypes

# C 라이브러리 로드
# 먼저 C 코드를 컴파일하여 shared library (.so 파일)로 만들어야 합니다.
c_lib = ctypes.CDLL('./build/vector_add.so')

# 벡터 합 연산을 수행하는 C 함수 정의
# void vector_add(double* a, double* b, double* result, int n)
c_lib.vector_add.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_int]

n = 1024
a = np.random.randn(n).astype(np.float32)
b = np.random.randn(n).astype(np.float32)
c = np.zeros_like(a)

# C 함수 호출
c_lib.vector_add(a.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                 b.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                 c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                 len(a))

# 결과 출력
print(np.max(c -(a+b)))