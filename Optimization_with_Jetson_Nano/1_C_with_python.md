C언어로 작성한 함수를 Python에서 사용하는 것을 실습합니다.

실습 과정에서 vector 합 함수를 구현하고 각각의 환경에서 속도를 비교해보겠습니다.

우선 Python 구현은 아래와 같습니다.
```
def vector_add_python(a, b, c, n):
    for i in range(n):
        c[i] = a[i] + b[i]
    return c

def vector_add_numpy(a, b, c, n):
    c = a + b
    return c
```

C언어 구현을 확인해보겠습니다.
```
#include <stddef.h>

extern "C" void vector_add(float* a, float* b, float* result, int n) {
    for (int i = 0; i < n; i++) {
        result[i] = a[i] + b[i];
    }
}
```

CMake를 사용하여 빌드한 후 Python에서 사용합니다.

작업 경로로 이동합니다.
```
cd /home/nvidia/workspace/deeplearningwithjetson/Optimization_with_Jetson_Nano/1_C_with_Python
```

빌드합니다.
```
mkdir -p build
cd build
cmake ..
make
```

Python에서 시간을 확인합니다.
```
python3 /home/nvidia/workspace/deeplearningwithjetson/Optimization_with_Jetson_Nano/1_C_with_Python/vector_add.py
```

아래와 유사한 시간을 확인할 수 있습니다.
```
vector_add_python 10회 반복 평균: 4370.642379ms
vector_add_numpy 10회 반복 평균: 15.265379ms
vector_add 10회 반복 평균: 10.855418ms
```