#include <stddef.h>


extern "C" void vector_add(float* a, float* b, float* result, int n) {
    for (int i = 0; i < n; i++) {
        result[i] = a[i] + b[i];
    }
}