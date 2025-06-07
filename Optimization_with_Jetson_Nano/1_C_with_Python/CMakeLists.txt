cmake_minimum_required(VERSION 3.5)
project(vector_add C)

# -fPIC: Position Independent Code, 공유 라이브러리에 필요
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC")

# vector_add.c 로부터 shared library 생성
add_library(vector_add SHARED vector_add.c)

# 라이브러리 이름을 그대로 vector_add.so 로 출력
set_target_properties(vector_add PROPERTIES
    PREFIX ""       # 기본 'lib' 접두사 제거
    SUFFIX ".so"    # .so 확장자 지정
    OUTPUT_NAME "vector_add"
)
