cmake_minimum_required(VERSION 3.22.1)
project(smslm)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- CRITICAL PERFORMANCE FLAGS ---
if(ANDROID_ABI STREQUAL "arm64-v8a")
    add_compile_options(-O3 -DNDEBUG -march=armv8.2-a+fp16+dotprod -fno-finite-math-only)
else()
    add_compile_options(-O3 -DNDEBUG)
endif()

# 1. Path to llama.cpp root
get_filename_component(LLAMA_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../../llama.cpp" ABSOLUTE)

# 2. Modern llama.cpp Options
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER   OFF CACHE BOOL "" FORCE)
# Ensure we use the static library to avoid linkage issues
set(BUILD_SHARED_LIBS    OFF CACHE BOOL "" FORCE)
set(GGML_VULKAN OFF CACHE BOOL "" FORCE)
set(LLAMA_VULKAN OFF CACHE BOOL "" FORCE)

# 3. Add llama.cpp
add_subdirectory(${LLAMA_DIR} ${CMAKE_CURRENT_BINARY_DIR}/llama.cpp EXCLUDE_FROM_ALL)

# 4. Create your shared library
add_library(smslm SHARED
        smslm.cpp
        llm_inference.cpp
)

# 5. Includes
target_include_directories(smslm PRIVATE
        ${LLAMA_DIR}/include
        ${LLAMA_DIR}/common
        ${LLAMA_DIR}/src
)

# 6. Link
target_link_libraries(smslm
        llama
        android
        log
)