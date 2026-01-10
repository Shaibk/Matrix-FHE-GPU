cmake_minimum_required(VERSION 3.18)
project(matrix-fhe-gpu LANGUAGES CXX CUDA)

# ==========================================
# 1. Configuration
# ==========================================
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)

# 针对 NVIDIA A100 (Ampere SM_80)
set(CMAKE_CUDA_ARCHITECTURES 80)

# 启用 Phantom 的 PTX 汇编优化
add_compile_definitions(PHANTOM_USE_CUDA_PTX)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# ==========================================
# 2. Dependencies & Paths
# ==========================================
set(PHANTOM_HOME ${CMAKE_CURRENT_SOURCE_DIR}/extern/phantom-fhe)

# [修正] 补全了 fft.cu, prng.cu, batchencoder.cu 以及 ntt 目录下的遗漏文件
set(PHANTOM_SOURCES
    # --- 核心逻辑 ---
    ${PHANTOM_HOME}/src/context.cu
    ${PHANTOM_HOME}/src/ckks.cu
    ${PHANTOM_HOME}/src/rns.cu
    ${PHANTOM_HOME}/src/polymath.cu
    ${PHANTOM_HOME}/src/secretkey.cu
    ${PHANTOM_HOME}/src/galois.cu
    ${PHANTOM_HOME}/src/rns_base.cu
    ${PHANTOM_HOME}/src/rns_bconv.cu
    ${PHANTOM_HOME}/src/scalingvariant.cu
    ${PHANTOM_HOME}/src/evaluate.cu
    ${PHANTOM_HOME}/src/eval_key_switch.cu
    
    # [新增/补全] 解决链接错误的关键文件
    ${PHANTOM_HOME}/src/fft.cu           # 解决 special_fft, ComplexRoots
    ${PHANTOM_HOME}/src/prng.cu          # 解决 sample_error_poly 等
    ${PHANTOM_HOME}/src/batchencoder.cu  # PhantomCKKSEncoder 可能依赖
    
    # --- Host 端辅助 ---
    ${PHANTOM_HOME}/src/host/modulus.cu
    ${PHANTOM_HOME}/src/host/rns.cu
    ${PHANTOM_HOME}/src/host/ntt.cu
    ${PHANTOM_HOME}/src/host/uintarith.cu
    ${PHANTOM_HOME}/src/host/uintarithmod.cu
    ${PHANTOM_HOME}/src/host/uintarithsmallmod.cu
    ${PHANTOM_HOME}/src/host/globals.cu
    ${PHANTOM_HOME}/src/host/hash.cu
    ${PHANTOM_HOME}/src/host/numth.cu
    ${PHANTOM_HOME}/src/host/blake2b.cu
    ${PHANTOM_HOME}/src/host/blake2xb.cu
    
    # --- NTT 核心 (补全) ---
    ${PHANTOM_HOME}/src/ntt/ntt_1d.cu
    ${PHANTOM_HOME}/src/ntt/fntt_2d.cu
    ${PHANTOM_HOME}/src/ntt/intt_2d.cu
    ${PHANTOM_HOME}/src/ntt/ntt_modup.cu
    ${PHANTOM_HOME}/src/ntt/ntt_moddown.cu
    ${PHANTOM_HOME}/src/ntt/ntt_keyswitch_old.cu # 防止 modup 依赖旧实现
)

# ==========================================
# 3. Project Source Files
# ==========================================
file(GLOB_RECURSE MY_SRC_FILES 
    src/*.cu 
    src/*.cpp
)

# ==========================================
# 4. Build Target
# ==========================================
add_executable(matrix_fhe_gpu 
    ${MY_SRC_FILES} 
    ${PHANTOM_SOURCES}
)

# ==========================================
# 5. Include Directories & Compile Options
# ==========================================
target_include_directories(matrix_fhe_gpu PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${PHANTOM_HOME}/include
    ${PHANTOM_HOME}/src
)

if(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(matrix_fhe_gpu PRIVATE 
        $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math>
        $<$<COMPILE_LANGUAGE:CXX>:-O3>
    )
endif()

target_link_libraries(matrix_fhe_gpu PRIVATE m)

message(STATUS "Configured matrix-fhe-gpu with FULL Phantom sources.")