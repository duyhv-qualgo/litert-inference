cmake_minimum_required(VERSION 3.20)
project(gemma_inference CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(LITERT_LM_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/litert_lm"
    CACHE PATH "Root of fetched LiteRT-LM prebuilts (see scripts/fetch_prebuilt.sh)")

if(NOT EXISTS "${LITERT_LM_DIR}/include/litert_lm/c/engine.h")
  message(FATAL_ERROR
    "LiteRT-LM C header not found at ${LITERT_LM_DIR}/include/litert_lm/c/engine.h. "
    "Run scripts/fetch_prebuilt.sh first.")
endif()

# Pick the prebuilt lib that matches the target platform.
if(ANDROID)
  set(_LITERT_LM_LIB_DIR "${LITERT_LM_DIR}/lib/android_${ANDROID_ABI}")
elseif(APPLE)
  set(_LITERT_LM_LIB_DIR "${LITERT_LM_DIR}/lib/ios")
elseif(UNIX AND NOT APPLE)
  set(_LITERT_LM_LIB_DIR "${LITERT_LM_DIR}/lib/linux_x86_64")
else()
  set(_LITERT_LM_LIB_DIR "${LITERT_LM_DIR}/lib/desktop")
endif()

find_library(LITERT_LM_C_LIB
    NAMES LiteRtLmCApi litert_lm_c_api litert_lm
    PATHS "${_LITERT_LM_LIB_DIR}"
    NO_DEFAULT_PATH)
if(NOT LITERT_LM_C_LIB)
  message(FATAL_ERROR "LiteRT-LM C shared library not found in ${_LITERT_LM_LIB_DIR}")
endif()

add_library(gemma_engine SHARED
    src/gemma_engine.cpp
)
target_include_directories(gemma_engine
    PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE ${LITERT_LM_DIR}/include
)
target_link_libraries(gemma_engine PUBLIC ${LITERT_LM_C_LIB})
target_link_options(gemma_engine PUBLIC
    "-Wl,-rpath-link,${_LITERT_LM_LIB_DIR}"
    "-Wl,-rpath,\$ORIGIN"
    "-Wl,-rpath,${_LITERT_LM_LIB_DIR}")
if(DEFINED ENV{CONDA_PREFIX})
    target_link_options(gemma_engine PUBLIC
        "-L$ENV{CONDA_PREFIX}/lib"
        "-Wl,-rpath,$ENV{CONDA_PREFIX}/lib")
endif()
target_compile_options(gemma_engine PRIVATE -Wall -Wextra -O3)

if(NOT ANDROID AND NOT IOS)
  add_executable(test_gemma src/test_main.cc)
  target_link_libraries(test_gemma PRIVATE gemma_engine)
endif()

if(ANDROID)
  add_library(gemma_jni SHARED platform/android/gemma_jni.cpp)
  target_link_libraries(gemma_jni PRIVATE gemma_engine log)
endif()
