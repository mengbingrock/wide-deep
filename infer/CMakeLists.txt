
cmake_minimum_required(VERSION 3.16)
project(llama_infer)

set(CMAKE_CXX_STANDARD 20)
find_package(glog REQUIRED)
find_package(Armadillo REQUIRED)

aux_source_directory(inference/source/tensor/ DIR_TENSOR)
aux_source_directory(inference/source/base/ DIR_BASE)
aux_source_directory(inference/source/op/ DIR_OP)
aux_source_directory(inference/source/model/ DIR_MODEL)
aux_source_directory(inference/source/op/kernels DIR_KERNEL)
aux_source_directory(inference/source/sampler DIR_SAMPLE)


# Add debug symbols
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0")

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
find_package(GTest REQUIRED)
add_library(llama SHARED ${DIR_TENSOR} ${DIR_BASE} ${DIR_OP} ${DIR_MODEL} ${DIR_KERNEL} ${DIR_SAMPLE})
target_link_libraries(llama sentencepiece glog::glog  gtest gtest_main pthread)
target_link_libraries(llama ${ARMADILLO_LIBRARIES})
target_include_directories(llama PUBLIC ${glog_INCLUDE_DIR})
target_include_directories(llama PUBLIC ${PROJECT_SOURCE_DIR}/inference/include)
target_include_directories(llama PUBLIC ${Armadillo_INCLUDE_DIR})


add_executable(llama_infer main.cpp)
target_link_directories(llama_infer PUBLIC ${PROJECT_SOURCE_DIR}/lib)
target_link_libraries(llama_infer llama)
add_subdirectory(test)
