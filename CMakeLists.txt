cmake_minimum_required (VERSION 3.18) #Usually the latest one you are using
project(
        My_Torch_Project
        VERSION 0.0
        DESCRIPTION "My Torch Project")

# --------- Optionally enable standards ---------
# ------------------ STANDARDS ------------------
enable_language(C)
enable_language(CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ------------------ Path Settings ------------------
set(MAIN_PATH "${CMAKE_SOURCE_DIR}") # ------- Makes source dir
set(CMAKE_PREFIX_PATH "${MAIN_PATH}/external/libtorch")
set(SRC "${MAIN_PATH}/src")
file(GLOB MAIN_SRC
     "${SRC}/create_state.cpp"
     "${SRC}/mcts.cpp"
     "${SRC}/main.cpp"
)

# ------------------ Torch Settings ------------------
include_directories("${MAIN_PATH}/external/libtorch")
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable (main ${MAIN_SRC})
target_include_directories(main PRIVATE ${SRC}/include)

# ------------------ CUDA Settings ------------------
if ((DEFINED ENV{CMAKE_CUDA_COMPILER}) AND (CMAKE_CUDA_COMPILER))
  set(CUDA_SOURCES "${SRC}")
  file(GLOB CUDA_SRC
     "${CUDA_SOURCES}/*.cu"
     "${SRC}/create_state_fast.cpp"
  )

  enable_language(CUDA)
  cuda_add_library(cuda_uint64 STATIC CUDA_SRC)
  target_include_directories(cuda_uint64 PRIVATE ${CUDA_SOURCES}/include)
  set_target_properties(cuda_uint64
                        PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  target_link_libraries(main PRIVATE ${TORCH_LIBRARIES} cuda_uint64)
  set_property(TARGET main
               PROPERTY CUDA_SEPARABLE_COMPILATION ON)
  target_compile_options(main PRIVATE -Wall -Wextra -Wpedantic -Werror -g)
else()
  target_link_libraries(main ${TORCH_LIBRARIES})
  target_compile_options(main PRIVATE -Wall -Wextra -Wpedantic -Werror -g)
endif()