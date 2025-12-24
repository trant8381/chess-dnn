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

# ------------------ Torch Settings ------------------
include_directories("${MAIN_PATH}/external/libtorch")
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable (main "${SRC}/main.cpp"
                     "${SRC}/create_state.cpp")
target_link_libraries(main ${TORCH_LIBRARIES})

# -------------------- REMOVE WINDOWS ERRORS --------------------
# The following code block is suggested to be used on Windows.
# According to https://github.com/pytorch/pytorch/issues/25457,
# the DLLs need to be copied to avoid memory errors.
if (MSVC)
        file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
        add_custom_command(TARGET main 
                        POST_BUILD
                        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                        ${TORCH_DLLS}
                        $<TARGET_FILE_DIR:main>)
endif (MSVC)