cmake_minimum_required(VERSION 3.6)
project(cuda_samples LANGUAGES CXX CUDA)

set(CMAKE_BUILD_TYPE "debug")
#add_definitions("-DENABLE_SSE")
#set(CMAKE_CXX_FLAGS "-std=c++17 ${SSE_FLAGS} -g -O3 -march=native")
#set(CMAKE_CXX_FLAGS "-std=c++17 -O3 -fopenmp") # -fopenmp enable multi-thread
set(CMAKE_CXX_FLAGS "-std=c++17")
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules) # Add elements to list
#set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)

find_package(OpenCV 4 REQUIRED)
find_package(Sophus REQUIRED)
#find_package(Pangolin REQUIRED) # used to show pointcloud

find_package(G2O REQUIRED)
if (G2O_FOUND)
    message(STATUS "G2O found. Version: ${G2O_VERSION}")
endif (G2O_FOUND)

find_package(Eigen3 REQUIRED)
find_package(Glog REQUIRED)
#find_package(realsense2 REQUIRED)
#find_package(GLFW REQUIRED)
#find_package(Boost COMPONENTS system filesystem serialization REQUIRED)
find_package(OpenMP REQUIRED)
find_package(Ceres REQUIRED)

find_package(CUDA REQUIRED)
if (CUDA_FOUND)
    message("-- Found CUDA version ${CUDA_VERSION} installed in: "
            "${CUDA_TOOLKIT_ROOT_DIR} via legacy (< 3.17) CMake module. "
            "Using the legacy CMake module means that any installation of "
            "Ceres will require that the CUDA libraries be installed in a "
            "location included in the LD_LIBRARY_PATH.")
    enable_language(CUDA)

    macro(DECLARE_IMPORTED_CUDA_TARGET COMPONENT)
        add_library(CUDA::${COMPONENT} INTERFACE IMPORTED)
        target_include_directories(
                CUDA::${COMPONENT} INTERFACE ${CUDA_INCLUDE_DIRS})
        target_link_libraries(
                CUDA::${COMPONENT} INTERFACE ${CUDA_${COMPONENT}_LIBRARY} ${ARGN})
    endmacro()

    declare_imported_cuda_target(cublas)
    declare_imported_cuda_target(cusolver)
    declare_imported_cuda_target(cusparse)
    declare_imported_cuda_target(cudart ${CUDA_LIBRARIES})

    set(CUDAToolkit_BIN_DIR ${CUDA_TOOLKIT_ROOT_DIR}/bin)
else (CUDA_FOUND)
    message("-- Did not find CUDA, disabling CUDA support.")
    update_cache_variable(USE_CUDA OFF)
endif (CUDA_FOUND)
set(CUDA_ADD_CUBLAS_TO_TARGET ON)

if (OPENMP_FOUND)
    message(STATUS "OpenMP found will try to link!")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif ()

include_directories(
        ${OpenCV_INCLUDE_DIRS}
        ${G2O_INCLUDE_DIRS}
        ${Sophus_INCLUDE_DIRS}
        ${EIGEN3_INCLUDE_DIR}
        #        "/usr/include/eigen3/"
        ${Pangolin_INCLUDE_DIRS}
        ${CSPARSE_INCLUDE_DIR}
        #        ${REALSENSE_INCLUDE_DIR}
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
        #        ${GLFW_INCLUDE_DIRS}
        #        ${Boost_INCLUDE_DIR}
        ${CERES_INCLUDE_DIRS}
)

add_executable(testing_codes test.cpp) # "name "test" is not valid in TX2
#target_include_directories(test PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} ${GLOG_INCLUDE_DIRS})
target_link_libraries(testing_codes
        my_lib
        matplotlib_cpp
        ${realsense2_LIBRARY}
        ${OpenCV_LIBS}
        ${Pangolin_LIBRARIES}
        ${GLOG_LIBRARIES}
        #        ${GLFW_LIBRARIES}
        g2o_core
        g2o_stuff
        g2o_types_sba
        g2o_solver_csparse
        g2o_csparse_extension
        Sophus::Sophus
        -lboost_serialization
        ${CERES_LIBRARIES}
)

file(GLOB_RECURSE
        SRC_LIST ${CMAKE_SOURCE_DIR}/Source/*.cu
)

foreach (EACH_SRC ${SRC_LIST})
    string(REGEX REPLACE ".+/(.+)\\..*" "\\1" EACH_EXE ${EACH_SRC})
    message(STATUS "EACH_EXE: ${EACH_EXE}")
    # generate the library
    add_executable(${EACH_EXE} ${EACH_SRC})
        target_link_libraries(${EACH_EXE}
                ${OpenCV_LIBS}
        )
    target_include_directories(${EACH_EXE} PUBLIC
            ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
            ${GLOG_INCLUDE_DIRS}
            ${PROJECT_SOURCE_DIR}/Common
    )
endforeach ()