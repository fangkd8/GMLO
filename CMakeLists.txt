cmake_minimum_required( VERSION 3.8 )
set(CMAKE_CXX_STANDARD_REQUIRED ON)
SET(CMAKE_BUILD_TYPE Release)
SET(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "-std=c++17 -O3 -fopenmp -pthread -msse -msse2 -msse3 -msse4 -msse4.1 -msse4.2")

project(GMLO)

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# PCL & OpenMP & OpenCV
find_package(PCL 1.12 REQUIRED)
find_package(OpenMP REQUIRED)
find_package(Sophus REQUIRED)
if(OPENMP_FOUND)
  message(STATUS "OPENMP FOUND")
  set(OpenMP_FLAGS ${OpenMP_CXX_FLAGS})
  set(OpenMP_LIBS gomp)
endif()

# Eigen
SET(EIGEN_INCLUDE_DIR /usr/include/eigen3)

include_directories(
  ${PCL_INCLUDE_DIRS}
  ${EIGEN_INCLUDE_DIR}
  ${Sophus_INCLUDE_DIR}
  ${PROJECT_SOURCE_DIR}/include
)

set(THIRD_PARTY_LIBS
  ${PCL_LIBRARIES} 
  ${OpenMP_LIBS}
  ${Sophus_LIBRARIES} fmt
)

add_definitions(${PCL_DEFINITIONS})

add_subdirectory(src)
add_subdirectory(exec)
