cmake_minimum_required(VERSION 3.10)
project(opencv_test)

aux_source_directory(. DIR_SRCS)

find_package(OpenCV REQUIRED)
add_executable(opencv_test ${DIR_SRCS})
target_include_directories(opencv_test PUBLIC ${OpenCV_INCLUDE_DIRS})
target_link_libraries(opencv_test ${OpenCV_LIBS})
