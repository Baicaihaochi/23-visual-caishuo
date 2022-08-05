set(curr_dir ${CMAKE_CURRENT_SOURCE_DIR})

include_directories(
    include
)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
aux_source_directory(${curr_dir}/src DIR_math_SRCS)

add_library(math ${DIR_math_SRCS}) 
