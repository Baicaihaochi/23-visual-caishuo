set(curr_dir ${CMAKE_CURRENT_SOURCE_DIR})

include_directories(
    include
    ${PROJECT_SOURCE_DIR}/modules/A1/include
    ${PROJECT_SOURCE_DIR}/modules/A2/include
    ${PROJECT_SOURCE_DIR}/modules/M1/include
    ${PROJECT_SOURCE_DIR}/common/kalman/include
)
find_package(OpenCV REQUIRED)
link_libraries(A1 A2)
include_directories(${OpenCV_INCLUDE_DIRS})
aux_source_directory(${curr_dir}/src DIR_M2_SRCS)

add_library(M2 ${DIR_M2_SRCS}) 
