set(curr_dir ${CMAKE_CURRENT_SOURCE_DIR})

include_directories(
    include
)

aux_source_directory(${curr_dir}/src DIR_A1_SRCS)

add_library(A1 ${DIR_A1_SRCS}) 
