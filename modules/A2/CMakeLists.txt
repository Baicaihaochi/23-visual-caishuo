set(curr_dir ${CMAKE_CURRENT_SOURCE_DIR})

include_directories(
    include
)
link_libraries(math)
aux_source_directory(${curr_dir}/src DIR_A2_SRCS)

add_library(A2 ${DIR_A2_SRCS}) 
