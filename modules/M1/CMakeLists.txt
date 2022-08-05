set(curr_dir ${CMAKE_CURRENT_SOURCE_DIR})

include_directories(
    include
    ${PROJECT_SOURCE_DIR}/modules/A1/include
    ${PROJECT_SOURCE_DIR}/modules/A2/include
)
link_libraries(A1)
aux_source_directory(${curr_dir}/src DIR_M1_SRCS)

add_library(M1 ${DIR_M1_SRCS}) 
