# 在此处进行 CMake 设置（最小版本号，项目名）
cmake_minimum_required(VERSION 3.10)
project(Test)
####### 以下禁止修改 #######
add_subdirectory(common)   #
add_subdirectory(modules)  #
####### 以上禁止修改 #######

# 在此处生成可执行程序
set(curr_dir ${CMAKE_CURRENT_SOURCE_DIR})
set(common_dir ${curr_dir}/common)
set(modules_dir ${curr_dir}/modules)
find_package(OpenCV REQUIRED)
add_executable(test main.cpp)

# 在此处链接库
target_include_directories(test PUBLIC ${OpenCV_INCLUDE_DIRS})
target_link_libraries(test ${OpenCV_LIBS})
target_link_libraries(test A1 A2 M1 M2 math)
include_directories(
    ${common_dir}/math/include
    ${common_dir}/kalman/include
    ${modules_dir}/A1/include
    ${modules_dir}/A2/include
    ${modules_dir}/M1/include
    ${modules_dir}/M2/include
)
