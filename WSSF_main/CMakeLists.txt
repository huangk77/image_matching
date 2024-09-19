cmake_minimum_required(VERSION 3.1)
project(WSSF)

# ************************ 加入第三方库 ***********************
# set(OpenCV_DIRS "/home/starsz/Documents/opencv-4.10.0/opencv_install/")
find_package( OpenCV REQUIRED REQUIRED 
                PATHS /home/starsz/Documents/opencv-4.10.0/opencv_install/
                NO_DEFAULT_PATH)
include_directories( ${OpenCV_DIRS} )
message(STATUS "    OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

# ************************ 设置路径变量 ************************ 
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_CURRENT_LIST_DIR}/build)#设置可执行文件的输出目录
message("EXECUTABLE_OUTPUT_PATH is ${EXECUTABLE_OUTPUT_PATH}")
# 必须的编译文件路径
set(SRC_DIR "${CMAKE_CURRENT_LIST_DIR}/src")#cpp源码目录
set(INCLUDE_DIR "${CMAKE_CURRENT_LIST_DIR}/include")#cpp头文件目录
set(LIB_DIR "${CMAKE_CURRENT_LIST_DIR}/lib")#编译后的链接库的目录

# 打印出一些信息
message("SRC_DIR is ${SRC_DIR}")
message("INCLUDE_DIR is ${INCLUDE_DIR}")
message("LIB_DIR is ${LIB_DIR}")
 
# ************************ 设置编译的配置 ************************ 
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")#编译时产生调试信息
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O3 -pthread")
set(CMAKE_CXX_COMPILER "/usr/bin/g++")
set(CMAKE_C_COMPILER "/usr/bin/gcc")


# ************************ 把头文件添加进编译器中 ************
include_directories(${INCLUDE_DIR})
include_directories("/usr/local/include")
include_directories("/home/starsz/Documents/opencv-4.10.0/opencv_install/include/opencv4/")
# include_directories("/usr/local/include/opencv2")


link_directories(
  ${CMAKE_CURRENT_LIST_DIR}/lib
)

list(APPEND EXTRA_LIBS
  # fftw3
)

# ************************ 添加源码 ************************ 
aux_source_directory(${SRC_DIR} SOURCES_VAR)

# ************************ 开始编译 ************************
# 编译动态库
# add_library(WSSF STATIC ${SOURCES_VAR})

# 编译目标
add_executable(WSSF ${SOURCES_VAR})

# 添加额外的库
target_link_libraries(WSSF ${OpenCV_LIBS} ${EXTRA_LIBS} phasecong3 EPSIF)

