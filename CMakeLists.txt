cmake_minimum_required(VERSION 3.20)
project(Face_Recognition)

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
set(CMAKE_CXX_STANDARD 11)

add_executable(Face_Recognition main.cpp)
target_link_libraries( Face_Recognition ${OpenCV_LIBS} )

