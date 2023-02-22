cmake_minimum_required(VERSION 3.15.3)
project(VideoMixer)

set(CMAKE_CXX_STANDARD 20)

find_package(PkgConfig REQUIRED)
pkg_search_module(gstreamer REQUIRED IMPORTED_TARGET gstreamer-1.0>=1.4)
pkg_search_module(gstreamer-app REQUIRED IMPORTED_TARGET gstreamer-app-1.0>=1.4)

add_executable(VideoMixer src/mixer.c)

target_link_libraries(VideoMixer
    PkgConfig::gstreamer
    PkgConfig::gstreamer-app
)