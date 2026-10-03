cmake_minimum_required(VERSION 3.11)
project(delivery_sim LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_SANTDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE relWithDebInfo)
endif()

option(ENABLE_TSAN "Instrument delivery_sim with ThreadSanitizer" OFF)
option(ENABLE_ASAN "Instrument delivery_sim with AddressSanitizer + LeakSanitizier" OFF)

if(ENABLE_TSAN AND ENABLE_ASAN)
  message(FATAL_ERROR "ENABLE_TSAN and ENABLE_ASAN cannot be used together")
endif()
find_package(Threads REQUIRED)
find_package(OpenCV 4.1 REQUIRED COMPONENTS core imgproc imgcodecs highgui)

file(GLOB SOURCES src/*.cpp)
add_executable(delivery_sim ${SOURCES})

target_compile_options(delivery_sim PRIVATE -Wall -Wextra)

target_include_directories(delivery_sim SYSTEM PRIVATE ${OpenCV_INCLUDE_DIRS} ${CMAKE_SOURCE_DIR}/third_party)

target_link_libraries(delivery_sim PRIVATE ${OpenCV_LIBS} Threads::Threads)
if(ENABLE_TSAN)
  target_compile_options(delivery_sim PRIVATE -fsanitize=thread -g -O1 -fno-omit-frame-pointer)
  target_link_libraries(delivery_sim PRIVATE -fsanitize=thread)
endif()

if(ENABLE_ASAN)
  target_compile_options(delivery_sim PRIVATE -fsanitize=address,undefined -g -O1 -fno-omit-frame-pointer)
  target_link_libraries(delivery_sim PRIVATE -fsanitize=address,undefined)
endif()