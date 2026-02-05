cmake_minimum_required(VERSION 3.10)
project(SNNSimulator VERSION 1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")

include(CTest)
enable_testing()

find_package(CLI11 CONFIG REQUIRED)
find_package(Catch2 3 REQUIRED)

add_library(snn_core
    src/core/neuron.cpp
    src/core/network.cpp
)

target_include_directories(snn_core PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)

# Simulador principal
add_executable(snn_simulator
    src/core/main.cpp
)
target_link_libraries(snn_simulator 
    PRIVATE 
        snn_core
        CLI11::CLI11
)

# Tests
add_executable(snn_tests
    tests/core/networkTests.cpp
    tests/core/neuronTests.cpp
)
target_link_libraries(snn_tests
    PRIVATE
        snn_core
        Catch2::Catch2WithMain
)

include(Catch)
catch_discover_tests(snn_tests)

set_target_properties(snn_simulator PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/build
)

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "C++ flags: ${CMAKE_CXX_FLAGS}")

