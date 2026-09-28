cmake_minimum_required(VERSION 3.10)
project(SNNSimulator VERSION 1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
if(CMAKE_CROSSCOMPILING AND CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
  set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=armv8-a+crc")
else()
  set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native")
endif()
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")

include(CTest)
enable_testing()

find_package(CLI11 CONFIG REQUIRED)
find_package(OpenMP REQUIRED)

# RLTools (header-only)
set(RL_TOOLS_INCLUDE_DIR external/rl-tools/include
    CACHE PATH "Directorio de cabeceras de RLTools")
add_library(rl_tools INTERFACE)
target_include_directories(rl_tools INTERFACE ${RL_TOOLS_INCLUDE_DIR})

# Core del simulador
add_library(snn_core
    src/core/neuron.cpp
    src/core/network.cpp
)

target_include_directories(snn_core PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)

# Codificador del simulador
add_library(snn_encoding
    src/encoding/encoder.cpp
    src/encoding/poissonEncoder.cpp
    src/encoding/rateEncoder.cpp
    src/encoding/rlEncoder.cpp
    src/encoding/ttfsEncoder.cpp
)

target_include_directories(snn_encoding PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)

# Motor + decodificador
add_library(snn_engine
    src/core/simulator.cpp
    src/decoding/decoder.cpp
)

target_include_directories(snn_engine PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(snn_engine
    PUBLIC
        snn_core
        snn_encoding
        OpenMP::OpenMP_CXX
)

# Simulador principal
add_executable(snn_simulator
    src/core/main.cpp
)
target_link_libraries(snn_simulator 
    PRIVATE
        snn_engine
        CLI11::CLI11
)

# Tests — gated behind BUILD_TESTING (from include(CTest) above, default ON)
# so environments without Catch2 3 (e.g. the ZCU104/PYNQ board, which only
# ships Catch2 2.13.8 via apt) can still build the core libraries with
# -DBUILD_TESTING=OFF, without needing Catch2 at all.
if(BUILD_TESTING)
  find_package(Catch2 3 REQUIRED)

  add_executable(snn_tests
      tests/core/networkTests.cpp
      tests/core/neuronTests.cpp
      tests/core/izhikevichTests.cpp
  )
  target_link_libraries(snn_tests
      PRIVATE
          snn_core
          Catch2::Catch2WithMain
  )

  include(Catch)
  catch_discover_tests(snn_tests)

  # BipedalWalker en C++ puro: solo cabeceras, sin dependencias del simulador SNN.
  # -ffp-contract=off evita FMA para que los resultados float32 sean comparables con Box2D
  # (el binario de producción para la A53 sí puede fusionar multiplicaciones-sumas).
  add_executable(bipedal_tests
      tests/rl/bipedal_walker/geometryTests.cpp
      tests/rl/bipedal_walker/terrainTests.cpp
      tests/rl/bipedal_walker/freeWalkerTests.cpp
      tests/rl/bipedal_walker/groundWalkerTests.cpp
      tests/rl/bipedal_walker/resetTests.cpp
      tests/rl/bipedal_walker/envTests.cpp
      tests/rl/bipedal_walker/hardcoreTerrainTests.cpp
      tests/rl/bipedal_walker/obstacleTests.cpp
      tests/rl/bipedal_walker/hardcoreLidarTests.cpp
  )
  target_include_directories(bipedal_tests PRIVATE ${PROJECT_SOURCE_DIR}/include)
  target_compile_options(bipedal_tests PRIVATE -ffp-contract=off -Wdouble-promotion)
  target_compile_definitions(bipedal_tests PRIVATE
      BIPEDAL_TEST_DATA_DIR="${PROJECT_SOURCE_DIR}/tests/data/bipedal_walker")
  target_link_libraries(bipedal_tests PRIVATE Catch2::Catch2WithMain rl_tools)
  catch_discover_tests(bipedal_tests)

  # Benchmark de host: coste por paso según las iteraciones del solver (ver comentario en el
  # archivo). No usa Catch2 y no corre en ctest; se ejecuta a mano.
  add_executable(bipedal_bench
      tests/rl/bipedal_walker/bench_solver.cpp
  )
  target_include_directories(bipedal_bench PRIVATE ${PROJECT_SOURCE_DIR}/include)
  set_target_properties(bipedal_bench PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/build
  )

  # Estudio de fidelidad por ranking (controladores aleatorios vs Box2D). No usa Catch2.
  add_executable(bipedal_ranking_eval
      tests/rl/bipedal_walker/ranking_eval.cpp
  )
  target_include_directories(bipedal_ranking_eval PRIVATE ${PROJECT_SOURCE_DIR}/include)
  target_link_libraries(bipedal_ranking_eval PRIVATE rl_tools)
  set_target_properties(bipedal_ranking_eval PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/build
  )
endif()

set_target_properties(snn_simulator PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/build
)

# Inyección de corriente directa en neuronas individuales
add_executable(snn_current_injection
    src/core/current_injection_main.cpp
)
target_link_libraries(snn_current_injection
    PRIVATE
        snn_core
        CLI11::CLI11
)
set_target_properties(snn_current_injection PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/build
)

# Evaluación del péndulo con RLTools
add_executable(snn_pendulum
    src/rl/pendulum_eval.cpp
)
target_link_libraries(snn_pendulum
    PRIVATE
        snn_engine
        rl_tools
)
set_target_properties(snn_pendulum PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/build
)

# Evaluación del acrobot con RLTools
add_executable(snn_acrobot
    src/rl/acrobot_eval.cpp
)
target_link_libraries(snn_acrobot
    PRIVATE
        snn_engine
        rl_tools
)
set_target_properties(snn_acrobot PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/build
)

# Evaluación del BipedalWalker (port a C++, sin Box2D) con RLTools
add_executable(snn_bipedal_walker
    src/rl/bipedal_walker_eval.cpp
)
target_link_libraries(snn_bipedal_walker
    PRIVATE
        snn_engine
        rl_tools
)
set_target_properties(snn_bipedal_walker PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/build
)

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "C++ flags: ${CMAKE_CXX_FLAGS}")

