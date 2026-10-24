cmake_minimum_required(VERSION 3.12)
project(PerfectPaoCPP LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

if(MSVC)
  add_compile_options(/bigobj /utf-8)
else()
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86_64|AMD64)")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
  else()
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
  endif()
endif()

find_package(Threads REQUIRED)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_library(perfectpao STATIC
  src/pdk/cards.cpp
  src/pdk/moves.cpp
  src/pdk/oracle.cpp
  src/pdk/game.cpp
  src/pdk/features.cpp
  src/pdk/pimc_ai.cpp
  src/algo/rollout.cpp
  src/algo/ppo.cpp
  src/algo/eval_rule.cpp
  src/algo/eval_pimc.cpp
  src/torch_nn/models.cpp
  src/torch_nn/ppo_learner.cpp
  src/torch_nn/tensor_bridge.cpp
)
target_include_directories(perfectpao PUBLIC src ${TORCH_INCLUDE_DIRS})
target_link_libraries(perfectpao PUBLIC Threads::Threads ${TORCH_LIBRARIES})

add_executable(perfectpao_train src/train.cpp)
target_link_libraries(perfectpao_train PRIVATE perfectpao)
if(WIN32)
  target_link_libraries(perfectpao_train PRIVATE powrprof)
endif()

add_executable(perfectpao_eval src/eval_main.cpp)
target_link_libraries(perfectpao_eval PRIVATE perfectpao)

enable_testing()
include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.tar.gz
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)
include(GoogleTest)

add_executable(test_rules tests/test_rules.cpp)
target_link_libraries(test_rules PRIVATE perfectpao GTest::gtest_main)
gtest_discover_tests(test_rules)

add_executable(test_algo tests/test_algo.cpp)
target_link_libraries(test_algo PRIVATE perfectpao GTest::gtest_main)
gtest_discover_tests(test_algo)

add_executable(test_torch_nn tests/test_torch_nn.cpp)
target_link_libraries(test_torch_nn PRIVATE perfectpao GTest::gtest_main)
gtest_discover_tests(test_torch_nn)

add_executable(test_pimc tests/test_pimc.cpp)
target_link_libraries(test_pimc PRIVATE perfectpao GTest::gtest_main)
gtest_discover_tests(test_pimc)
