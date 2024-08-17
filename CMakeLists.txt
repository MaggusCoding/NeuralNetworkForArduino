cmake_minimum_required(VERSION 3.10)
project(NeuralNetwork)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add the include directory to the include path
include_directories(${CMAKE_SOURCE_DIR}/include)

# Add all source files
file(GLOB SOURCES "src/*.cpp" "main.cpp")

# Create the executable
add_executable(NeuralNetwork ${SOURCES})

# Link against the filesystem library
target_link_libraries(NeuralNetwork stdc++fs)