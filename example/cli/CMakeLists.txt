add_executable(ChatCli ChatCli.cpp)

target_compile_features(ChatCli PUBLIC cxx_std_20)
target_compile_definitions(ChatCli PRIVATE ALPAKANN_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
target_link_libraries(ChatCli PUBLIC alpaka::alpaka alpakaNN::alpakaNN)
alpaka_finalize(ChatCli)
