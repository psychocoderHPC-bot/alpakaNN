add_executable(heatEquationNn src/heatEquationNn.cpp)
target_compile_features(heatEquationNn PRIVATE cxx_std_20)
target_include_directories(heatEquationNn PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
if(BUILD_TESTING)
    add_executable(heatEquationNn_solver_test test/solver_test.cpp)
    target_compile_features(heatEquationNn_solver_test PRIVATE cxx_std_20)
    add_test(NAME heatEquationNn_strict_solver COMMAND heatEquationNn_solver_test)
    add_executable(heatEquationNn_device_solver_test test/device_solver_test.cpp)
    target_compile_features(heatEquationNn_device_solver_test PRIVATE cxx_std_20)
    target_link_libraries(heatEquationNn_device_solver_test PRIVATE alpaka::nn alpaka::alpaka)
    target_compile_definitions(
        heatEquationNn_device_solver_test
        PRIVATE HEAT_CLOSURE_MODEL_DIR="${PROJECT_SOURCE_DIR}/models/heat_closure"
    )
    alpaka_finalize(heatEquationNn_device_solver_test)
    add_test(NAME heatEquationNn_device_solver COMMAND heatEquationNn_device_solver_test)
endif()
target_link_libraries(heatEquationNn PRIVATE alpaka::nn alpaka::alpaka)
alpaka_finalize(heatEquationNn)
if(BUILD_TESTING)
    add_executable(heatEquationNn_model_loader_test test/model_loader_test.cpp)
    target_compile_features(heatEquationNn_model_loader_test PRIVATE cxx_std_20)
    target_compile_definitions(
        heatEquationNn_model_loader_test
        PRIVATE HEAT_CLOSURE_MODEL_DIR="${PROJECT_SOURCE_DIR}/models/heat_closure"
    )
    add_test(NAME heatEquationNn_model_loader COMMAND heatEquationNn_model_loader_test)
endif()
