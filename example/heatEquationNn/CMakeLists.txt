add_executable(heatEquationNn src/heatEquationNn.cpp)
target_compile_features(heatEquationNn PRIVATE cxx_std_20)
if(BUILD_TESTING)
    add_executable(heatEquationNn_solver_test test/solver_test.cpp)
    target_compile_features(heatEquationNn_solver_test PRIVATE cxx_std_20)
    add_test(NAME heatEquationNn_strict_solver COMMAND heatEquationNn_solver_test)
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
