add_executable(heatEquationNn src/heatEquationNn.cpp)
target_compile_features(heatEquationNn PRIVATE cxx_std_20)
target_include_directories(heatEquationNn PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
if(BUILD_TESTING)
    add_executable(heatEquationNn_solver_test test/solver_test.cpp)
    target_compile_features(heatEquationNn_solver_test PRIVATE cxx_std_20)
    add_test(NAME heatEquationNn_strict_solver COMMAND heatEquationNn_solver_test)
    add_executable(heatEquationNn_device_solver_test test/device_solver_test.cpp)
    target_compile_features(heatEquationNn_device_solver_test PRIVATE cxx_std_20)
    # alpaka_finalize copies sources into the build tree for HIP/CUDA. Use target
    # include directories instead of `../src/...` / `ModelFixture.hpp` so the header
    # lookup survives the copy.
    target_include_directories(
        heatEquationNn_device_solver_test
        PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/test
    )
    target_link_libraries(heatEquationNn_device_solver_test PRIVATE alpaka::nn alpaka::alpaka)
    # HEAT_CLOSURE_MODEL_DIR only names the optional real checkpoint location; the
    # test builds a fixture when no trained weights.bin is present, so configure
    # and test do not require the (intentionally uncommitted) binary.
    target_compile_definitions(
        heatEquationNn_device_solver_test
        PRIVATE
            HEAT_CLOSURE_MODEL_DIR="${PROJECT_SOURCE_DIR}/models/heat_closure"
            HEAT_CLOSURE_EXAMPLE_BIN="$<TARGET_FILE:heatEquationNn>"
    )
    # The CLI export/timing checks below execute the example binary, so it must be built first.
    add_dependencies(heatEquationNn_device_solver_test heatEquationNn)
    alpaka_finalize(heatEquationNn_device_solver_test)
    add_test(NAME heatEquationNn_device_solver COMMAND heatEquationNn_device_solver_test)
endif()
target_link_libraries(heatEquationNn PRIVATE alpaka::nn alpaka::alpaka)
alpaka_finalize(heatEquationNn)
if(BUILD_TESTING)
    # The host selector must remain accepted and run to completion without touching the filesystem.
    add_test(
        NAME heatEquationNn_backend_host
        COMMAND heatEquationNn --material preset --backend host --grid 16 --tmax 0.01 --no-output
    )
    # An unknown selector must be rejected with a non-zero exit code.
    add_test(NAME heatEquationNn_backend_invalid COMMAND heatEquationNn --backend __invalid__ --no-output)
    set_tests_properties(heatEquationNn_backend_invalid PROPERTIES WILL_FAIL TRUE)
endif()
if(BUILD_TESTING)
    add_executable(heatEquationNn_model_loader_test test/model_loader_test.cpp)
    target_compile_features(heatEquationNn_model_loader_test PRIVATE cxx_std_20)
    add_test(NAME heatEquationNn_model_loader COMMAND heatEquationNn_model_loader_test)
endif()
