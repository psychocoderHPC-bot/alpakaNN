cmake_minimum_required(VERSION 3.25)

set(LIBRARY_PATH "" CACHE PATH "Path where the model will be downloaded")

if(NOT LIBRARY_PATH)
    message(FATAL_ERROR "LIBRARY_PATH is required")
endif()

file(MAKE_DIRECTORY "${LIBRARY_PATH}")

execute_process(
    COMMAND python3 "${CMAKE_CURRENT_LIST_DIR}/../tools/download_tiny_llama.py" tiny_llama "${LIBRARY_PATH}"
    RESULT_VARIABLE DOWNLOAD_RESULT
)

if(NOT DOWNLOAD_RESULT EQUAL 0)
    message(FATAL_ERROR "Failed to download tiny_llama via tools/download_tiny_llama.py")
endif()
