# Download tiny LLaMA model for testing
# Usage: cmake -DLIBRARY_PATH=<path> -P download_tiny_llama.cmake

cmake_minimum_required(VERSION 3.25)

set(LIBRARY_PATH "" CACHE PATH "Path where the model will be downloaded")

if(NOT LIBRARY_PATH)
    message(FATAL_ERROR "LIBRARY_PATH is required")
endif()

# Create output directory
file(MAKE_DIRECTORY ${LIBRARY_PATH})

# Download and extract model config
set(CONFIG_URL "https://huggingface.co/hf-internal-testing/tiny-random-LlamaForCausalLM/resolve/main/config.json")
set(CONFIG_FILE ${LIBRARY_PATH}/config.json)

message(STATUS "Downloading config.json...")
file(DOWNLOAD ${CONFIG_URL} ${CONFIG_FILE} SHOW_PROGRESS)

# Download model safetensors
set(MODEL_URL "https://huggingface.co/hf-internal-testing/tiny-random-LlamaForCausalLM/resolve/main/model.safetensors")
set(MODEL_FILE ${LIBRARY_PATH}/model.safetensors)

message(STATUS "Downloading model.safetensors...")
file(DOWNLOAD ${MODEL_URL} ${MODEL_FILE} SHOW_PROGRESS)

message(STATUS "Model downloaded to: ${LIBRARY_PATH}")
