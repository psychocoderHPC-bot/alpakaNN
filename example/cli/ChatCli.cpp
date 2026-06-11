/*
 * Copyright 2026 René Widera
 * SPDX-License-Identifier: ISC
 */

#include <alpaka/alpaka.hpp>
#include <alpakaNN/alpakaNN.hpp>

#include <algorithm>
#include <charconv>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <string_view>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace fs = std::filesystem;

namespace
{
    constexpr std::string_view MODEL_DIR = "models";
    constexpr std::string_view CONFIG_FILE = "config.json";

    std::string getModelBinName(std::string_view modelName)
    {
        return std::string(modelName) + ".bin";
    }

    std::string getBasePath()
    {
        auto cwd = fs::current_path();
        return cwd.string();
    }

    std::string getModelPath(std::string_view modelName)
    {
        return fs::path(getBasePath()) / MODEL_DIR / modelName / getModelBinName(modelName);
    }

    bool modelExists(std::string_view modelName)
    {
        return fs::exists(getModelPath(modelName));
    }

    void listModels()
    {
        std::cout << "Available models:" << std::endl;
        auto base = fs::path(getBasePath()) / MODEL_DIR;
        if(fs::exists(base))
        {
            for(const auto& entry : fs::directory_iterator(base))
            {
                if(entry.is_directory())
                {
                    std::cout << "  - " << entry.path().filename().string() << std::endl;
                }
            }
        }
        else
        {
            std::cout << "  No models downloaded yet." << std::endl;
        }
    }

    void downloadModel(std::string_view modelName)
    {
        std::cout << "Downloading model: " << modelName << std::endl;

        auto modelPath = fs::path(getBasePath()) / MODEL_DIR / modelName;
        fs::create_directories(modelPath);

        std::cout << "  Converting to binary format..." << std::endl;
        std::string downloadCmd = "python3 " + 
            std::string(fs::path(getBasePath()) / "tools" / "download_tiny_llama.py") + 
            " " + std::string(modelName) + 
            " " + modelPath.string();
        
        if(std::system(downloadCmd.c_str()) != 0)
        {
            std::cerr << "Failed to download or convert model" << std::endl;
            return;
        }

        std::cout << "Model downloaded successfully to: " << modelPath << std::endl;
    }

    struct Tokenizer
    {
        std::vector<std::string> vocab;
        std::unordered_map<std::string, uint32_t> tokenToId;

        Tokenizer(std::string_view modelDir, std::string_view modelName)
        {
            fs::path vocabPath = fs::path(modelDir) / modelName / "vocab.json";
            
            std::ifstream vocabFile(vocabPath.string());
            if(!vocabFile)
            {
                throw std::runtime_error("Failed to open vocab file: " + vocabPath.string());
            }

            std::string content((std::istreambuf_iterator<char>(vocabFile)), std::istreambuf_iterator<char>());
            vocabFile.close();

            // Parse vocab.json array format: ["<unk>", "<s>", ...]
            // Each token is a quoted string, with optional int index before it
            size_t pos = 0;
            uint32_t id = 0;
            
            while(pos < content.length())
            {
                // Skip whitespace, commas, and opening bracket
                while(pos < content.length() && 
                      (content[pos] == ' ' || content[pos] == '\t' || 
                       content[pos] == '\n' || content[pos] == '\r' || 
                       content[pos] == ',' || content[pos] == '[' || content[pos] == ']'))
                    pos++;
                
                if(pos >= content.length())
                    break;
                
                // Now we should be at a quoted string
                if(content[pos] == '"')
                {
                    pos++; // Skip opening quote
                    size_t quoteEnd = content.find('"', pos);
                    if(quoteEnd == std::string::npos)
                        break;
                    
                    std::string token = content.substr(pos, quoteEnd - pos);
                    pos = quoteEnd + 1; // Skip closing quote
                    
                    // Resize vocab if needed
                    if(id >= vocab.size())
                        vocab.resize(id + 1);
                    
                    vocab[id] = token;
                    tokenToId[token] = id;
                    
                    id++;
                }
                else
                {
                    // Skip this character and continue
                    pos++;
                }
            }
        }

        std::vector<uint32_t> encode(std::string_view text) const
        {
            std::vector<uint32_t> tokens;
            tokens.push_back(1);

            std::string lowerText;
            for(char c : text)
            {
                lowerText += std::tolower(c);
            }

            std::string currentToken;
            for(size_t i = 0; i < lowerText.length(); ++i)
            {
                currentToken += lowerText[i];

                bool found = false;
                for(int j = currentToken.length(); j > 0; --j)
                {
                    std::string substr = currentToken.substr(0, j);
                    auto it = std::find(vocab.begin(), vocab.end(), substr);
                    if(it != vocab.end())
                    {
                        tokens.push_back(std::distance(vocab.begin(), it));
                        currentToken = currentToken.substr(j);
                        found = true;
                        break;
                    }
                }

                if(!found)
                {
                    currentToken = "";
                }
            }

            if(!currentToken.empty())
            {
                for(char c : currentToken)
                {
                    auto it = std::find(vocab.begin(), vocab.end(), std::string(1, c));
                    if(it != vocab.end())
                    {
                        tokens.push_back(std::distance(vocab.begin(), it));
                    }
                }
            }

            tokens.push_back(2);
            return tokens;
        }

        std::string decode(const std::vector<uint32_t>& tokens) const
        {
            std::string result;
            for(size_t i = 1; i < tokens.size() - 1; ++i)
            {
                if(tokens[i] < vocab.size())
                {
                    result += vocab[tokens[i]];
                }
            }
            return result;
        }
    };
}

int main(int argc, char* argv[])
{
    std::string modelPath;
    bool showList = false;

    for(int i = 1; i < argc; ++i)
    {
        std::string_view arg = argv[i];
        if(arg == "--model" || arg == "-m")
        {
            if(i + 1 < argc)
            {
                modelPath = argv[++i];
            }
            else
            {
                std::cerr << "Error: --model requires a value" << std::endl;
                return 1;
            }
        }
        else if(arg == "--list" || arg == "-l")
        {
            showList = true;
        }
        else if(arg == "--help" || arg == "-h")
        {
            std::cout << "Usage: " << argv[0] << " [options]" << std::endl;
            std::cout << std::endl;
            std::cout << "Options:" << std::endl;
            std::cout << "  --model NAME, -m NAME  Specify model name to use" << std::endl;
            std::cout << "  --list, -l             List available models" << std::endl;
            std::cout << "  --help, -h             Show this help message" << std::endl;
            std::cout << std::endl;
            std::cout << "Examples:" << std::endl;
            std::cout << "  " << argv[0] << " --list" << std::endl;
            std::cout << "  " << argv[0] << " --model tiny_llama" << std::endl;
            return 0;
        }
    }

    if(showList)
    {
        listModels();
        return 0;
    }

    if(modelPath.empty())
    {
        modelPath = "tiny_llama";
    }

    auto modelFilePath = getModelPath(modelPath);
    
    if(!modelExists(modelPath))
    {
        std::cout << "Model '" << modelPath << "' not found. Downloading..." << std::endl;
        downloadModel(modelPath);
    }

    if(!modelExists(modelPath))
    {
        std::cerr << "Failed to download or find model: " << modelPath << std::endl;
        return 1;
    }
    
    std::cout << "Loading model: " << modelPath << std::endl;

    try
    {
        auto backends = alpaka::onHost::allBackends(
            alpaka::onHost::enabledDeviceSpecs,
            alpaka::exec::enabledExecutors);
        auto cfg = std::get<0>(backends);
        auto selector = alpaka::onHost::makeDeviceSelector(cfg[alpaka::object::deviceSpec]);
        auto device = selector.makeDevice(0);
        auto queue = device.makeQueue();
        auto exec = cfg[alpaka::object::exec];

        auto model = alpakaNN::model::loadTinyLlama<float>(device, modelFilePath);

        Tokenizer tokenizer(MODEL_DIR.data(), modelPath);

        std::cout << "Model loaded successfully!" << std::endl;
        
        if(modelPath == "tiny_llama" && argc <= 1)
        {
            std::cout << "=== alpakaNN Chat CLI - Self Test ===" << std::endl;
            std::cout << std::endl;

            std::cout << "Running self-test with sample input..." << std::endl;
            std::cout << "---" << std::endl;

            std::string testInput = "Hello";
            auto testTokens = tokenizer.encode(testInput);
            std::cout << "Tokens: ";
            for(auto t : testTokens) std::cout << t << " ";
            std::cout << "(size=" << testTokens.size() << ")" << std::endl;
            std::cout << "About to call generateGreedy..." << std::endl;
            auto responseTokens = alpakaNN::inference::generateGreedy(
                queue, exec, model, testTokens, 20u);
            auto response = tokenizer.decode(responseTokens);
            std::cout << "Decoded response!" << std::endl;

            std::cout << "> " << testInput << std::endl;
            std::cout << response << std::endl;
            std::cout << "---" << std::endl;
            std::cout << std::endl;
        }

        std::cout << "Starting interactive chat. Type 'quit' or 'exit' to end." << std::endl;
        std::cout << "Usage: " << argv[0] << " --model tiny_llama" << std::endl;
        std::cout << "  or simply: " << argv[0] << std::endl;
        std::cout << std::endl;

        std::vector<uint32_t> chatHistory;
        chatHistory.push_back(1);

        std::string line;
        while(std::getline(std::cin, line))
        {
            if(line == "quit" || line == "exit")
            {
                std::cout << "Goodbye!" << std::endl;
                break;
            }

            if(line.empty())
                continue;

            auto promptTokens = tokenizer.encode(line);
            chatHistory.insert(chatHistory.end(), promptTokens.begin(), promptTokens.end());

            std::cout << "> " << line << std::endl;

            auto responseTokens = alpakaNN::inference::generateGreedy(
                queue, exec, model, chatHistory, 50u);

            auto response = tokenizer.decode(responseTokens);
            std::cout << response << std::endl;

            chatHistory = responseTokens;
        }
    }
    catch(const std::exception& e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
