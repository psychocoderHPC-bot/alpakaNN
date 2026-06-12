/*
 * Copyright 2026 René Widera
 * SPDX-License-Identifier: ISC
 */

#include <alpaka/alpaka.hpp>

#include <alpaka/nn/nn.hpp>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <vector>

namespace fs = std::filesystem;

namespace
{
    constexpr std::string_view MODEL_DIR = "models";
    constexpr std::string_view MODEL_MANIFEST = "alpaka_model.json";

    struct CliOptions
    {
        std::string modelArg = "tiny_llama";
        std::string systemPrompt;
        size_t maxNewTokens = 64u;
        bool interactive = false;
        bool showList = false;
        bool showHelp = false;
    };

    struct ModelAssets
    {
        std::string displayName;
        fs::path modelDir;
        fs::path modelFile;
        fs::path configFile;
        fs::path tokenizerFile;
        fs::path tokenizerConfigFile;
    };

    struct ChatMessage
    {
        std::string role;
        std::string content;
    };

    enum class ChatTemplateSupport
    {
        none,
        tinyLlama,
        unsupported
    };

    std::string shellQuote(std::string_view value)
    {
        std::string quoted = "'";
        for(char ch : value)
        {
            if(ch == '\'')
                quoted += "'\\''";
            else
                quoted += ch;
        }
        quoted += "'";
        return quoted;
    }

    fs::path repoRoot()
    {
#ifdef ALPAKANN_SOURCE_DIR
        return fs::path(ALPAKANN_SOURCE_DIR);
#else
        return fs::current_path();
#endif
    }

    std::string readTextFile(fs::path const& path)
    {
        std::ifstream input(path);
        if(!input)
            throw std::runtime_error("Failed to open file: " + path.string());
        return std::string((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
    }

    void writeTextFile(fs::path const& path, std::string const& content)
    {
        std::ofstream output(path);
        if(!output)
            throw std::runtime_error("Failed to write file: " + path.string());
        output << content;
    }

    std::string trim(std::string value)
    {
        auto const start = value.find_first_not_of(" \t\r\n");
        if(start == std::string::npos)
            return {};
        auto const end = value.find_last_not_of(" \t\r\n");
        return value.substr(start, end - start + 1u);
    }

    std::string extractJsonString(std::string const& json, std::string const& key)
    {
        auto const needle = "\"" + key + "\"";
        auto const keyPos = json.find(needle);
        if(keyPos == std::string::npos)
            return {};
        auto const colonPos = json.find(':', keyPos + needle.size());
        if(colonPos == std::string::npos)
            return {};
        auto const firstQuote = json.find('"', colonPos + 1u);
        if(firstQuote == std::string::npos)
            return {};
        std::string value;
        for(size_t pos = firstQuote + 1u; pos < json.size(); ++pos)
        {
            auto const ch = json[pos];
            if(ch == '"' && json[pos - 1u] != '\\')
                return value;
            value += ch;
        }
        return {};
    }

    std::string jsonQuote(std::string_view value)
    {
        std::string out = "\"";
        for(char ch : value)
        {
            switch(ch)
            {
            case '\\':
                out += "\\\\";
                break;
            case '"':
                out += "\\\"";
                break;
            case '\n':
                out += "\\n";
                break;
            case '\r':
                out += "\\r";
                break;
            case '\t':
                out += "\\t";
                break;
            default:
                out += ch;
                break;
            }
        }
        out += '"';
        return out;
    }

    std::string collectPipeOutput(std::FILE* pipe)
    {
        std::string output;
        char buffer[512];
        while(std::fgets(buffer, static_cast<int>(sizeof(buffer)), pipe) != nullptr)
            output += buffer;
        return output;
    }

    std::string runTokenizerHelper(
        fs::path const& modelDir,
        std::string_view mode,
        std::string const& payload,
        bool addBos = false,
        bool addEos = false)
    {
        auto const tmpPath
            = fs::temp_directory_path()
              / ("alpakaNN-tokenizer-" + std::to_string(::getpid()) + "-" + std::to_string(std::rand()) + ".txt");
        writeTextFile(tmpPath, payload);

        auto const helperPath = repoRoot() / "tools" / "llama_tokenizer.py";
        std::string command = "python3 " + shellQuote(helperPath.string()) + " " + std::string(mode) + " "
                              + shellQuote(modelDir.string());
        if(addBos)
            command += " --bos";
        if(addEos)
            command += " --eos";
        command += " < " + shellQuote(tmpPath.string()) + " 2>&1";

        auto* pipe = ::popen(command.c_str(), "r");
        if(pipe == nullptr)
        {
            fs::remove(tmpPath);
            throw std::runtime_error("Failed to launch tokenizer helper");
        }

        auto const output = collectPipeOutput(pipe);
        auto const rc = ::pclose(pipe);
        fs::remove(tmpPath);
        if(rc != 0)
            throw std::runtime_error("Tokenizer helper failed: " + trim(output));
        return output;
    }

    std::vector<uint32_t> parseTokenIds(std::string const& text)
    {
        std::vector<uint32_t> tokenIds;
        std::string number;
        for(char ch : text)
        {
            if(ch >= '0' && ch <= '9')
            {
                number += ch;
                continue;
            }
            if(!number.empty())
            {
                tokenIds.push_back(static_cast<uint32_t>(std::stoul(number)));
                number.clear();
            }
        }
        if(!number.empty())
            tokenIds.push_back(static_cast<uint32_t>(std::stoul(number)));
        return tokenIds;
    }

    std::string tokenIdsToJson(std::vector<uint32_t> const& tokenIds)
    {
        std::string json = "[";
        for(size_t idx = 0; idx < tokenIds.size(); ++idx)
        {
            if(idx != 0u)
                json += ",";
            json += std::to_string(tokenIds[idx]);
        }
        json += "]";
        return json;
    }

    std::string chatMessagesToJson(std::vector<ChatMessage> const& messages, bool addGenerationPrompt)
    {
        std::string json = "{\"messages\":[";
        for(size_t idx = 0; idx < messages.size(); ++idx)
        {
            if(idx != 0u)
                json += ",";
            json += "{\"role\":";
            json += jsonQuote(messages[idx].role);
            json += ",\"content\":";
            json += jsonQuote(messages[idx].content);
            json += "}";
        }
        json += "],\"add_generation_prompt\":";
        json += addGenerationPrompt ? "true" : "false";
        json += "}";
        return json;
    }

    struct PythonTokenizer
    {
        explicit PythonTokenizer(fs::path modelDirectory) : modelDir(std::move(modelDirectory))
        {
        }

        std::vector<uint32_t> encodePrompt(std::string const& text) const
        {
            return parseTokenIds(runTokenizerHelper(modelDir, "encode", text, true, false));
        }

        std::vector<uint32_t> encodeChatPrompt(std::vector<ChatMessage> const& messages) const
        {
            auto const prompt = runTokenizerHelper(modelDir, "format-chat", chatMessagesToJson(messages, true));
            return parseTokenIds(runTokenizerHelper(modelDir, "encode", prompt, true, false));
        }

        std::string decodeTokens(std::vector<uint32_t> const& tokenIds) const
        {
            return runTokenizerHelper(modelDir, "decode", tokenIdsToJson(tokenIds));
        }

        ChatTemplateSupport chatTemplateSupport() const
        {
            auto const configPath = modelDir / "tokenizer_config.json";
            if(!fs::exists(configPath))
                return ChatTemplateSupport::none;
            auto const text = readTextFile(configPath);
            auto const hasTemplate = text.find("\"chat_template\"") != std::string::npos;
            if(!hasTemplate)
                return ChatTemplateSupport::none;
            auto const isTinyLlamaTemplate = text.find("message['role'] == 'user'") != std::string::npos
                                             && text.find("message['role'] == 'assistant'") != std::string::npos
                                             && text.find("loop.last and add_generation_prompt") != std::string::npos
                                             && text.find("<|user|>") != std::string::npos
                                             && text.find("<|assistant|>") != std::string::npos;
            return isTinyLlamaTemplate ? ChatTemplateSupport::tinyLlama : ChatTemplateSupport::unsupported;
        }

        fs::path modelDir;
    };

    std::string getModelBinName(std::string_view modelName)
    {
        return std::string(modelName) + ".bin";
    }

    ModelAssets resolveModelAssets(std::string const& modelArg)
    {
        auto const root = repoRoot();
        fs::path candidate = modelArg;
        if(!candidate.is_absolute())
            candidate = root / candidate;

        ModelAssets assets{};
        if(fs::is_regular_file(candidate))
        {
            assets.modelFile = candidate;
            assets.modelDir = candidate.parent_path();
            assets.displayName = candidate.stem().string();
        }
        else
        {
            auto modelDir = candidate;
            if(!fs::is_directory(modelDir))
                modelDir = root / MODEL_DIR / modelArg;
            if(!fs::is_directory(modelDir))
                throw std::runtime_error("Model directory not found: " + modelArg);

            assets.modelDir = modelDir;
            assets.displayName = modelDir.filename().string();
            auto const manifestPath = modelDir / MODEL_MANIFEST;
            if(fs::exists(manifestPath))
            {
                auto const manifest = readTextFile(manifestPath);
                auto const binaryName = extractJsonString(manifest, "binary_name");
                if(!binaryName.empty())
                    assets.modelFile = modelDir / binaryName;
            }
            if(assets.modelFile.empty())
            {
                auto const defaultPath = modelDir / getModelBinName(assets.displayName);
                if(fs::exists(defaultPath))
                    assets.modelFile = defaultPath;
                else
                {
                    for(auto const& entry : fs::directory_iterator(modelDir))
                    {
                        if(entry.is_regular_file() && entry.path().extension() == ".bin")
                        {
                            assets.modelFile = entry.path();
                            break;
                        }
                    }
                }
            }
        }

        assets.configFile = assets.modelDir / "config.json";
        assets.tokenizerFile = assets.modelDir / "tokenizer.json";
        assets.tokenizerConfigFile = assets.modelDir / "tokenizer_config.json";
        if(assets.modelFile.empty())
            throw std::runtime_error("No .bin model file found in: " + assets.modelDir.string());
        if(!fs::exists(assets.configFile))
            throw std::runtime_error("Missing config.json in: " + assets.modelDir.string());
        if(!fs::exists(assets.tokenizerFile))
            throw std::runtime_error("Missing tokenizer.json in: " + assets.modelDir.string());
        return assets;
    }

    bool modelInstalled(std::string const& modelArg)
    {
        try
        {
            auto const assets = resolveModelAssets(modelArg);
            return fs::exists(assets.modelFile);
        }
        catch(...)
        {
            return false;
        }
    }

    void listModels()
    {
        auto const modelsRoot = repoRoot() / MODEL_DIR;
        std::cout << "Available models:" << std::endl;
        if(!fs::exists(modelsRoot))
        {
            std::cout << "  No models downloaded yet." << std::endl;
            return;
        }

        for(auto const& entry : fs::directory_iterator(modelsRoot))
        {
            if(!entry.is_directory())
                continue;
            std::cout << "  - " << entry.path().filename().string();
            auto const manifestPath = entry.path() / MODEL_MANIFEST;
            if(fs::exists(manifestPath))
            {
                auto const manifest = readTextFile(manifestPath);
                auto const status = extractJsonString(manifest, "status");
                if(!status.empty())
                    std::cout << " (" << status << ")";
            }
            std::cout << std::endl;
        }
    }

    void downloadModel(std::string const& modelName)
    {
        auto const modelDir = repoRoot() / MODEL_DIR / modelName;
        fs::create_directories(modelDir);
        auto const downloader = repoRoot() / "tools" / "download_tiny_llama.py";
        std::string command = "python3 " + shellQuote(downloader.string()) + " " + shellQuote(modelName) + " "
                              + shellQuote(modelDir.string());
        auto const rc = std::system(command.c_str());
        if(rc != 0)
            throw std::runtime_error("Failed to download model: " + modelName);
    }

    CliOptions parseArgs(int argc, char* argv[])
    {
        CliOptions options{};
        for(int i = 1; i < argc; ++i)
        {
            std::string_view arg = argv[i];
            if(arg == "-m" || arg == "--model")
            {
                if(i + 1 >= argc)
                    throw std::runtime_error("--model requires a value");
                options.modelArg = argv[++i];
            }
            else if(arg == "-i" || arg == "--interactive")
            {
                options.interactive = true;
            }
            else if(arg == "--system-prompt")
            {
                if(i + 1 >= argc)
                    throw std::runtime_error("--system-prompt requires a value");
                options.systemPrompt = argv[++i];
            }
            else if(arg == "--max-new-tokens")
            {
                if(i + 1 >= argc)
                    throw std::runtime_error("--max-new-tokens requires a value");
                auto const value = std::stoul(argv[++i]);
                if(value == 0u)
                    throw std::runtime_error("--max-new-tokens must be greater than zero");
                options.maxNewTokens = value;
            }
            else if(arg == "-l" || arg == "--list")
            {
                options.showList = true;
            }
            else if(arg == "-h" || arg == "--help")
            {
                options.showHelp = true;
            }
            else
            {
                throw std::runtime_error("Unknown argument: " + std::string(arg));
            }
        }
        return options;
    }

    bool isNamedModelPreset(std::string const& modelArg)
    {
        return modelArg.find('/') == std::string::npos && modelArg.find('\\') == std::string::npos;
    }

    void printHelp(char const* argv0)
    {
        std::cout << "Usage: " << argv0 << " [options]\n\n"
                  << "Without arguments the CLI runs a tiny_llama self-test and exits.\n\n"
                  << "Options:\n"
                  << "  -i, --interactive     Start interactive chat mode\n"
                  << "  -m, --model MODEL     Model name, model directory, or .bin path\n"
                  << "      --system-prompt TEXT\n"
                  << "                        Add an initial system message in interactive mode\n"
                  << "      --max-new-tokens N\n"
                  << "                        Maximum tokens to generate per assistant turn (default: 64)\n"
                  << "  -l, --list            List downloaded models\n"
                  << "  -h, --help            Show this help message\n";
    }

    template<typename T_Model>
    void runSelfTest(auto& queue, auto exec, T_Model const& model, PythonTokenizer const& tokenizer)
    {
        std::string const prompt = "User: Hello\nAssistant:";
        auto const promptTokens = tokenizer.encodePrompt(prompt);
        if(promptTokens.empty())
            throw std::runtime_error("Self-test prompt encoding returned no tokens");

        auto generated = alpaka::nn::onHost::inference::generateGreedy(queue, exec, model, promptTokens, 8u);
        if(generated.size() <= promptTokens.size())
            throw std::runtime_error("Self-test generation returned no new tokens");

        std::vector<uint32_t> newTokens(
            generated.begin() + static_cast<std::ptrdiff_t>(promptTokens.size()),
            generated.end());
        auto const decoded = tokenizer.decodeTokens(newTokens);

        std::cout << "=== alpakaNN Chat CLI Self-Test ===" << std::endl;
        std::cout << "Model: tiny_llama (CI/smoke only)" << std::endl;
        std::cout << "Prompt token count: " << promptTokens.size() << std::endl;
        std::cout << "Generated token ids:";
        for(auto tokenId : newTokens)
            std::cout << ' ' << tokenId;
        std::cout << std::endl;
        std::cout << "Decoded suffix: " << decoded << std::endl;
        std::cout << "SELF-TEST PASSED" << std::endl;
    }

    template<typename T_Model>
    void runInteractiveChat(
        auto& queue,
        auto exec,
        T_Model const& model,
        PythonTokenizer const& tokenizer,
        CliOptions const& options)
    {
        auto const templateSupport = tokenizer.chatTemplateSupport();
        if(templateSupport == ChatTemplateSupport::unsupported)
        {
            throw std::runtime_error(
                "Model exposes an unsupported chat_template; only the TinyLlama minimal role-marker template is "
                "supported");
        }

        std::cout << "Interactive chat mode. Type 'quit' or 'exit' to end." << std::endl;
        std::cout << "Note: useful English output requires a trained supported Llama-family model." << std::endl;

        std::vector<ChatMessage> messages;
        std::string transcript;
        if(!options.systemPrompt.empty())
        {
            messages.push_back(ChatMessage{"system", options.systemPrompt});
            transcript = "System: " + options.systemPrompt;
        }
        std::string line;
        while(true)
        {
            std::cout << "> " << std::flush;
            if(!std::getline(std::cin, line))
                break;
            if(line == "quit" || line == "exit")
                break;
            if(line.empty())
                continue;

            std::vector<uint32_t> promptTokens;
            if(templateSupport == ChatTemplateSupport::tinyLlama)
            {
                messages.push_back(ChatMessage{"user", line});
                promptTokens = tokenizer.encodeChatPrompt(messages);
            }
            else
            {
                if(!transcript.empty())
                    transcript += '\n';
                transcript += "User: " + line + "\nAssistant:";
                promptTokens = tokenizer.encodePrompt(transcript);
            }
            auto generated
                = alpaka::nn::onHost::inference::generateGreedy(queue, exec, model, promptTokens, options.maxNewTokens);
            std::vector<uint32_t> newTokens(
                generated.begin() + static_cast<std::ptrdiff_t>(promptTokens.size()),
                generated.end());
            auto response = trim(tokenizer.decodeTokens(newTokens));
            if(response.empty())
                throw std::runtime_error("Assistant reply was empty after trimming");
            std::cout << "Assistant: " << response << std::endl;
            if(templateSupport == ChatTemplateSupport::tinyLlama)
                messages.push_back(ChatMessage{"assistant", response});
            else
                transcript += response;
        }
    }
} // namespace

int main(int argc, char* argv[])
{
    try
    {
        auto const options = parseArgs(argc, argv);
        if(options.showHelp)
        {
            printHelp(argv[0]);
            return 0;
        }
        if(options.showList)
        {
            listModels();
            return 0;
        }

        std::srand(static_cast<unsigned>(::getpid()));

        if(!modelInstalled(options.modelArg))
        {
            if(!isNamedModelPreset(options.modelArg))
                throw std::runtime_error("Model path not found: " + options.modelArg);
            std::cout << "Model '" << options.modelArg << "' not found locally. Downloading..." << std::endl;
            downloadModel(options.modelArg);
        }

        auto const assets = resolveModelAssets(options.modelArg);
        auto backends
            = alpaka::onHost::allBackends(alpaka::onHost::enabledDeviceSpecs, alpaka::exec::enabledExecutors);
        auto cfg = std::get<0>(backends);
        auto selector = alpaka::onHost::makeDeviceSelector(cfg[alpaka::object::deviceSpec]);
        auto device = selector.makeDevice(0);
        auto queue = device.makeQueue();
        auto exec = cfg[alpaka::object::exec];

        auto model = alpaka::nn::onHost::model::loadTinyLlama<float>(device, assets.modelFile.string());
        PythonTokenizer tokenizer(assets.modelDir);

        if(options.interactive)
            runInteractiveChat(queue, exec, model, tokenizer, options);
        else
            runSelfTest(queue, exec, model, tokenizer);

        return 0;
    }
    catch(std::exception const& error)
    {
        std::cerr << "Error: " << error.what() << std::endl;
        return 1;
    }
}
