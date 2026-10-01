// SPDX-License-Identifier: MPL-2.0
#include "../src/ModelLoader.hpp"

#include <iostream>
#include <stdexcept>
int main()
try
{
    auto model = heatclosure::loadModel(std::string(HEAT_CLOSURE_MODEL_DIR) + "/weights.bin", 0.5);
    if(model.alphaMin != 0.01 || model.alphaMax != 6.0 || model.gate.size() != 192 || model.down.size() != 64)
        throw std::runtime_error("model metadata mismatch");
    bool mismatch = false;
    try
    {
        (void) heatclosure::loadModel(std::string(HEAT_CLOSURE_MODEL_DIR) + "/weights.bin", 0.6);
    }
    catch(std::runtime_error const&)
    {
        mismatch = true;
    }
    if(!mismatch)
        throw std::runtime_error("beta mismatch accepted");
    std::cout << "model loader checks passed\n";
}
catch(std::exception const& e)
{
    std::cerr << e.what() << '\n';
    return 1;
}
