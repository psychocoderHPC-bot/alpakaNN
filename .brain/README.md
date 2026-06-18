# Knowledge Graph

- `include/alpaka/nn/nn.hpp`
  - Canonical umbrella header for the library surface.
- `include/alpaka/nn/core`
  - Generic host/acc-safe layout and shape utilities in `alpaka::nn`.
- `include/alpaka/nn/nn`
  - Generic enums shared by host and acc code such as rope and attention layouts.
- `include/alpaka/nn/onHost`
  - Host-callable APIs in `alpaka::nn::onHost`.
- `include/alpaka/nn/onAcc/internal`
  - Acc-only kernel building blocks used by host wrappers.
  - Attention kernels are sensitive to launch/indexing shape; the current fix keeps them on native 4D launch extents.
- `CMakeLists.txt`
  - Package name stays `alpakaNN`; exported target is `alpaka::nn`.
- `test/unit`
  - Validates the public `alpaka::nn` / `alpaka::nn::onHost` surface.
