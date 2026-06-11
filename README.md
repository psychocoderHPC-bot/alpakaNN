**alpaka** - GEMM example
=========================

This is a GEMM implementation with alpaka.
It shows two basic implementations:
- `gemm` is using global memory and no thread blocking
- `sgemm` is using shared memory for caching a tiles

Keep in mind, that both methods are not optimized for performance and will be much slower than optimized BLAS libraries.

Inference support also includes a tiny LLaMA-style decoder path built from alpaka3 kernels. The tiny end-to-end test downloads and converts `hf-internal-testing/tiny-random-LlamaForCausalLM` into the project’s compact binary format with:

`python3 tools/download_tiny_llama.py /workspace/testdata/tiny_llama`

For your own models later, keep the internal tensor order aligned with the converter:

- embeddings: `[vocab, hidden]`
- Q and O attention weights: `[in, out]`
- grouped K and V attention weights: `[in, num_key_value_heads * head_dim]`
- MLP weights: `[in, out]`
- norm weights: `[hidden]`
- lm head: `[hidden, vocab]`

The Llama-family runtime now supports grouped-query attention when `num_attention_heads % num_key_value_heads == 0`. The documented TinyLlama chat flow is:

`python3 tools/download_tiny_llama.py tinyllama-1.1b-chat models/tinyllama-1.1b-chat`

`./build/example/cli/ChatCli -i --model models/tinyllama-1.1b-chat`
