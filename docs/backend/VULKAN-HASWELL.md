# Vulkan compatibility mode for Intel Haswell iGPUs (HD Graphics 4400)

This fork adds an experimental, opt-in Vulkan path for Intel Haswell GT2 integrated GPUs (tested on HD Graphics 4400, PCI ID `8086:0a16`, Mesa `hasvk` driver). It is not part of upstream llama.cpp.

## Status

| Item | State |
|---|---|
| Correctness | Verified against the CPU backend (see [Validation](#validation)) |
| Performance | **Slower than the CPU backend** on the test system (see [Performance](#performance)) |
| Scope | Q8_0 x F32 `MUL_MAT` only; all other operations run on the CPU |
| Devices | Intel `0x0A16` only (other Haswell IDs are untested and not enabled) |
| Maturity | Experimental, intended for research and for running Q8_0 models where the Vulkan device must be usable |

## Background

The upstream Vulkan backend requires `storageBuffer16BitAccess`, because GGML quantization blocks contain fp16 values that the shaders read directly from storage buffers. Haswell iGPUs expose neither 16-bit nor 8-bit storage access, nor `shaderFloat16` / `shaderInt16`, so upstream rejects the device with:

```
ggml_vulkan: device Vulkan0 does not support 16-bit storage.
```

Simply skipping that check (as proposed in [ggml-org/llama.cpp#27723](https://github.com/ggml-org/llama.cpp/pull/27723)) dispatches shaders that use a feature the device does not provide. This fork takes a different approach: it does not report or enable any capability the device lacks. Instead it provides a dedicated kernel that only uses 32-bit storage access, and restricts the device to that kernel.

Related upstream discussion: [#7620](https://github.com/ggml-org/llama.cpp/issues/7620), [#8219](https://github.com/ggml-org/llama.cpp/issues/8219), [#14805](https://github.com/ggml-org/llama.cpp/issues/14805).

## How it works

Compatibility mode is enabled with `GGML_VK_ENABLE_HASWELL_COMPAT=1` and only applies to an Intel device with ID `0x0A16` that lacks 16-bit storage. When active:

- **Dedicated kernel** (`vulkan-shaders/mul_mat_vec_q8_0_compat.comp`). Q8_0 weights (34-byte blocks: one fp16 scale and 32 int8 values) are read as 32-bit words and decoded with integer operations. Both block alignments (4-byte and 2-byte) are handled. The fp16 scale is decoded with `unpackHalf2x16` (core GLSL 4.50), which requires no fp16 device feature. Accumulation is in fp32. Each workgroup computes 4 output rows for up to 4 input columns; arbitrary strides, batch dimensions and broadcasting are supported.
- **Restricted operation support.** `supports_op` reports only Q8_0 x F32 `MUL_MAT` (plus no-op view operations). No other shader, and therefore no shader requiring 16-bit storage, can be dispatched. `offload_op` is disabled so CPU-resident weights are never copied to the device per batch.
- **Host-visible buffers.** On this UMA device the default buffer type becomes the pinned host buffer type. Weights, KV cache, recurrent state and compute buffers are accessible to both the CPU and the GPU, so CPU operations update caches in place and the GPU reads weights without copies.
- **Bounded submissions.** Work per command-buffer submission is capped to stay well below the i915 hang-check timeout.

A one-time log line confirms the kernel is in use:

```
ggml_vulkan: Vulkan0: using compat Q8_0 x F32 MUL_MAT kernel
```

## Building

The standard Vulkan build is used; no extra CMake options are required.

```bash
cmake -B build -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j 2
```

On machines with limited memory, keep build parallelism low.

## Usage

```bash
export GGML_VK_ENABLE_HASWELL_COMPAT=1
./build/bin/llama-cli -m model-q8_0.gguf -ngl 99
```

Notes:

- Only Q8_0 models benefit; with other weight types every operation falls back to the CPU.
- Because all weights live in host-visible memory, every Q8_0 matrix multiplication is executed on the GPU regardless of `-ngl`. For CPU-only inference, unset the variable or pass `--device none`.
- `llama-server`'s automatic memory fitting does not account for host-visible device buffers and must be disabled: pass `-fit off` or set `LLAMA_ARG_FIT=off`.

### Environment variables

| Variable | Purpose |
|---|---|
| `GGML_VK_ENABLE_HASWELL_COMPAT=1` | Enable compatibility mode on Intel `0x0A16` |
| `GGML_VK_HASWELL_COMPAT_FORCE_KERNEL=1` | Testing only: use the compat kernel on any Vulkan device (for example Mesa llvmpipe). Does not change reported device features |

## Validation

Test system: Intel Core i3-4005U (2C/4T), HD Graphics 4400, 7.2 GiB RAM, Ubuntu 26.04, Mesa `hasvk`.

| Test | Result |
|---|---|
| `test-backend-ops test -b Vulkan0 -o MUL_MAT -p 'type_a=q8_0,type_b=f32'` on HD Graphics 4400 | 113/113 passed |
| Same suite on Mesa llvmpipe with `GGML_VK_HASWELL_COMPAT_FORCE_KERNEL=1` | 89/89 passed |
| `test-backend-ops support` in compatibility mode | Only Q8_0 x F32 `MUL_MAT` reported as supported |
| Perplexity, Qwen3.5 2B Q8_0, `docs/build.md`, c=256, 6 chunks | CPU 8.0972, GPU 8.0731 (+/- 0.82) |
| Repeated greedy generation | Deterministic across runs; no GPU hangs reported by the kernel |

The small difference from the CPU result is expected: the CPU backend quantizes activations to Q8_0, while this kernel keeps them in fp32.

## Performance

Qwen3.5 2B Q8_0, 2 threads, token generation:

| Configuration | Prompt (t/s) | Generation (t/s) |
|---|---|---|
| CPU backend | 13.9 - 14.6 | 4.8 - 5.0 |
| Compatibility mode (all Q8_0 matmuls on the GPU) | 9.6 - 11.2 | 3.0 |

Kernel throughput on a 4096 x 14336 matrix-vector product is about 11 ms (roughly 5.5 GB/s of weight data). Token generation on this hardware is memory-bandwidth bound, the iGPU shares system memory with the CPU, and every non-matmul operation runs on the CPU, so this mode does not outperform the CPU backend on the test system. Its value is correctness and the ability to use the Vulkan device where the stock backend refuses to load.

## Using with Ollama

Recent Ollama releases run llama.cpp's `llama-server` and load the Vulkan backend from `<ollama lib dir>/vulkan/libggml-vulkan.so`. With `OLLAMA_VULKAN=1`, the stock library rejects Haswell GPUs and model loading fails. A library built from this fork can replace it, provided the ggml ABI matches the llama.cpp version bundled with Ollama (check with `<ollama lib dir>/llama-server --version`).

Build the backend as a dynamically loadable module:

```bash
cmake -B build-dl -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON \
      -DGGML_BACKEND_DL=ON -DGGML_VULKAN=ON -DGGML_NATIVE=OFF -DGGML_CPU_ALL_VARIANTS=ON
cmake --build build-dl --target ggml-vulkan -j 2
```

Install it (the paths are those of the standard Linux installer; keep a backup of the original):

```bash
sudo cp -a /usr/local/lib/ollama/vulkan/libggml-vulkan.so /usr/local/lib/ollama/vulkan/libggml-vulkan.so.orig
sudo install -m 755 build-dl/bin/libggml-vulkan.so /usr/local/lib/ollama/vulkan/libggml-vulkan.so

sudo mkdir -p /etc/systemd/system/ollama.service.d
sudo tee /etc/systemd/system/ollama.service.d/haswell-compat.conf > /dev/null << 'EOF'
[Service]
Environment="OLLAMA_VULKAN=1"
Environment="GGML_VK_ENABLE_HASWELL_COMPAT=1"
Environment="LLAMA_ARG_FIT=off"
EOF
sudo systemctl daemon-reload && sudo systemctl restart ollama
```

To revert, restore `libggml-vulkan.so.orig`, remove `haswell-compat.conf` and restart the service. Ollama updates overwrite the library, so repeat the installation after upgrading.

## Limitations

- Only Q8_0 x F32 matrix multiplication is accelerated.
- Only PCI device ID `0x0A16` is enabled.
- If a Q8_0 tensor with an odd number of blocks per row ends exactly at the end of its allocation, the final two quantized values of the last block read as zero; a warning is logged when this occurs.
- `llama-server` memory fitting must be disabled in this mode.
- Not intended for upstream submission in its current form: upstream requires 16-bit storage, and the performance on this hardware does not justify the additional maintenance.

## Changed files

- `ggml/src/ggml-vulkan/vulkan-shaders/mul_mat_vec_q8_0_compat.comp` (new)
- `ggml/src/ggml-vulkan/vulkan-shaders/vulkan-shaders-gen.cpp`
- `ggml/src/ggml-vulkan/ggml-vulkan.cpp`
- `ggml/src/ggml-vulkan/ggml-vulkan-types.h`
- `ggml/src/ggml-vulkan/ggml-vulkan-push-constants.h`
- `docs/backend/VULKAN-HASWELL.md`
