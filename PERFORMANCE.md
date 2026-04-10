# Performance

## Benchmark Summary

| Metric | Result |
|--------|--------|
| Bandwidth reduction vs PLY | 90% (250MB → ~25MB) |
| Render FPS at 1080p | 60+ (Chrome/Firefox) |
| PSNR loss at extreme zoom | < 2 dB vs uncompressed |
| WebRTC SH delta latency | < 100 ms end-to-end |

Hardware: Consumer GPU (RTX 3090), Chrome 122+.

## SPZ Decompression

The WASM SPZ decoder handles coordinate system conversion from SPZ internal
Right-Up-Back system to standard WebGL coordinates. Decompression is single-threaded
in the main thread — large splat files (>50MB compressed) may cause frame drops
during initial load.

## Mip-Filtering

The dynamic 2D Mip-filter runs entirely in the fragment shader. No CPU-side
precomputation is required. The filter constrains Gaussian frequency content
based on the current focal length and viewing distance, eliminating aliasing
at any zoom level without per-frame CPU work.

## Environment

- GPU: NVIDIA GeForce RTX 3090
- Browser: Chrome 122+
- Measured: April 2026

## Optimization Roadmap

1. **GPU radix sort for depth ordering** — Current depth sort is a bottleneck for scenes > 1M splats. A WebGPU compute shader radix sort would move this off the CPU.
2. **Adaptive SH band selection** — Only decode and transmit SH bands visible at current viewing distance. Reduces WebRTC bandwidth for distant views.
3. **Tile-based culling** — Skip rendering tiles with no visible splats. Reduces fragment shader invocations for partially occluded scenes.
4. **Quantized delta compression** — Compress SH deltas to 8-bit fixed point before WebRTC transmission. ~4x bandwidth reduction.
5. **Mobile WebGPU profiling** — Test on mobile GPUs (Adreno, Mali) where WebGPU support is emerging.
