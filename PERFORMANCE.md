# Performance

## Measurement Plan

This repo is currently a rendering and workflow prototype. The metrics below are the measurements to collect before presenting this as production-grade viewer infrastructure.

| Metric | Current Status | Next Measurement |
|--------|----------------|------------------|
| SPZ compression ratio vs PLY | Parser exists | Compare a public PLY scene against equivalent SPZ bytes |
| Render FPS at 1080p | WebGPU/WebGL paths exist | Run `npm run benchmark` on a fixed sample scene |
| Mip-filter visual quality | Shader path exists | Capture side-by-side zoom images against baseline splatting |
| WebRTC SH delta latency | Client hooks exist | Add local server harness with timestamped delta packets |

## SPZ Decoding

The TypeScript SPZ decoder handles coordinate system conversion from SPZ internal Right-Up-Back coordinates to standard WebGL-style coordinates. Decompression is single-threaded in the main thread, so large splat files may cause frame drops during initial load.

## Mip-Filtering

The 2D Mip-filter exploration runs in the WebGPU fragment shader. The next step is to capture visual comparisons against a baseline renderer at multiple focal lengths and scene distances.

## Target Environment

- Browser: Chrome or Edge with WebGPU enabled
- Fallback: WebGL 2
- Measured: TBD with checked-in sample assets

## Optimization Roadmap

1. **GPU radix sort for depth ordering** - Current rendering does not include production-grade depth sorting. A WebGPU compute shader radix sort would move this off the CPU.
2. **Adaptive SH band selection** - Only decode and transmit SH bands visible at the current viewing distance.
3. **Tile-based culling** - Skip rendering tiles with no visible splats to reduce fragment shader invocations.
4. **Quantized delta compression** - Compress SH deltas to 8-bit fixed point before WebRTC transmission.
5. **Mobile WebGPU profiling** - Test on mobile GPUs where WebGPU support is emerging.
