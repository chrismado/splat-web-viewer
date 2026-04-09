# WASM SPZ Decoder

Future home of a WebAssembly-based SPZ decoder for improved decode performance.

## Planned Architecture

The WASM decoder will replace the pure TypeScript `decodeSPZ()` for large scenes
(>500k Gaussians) where JavaScript decode time becomes a bottleneck.

### Build Requirements

- Emscripten SDK (emsdk) 3.1+
- CMake 3.20+

### Target API

The WASM module will expose the same interface as the TypeScript decoder:

```
decodeSPZ(buffer: ArrayBuffer) -> GaussianSplat[]
```

Data will be passed via shared `ArrayBuffer` to avoid copy overhead.
