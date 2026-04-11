# splat-web-viewer

**Browser Gaussian Splat Review Prototype**

A TypeScript prototype for browser-native Gaussian splat review. It includes SPZ parsing with the browser `DecompressionStream` API, WebGPU/WebGL rendering paths, shader-side mip-filtering exploration, a built-in sample scene, and a WebRTC spherical-harmonic delta integration point.

This repo explores how spatial assets can become fast creative review material: inspect a scene in the browser, choose useful camera references, and carry those references into generative video or previs workflows.

Live demo target: [chrismado.github.io/splat-web-viewer](https://chrismado.github.io/splat-web-viewer/)

---

## Portfolio Context

This repo is part of [Creative AI Workflows](https://chrismado.github.io/creative-ai-workflows/) ([source](https://github.com/chrismado/creative-ai-workflows)), a portfolio showcase connecting generative video, 3D scene review, creative QA, and enterprise deployment.

In that system, `splat-web-viewer` is the visual demo anchor. It gives creative teams a browser-based way to inspect spatial assets, choose camera positions, and create stronger references before moving into generative video or animation.

### Customer-Facing Use Case

An architecture, product, VFX, or digital-human team wants to review spatial context without a heavy local 3D setup. This repo is positioned as the bridge between 3D scene understanding and AI-assisted video workflows: inspect the world, pick the visual intent, then generate or animate from a better reference.

### Demo Narrative

- Start with a spatial asset that would normally require specialist review tools.
- Open it in the browser, inspect camera positions, and save visual references.
- Connect those references to a generative video or previs workflow.

---

## Target Capability Gap

| Viewer | Framework | SPZ Support | Mip-Filtering | Streaming |
|--------|-----------|-------------|---------------|-----------|
| mkkellogg/GaussianSplats3D | Three.js/WebGL | Proprietary `.ksplat` focus | No | No |
| nianticlabs/spz | C++/TypeScript tools | Data format tools | No | No |
| autonomousvision/mip-splatting | Python/CUDA | No | Offline reference | No |
| LiXin97/gaussian-viewer | WebGL/Python backend | No | No | No |
| **splat-web-viewer** | **WebGPU/WebGL** | **Prototype parser** | **Prototype shader path** | **Client hooks for SH deltas** |

---

## Architecture

```text
Cloud or local process
  |
  | optional SH coefficient deltas
  v
Browser client
  |
  |-- SPZ decoder -> Gaussian splat data
  |-- WebGPU rasterizer -> Mip-filter shader path
  |-- WebGL fallback -> basic Gaussian quad rendering
  |
  v
Interactive scene review
```

### SPZ Decoding

SPZ format (Niantic Labs) uses a compact binary layout with gzipped arrays of positions, alphas, colors, scales, rotations, and spherical harmonics. The current prototype decodes those sections in TypeScript with browser-native gzip decompression and converts the SPZ internal Right-Up-Back coordinate frame to the viewer's WebGL-style frame.

### Mip-Filter Shader Path

Standard 3DGS dilation can create popping, dilation artifacts, and high-frequency aliasing when focal length or viewing distance changes. This repo explores a Mip-Splatting-inspired WebGPU shader path for smoothing projected covariance and reducing aliasing artifacts.

### SH Delta Streaming

The streaming layer is an integration point for creative review workflows: a WebRTC data channel can receive compressed delta-updates of spherical harmonic coefficients from a remote process. The browser applies those deltas to local Gaussian colors before rasterizing. End-to-end server integration and latency measurement remain future work.

---

## Measurement Plan

| Metric | Current Status |
|--------|----------------|
| SPZ compression ratio vs PLY | Parser exists; needs corpus measurement |
| Render FPS at 1080p | WebGPU/WebGL paths exist; needs repeatable browser benchmark |
| Mip-filter visual quality | Shader path implemented; needs side-by-side aliasing captures |
| WebRTC SH delta latency | Client hooks exist; needs server harness measurement |

The next credibility step is recording repeatable measurements on a known public splat scene and checking them into `PERFORMANCE.md`.

---

## Stack

- **Rendering:** WebGPU primary path, WebGL 2 fallback
- **Compression:** SPZ-style parsing with browser-native gzip decompression
- **Anti-aliasing:** Mip-Splatting-inspired shader path
- **Streaming:** WebRTC data channel client hooks
- **Format export:** glTF export prototype
- **Framework:** Vanilla TypeScript

---

## Directory Structure

```text
splat-web-viewer/
├── benchmarks/
├── demo/
├── shaders/
├── src/
│   ├── compression/
│   ├── export/
│   ├── renderer/
│   └── streaming/
├── tests/
├── wasm/
├── package.json
└── README.md
```

---

## Quick Start

```bash
git clone https://github.com/chrismado/splat-web-viewer
cd splat-web-viewer
npm install
npm run build

# Open http://localhost:8080/demo/
npm run demo

# Optional streaming harness
python src/streaming/webrtc_server.py --model your_model
npm run stream-demo
```

The demo now loads a small synthetic scene by default, so it should show visible splats even before a `.spz` upload.

---

## References

1. **Mip-Splatting: Alias-Free 3D Gaussian Splatting** - Zehao Yu et al., CVPR 2024.
2. **SPZ: A Compact Gaussian Splat Format** - Niantic Labs.
3. **3D Gaussian Splatting for Real-Time Radiance Field Rendering** - Kerbl et al., SIGGRAPH 2023.
4. **mkkellogg/GaussianSplats3D** - Existing Three.js viewer reference.
5. **autonomousvision/mip-splatting** - Desktop CUDA Mip-Splatting implementation reference.
