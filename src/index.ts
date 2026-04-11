/**
 * splat-web-viewer
 *
 * Anti-aliased WebGPU Gaussian splat viewer with SPZ decoding,
 * WebRTC streaming, and WebGL fallback.
 */

// Compression / decoding
export { decodeSPZ, convertCoordinates } from "./compression/spz_decoder";
export type { GaussianSplat } from "./compression/spz_decoder";
export {
  convertSPZToWebGL,
  convertWebGLToSPZ,
  convertGLTFToWebGL,
  CoordinateSystem,
} from "./compression/coordinate_converter";

// Streaming
export { WebRTCSplatClient } from "./streaming/webrtc_client";
export type { SphericalHarmonicDelta } from "./streaming/webrtc_client";
export { decodeSHDeltaBatch, applySHDeltaToSplat } from "./streaming/sh_delta_decoder";

// Renderers
export { WebGPURasterizer } from "./renderer/webgpu_renderer";
export { WebGLFallbackRenderer } from "./renderer/webgl_fallback";
export { cameraSortSignature, distanceSquared, sortSplatsBackToFront } from "./renderer/splat_sort";

// Export
export { exportToGLTF } from "./export/gltf_exporter";
