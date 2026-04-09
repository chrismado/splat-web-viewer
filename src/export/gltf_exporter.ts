/**
 * glTF 2.0 Exporter for Gaussian Splats
 *
 * Exports GaussianSplat[] to a glTF 2.0 JSON file with embedded binary data.
 * Uses the EXT_gaussian_splatting extension proposal for storing
 * splat-specific attributes (covariance, opacity, spherical harmonics).
 *
 * Output format: .gltf JSON with base64-encoded buffer
 */

import { GaussianSplat } from "../compression/spz_decoder";

interface GLTFDocument {
  asset: { version: string; generator: string };
  extensionsUsed: string[];
  buffers: Array<{ uri: string; byteLength: number }>;
  bufferViews: Array<{
    buffer: number;
    byteOffset: number;
    byteLength: number;
    target?: number;
  }>;
  accessors: Array<{
    bufferView: number;
    componentType: number;
    count: number;
    type: string;
    max?: number[];
    min?: number[];
  }>;
  meshes: Array<{
    primitives: Array<{
      attributes: Record<string, number>;
      mode: number;
      extensions?: Record<string, unknown>;
    }>;
  }>;
  nodes: Array<{ mesh: number; name: string }>;
  scenes: Array<{ nodes: number[] }>;
  scene: number;
}

// glTF component type constants
const GL_FLOAT = 5126;
const GL_UNSIGNED_BYTE = 5121;

/**
 * Export an array of Gaussian splats to a glTF 2.0 JSON string.
 *
 * The exported file uses standard POSITION and COLOR_0 attributes
 * plus custom attributes for splat-specific data under the
 * EXT_gaussian_splatting extension.
 */
export function exportToGLTF(splats: GaussianSplat[]): string {
  const numSplats = splats.length;
  if (numSplats === 0) {
    throw new Error("Cannot export empty splat array");
  }

  // Calculate buffer sizes
  const positionBytes = numSplats * 3 * 4;   // vec3<f32>
  const colorBytes = numSplats * 3 * 4;      // vec3<f32>
  const opacityBytes = numSplats * 4;         // f32
  const scaleBytes = numSplats * 3 * 4;       // vec3<f32>
  const rotationBytes = numSplats * 4 * 4;    // vec4<f32>
  const shDegree = estimateSHDegree(splats[0].sphericalHarmonics.length);
  const shCoeffsPerSplat = splats[0].sphericalHarmonics.length;
  const shBytes = numSplats * shCoeffsPerSplat * 4;

  const totalBytes = positionBytes + colorBytes + opacityBytes +
                     scaleBytes + rotationBytes + shBytes;

  // Build interleaved buffer
  const buffer = new ArrayBuffer(totalBytes);
  const f32 = new Float32Array(buffer);

  let posMin = [Infinity, Infinity, Infinity];
  let posMax = [-Infinity, -Infinity, -Infinity];

  let floatOffset = 0;

  // Positions
  const posStart = floatOffset;
  for (let i = 0; i < numSplats; i++) {
    const p = splats[i].position;
    f32[floatOffset++] = p[0];
    f32[floatOffset++] = p[1];
    f32[floatOffset++] = p[2];
    posMin[0] = Math.min(posMin[0], p[0]);
    posMin[1] = Math.min(posMin[1], p[1]);
    posMin[2] = Math.min(posMin[2], p[2]);
    posMax[0] = Math.max(posMax[0], p[0]);
    posMax[1] = Math.max(posMax[1], p[1]);
    posMax[2] = Math.max(posMax[2], p[2]);
  }

  // Colors
  const colorStart = floatOffset;
  for (let i = 0; i < numSplats; i++) {
    const c = splats[i].color;
    f32[floatOffset++] = c[0];
    f32[floatOffset++] = c[1];
    f32[floatOffset++] = c[2];
  }

  // Opacities
  const opacityStart = floatOffset;
  for (let i = 0; i < numSplats; i++) {
    f32[floatOffset++] = splats[i].alpha;
  }

  // Scales
  const scaleStart = floatOffset;
  for (let i = 0; i < numSplats; i++) {
    const s = splats[i].scale;
    f32[floatOffset++] = s[0];
    f32[floatOffset++] = s[1];
    f32[floatOffset++] = s[2];
  }

  // Rotations
  const rotStart = floatOffset;
  for (let i = 0; i < numSplats; i++) {
    const r = splats[i].rotation;
    f32[floatOffset++] = r[0];
    f32[floatOffset++] = r[1];
    f32[floatOffset++] = r[2];
    f32[floatOffset++] = r[3];
  }

  // Spherical harmonics
  const shStart = floatOffset;
  for (let i = 0; i < numSplats; i++) {
    const sh = splats[i].sphericalHarmonics;
    for (let j = 0; j < shCoeffsPerSplat; j++) {
      f32[floatOffset++] = sh[j];
    }
  }

  // Encode buffer as base64 data URI
  const base64 = arrayBufferToBase64(buffer);
  const dataUri = `data:application/octet-stream;base64,${base64}`;

  // Build glTF document
  const gltf: GLTFDocument = {
    asset: {
      version: "2.0",
      generator: "splat-web-viewer",
    },
    extensionsUsed: ["EXT_gaussian_splatting"],
    buffers: [{ uri: dataUri, byteLength: totalBytes }],
    bufferViews: [
      { buffer: 0, byteOffset: posStart * 4, byteLength: positionBytes },
      { buffer: 0, byteOffset: colorStart * 4, byteLength: colorBytes },
      { buffer: 0, byteOffset: opacityStart * 4, byteLength: opacityBytes },
      { buffer: 0, byteOffset: scaleStart * 4, byteLength: scaleBytes },
      { buffer: 0, byteOffset: rotStart * 4, byteLength: rotationBytes },
      { buffer: 0, byteOffset: shStart * 4, byteLength: shBytes },
    ],
    accessors: [
      { bufferView: 0, componentType: GL_FLOAT, count: numSplats, type: "VEC3", min: posMin, max: posMax },
      { bufferView: 1, componentType: GL_FLOAT, count: numSplats, type: "VEC3" },
      { bufferView: 2, componentType: GL_FLOAT, count: numSplats, type: "SCALAR" },
      { bufferView: 3, componentType: GL_FLOAT, count: numSplats, type: "VEC3" },
      { bufferView: 4, componentType: GL_FLOAT, count: numSplats, type: "VEC4" },
      { bufferView: 5, componentType: GL_FLOAT, count: numSplats, type: "SCALAR" },
    ],
    meshes: [{
      primitives: [{
        attributes: {
          POSITION: 0,
          COLOR_0: 1,
        },
        mode: 0, // POINTS
        extensions: {
          EXT_gaussian_splatting: {
            opacity: 2,
            scale: 3,
            rotation: 4,
            sphericalHarmonics: 5,
            shDegree,
          },
        },
      }],
    }],
    nodes: [{ mesh: 0, name: "GaussianSplats" }],
    scenes: [{ nodes: [0] }],
    scene: 0,
  };

  return JSON.stringify(gltf, null, 2);
}

/**
 * Estimate the SH degree from the number of coefficients.
 */
function estimateSHDegree(numCoeffs: number): number {
  if (numCoeffs === 0) return 0;
  if (numCoeffs <= 9) return 1;
  if (numCoeffs <= 24) return 2;
  return 3;
}

/**
 * Convert an ArrayBuffer to a base64 string.
 */
function arrayBufferToBase64(buffer: ArrayBuffer): string {
  const bytes = new Uint8Array(buffer);
  let binary = "";
  for (let i = 0; i < bytes.byteLength; i++) {
    binary += String.fromCharCode(bytes[i]);
  }
  return btoa(binary);
}
