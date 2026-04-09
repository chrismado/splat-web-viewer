/**
 * Spherical Harmonic Delta Batch Decoder
 *
 * Decodes batched SH coefficient delta packets from the WebRTC stream
 * and applies them to local GaussianSplat arrays for real-time updates.
 *
 * Batch binary format:
 *   batchCount: uint16 (2 bytes) -- number of deltas in this batch
 *   For each delta:
 *     gaussianIndex: uint32 (4 bytes)
 *     numCoeffs:     uint16 (2 bytes)
 *     coefficients:  float32[] (numCoeffs * 4 bytes)
 *     timestamp:     float64 (8 bytes)
 */

import { GaussianSplat } from "../compression/spz_decoder";
import { SphericalHarmonicDelta } from "./webrtc_client";

/**
 * Decode a batched delta packet containing multiple SH updates.
 *
 * This is more efficient than individual delta messages when the server
 * batches updates per frame (e.g., 10-50 Gaussians updated per 16ms frame).
 */
export function decodeSHDeltaBatch(buffer: ArrayBuffer): SphericalHarmonicDelta[] {
  const view = new DataView(buffer);

  if (buffer.byteLength < 2) {
    throw new Error(`Batch buffer too small: ${buffer.byteLength} bytes`);
  }

  const batchCount = view.getUint16(0, true);
  const deltas: SphericalHarmonicDelta[] = new Array(batchCount);
  let offset = 2;

  for (let i = 0; i < batchCount; i++) {
    if (offset + 6 > buffer.byteLength) {
      throw new Error(`Batch truncated at delta ${i}, offset ${offset}`);
    }

    const gaussianIndex = view.getUint32(offset, true);
    offset += 4;

    const numCoeffs = view.getUint16(offset, true);
    offset += 2;

    const expectedEnd = offset + numCoeffs * 4 + 8;
    if (expectedEnd > buffer.byteLength) {
      throw new Error(
        `Batch delta ${i} truncated: need ${expectedEnd} bytes, have ${buffer.byteLength}`
      );
    }

    const coefficients = new Float32Array(numCoeffs);
    for (let j = 0; j < numCoeffs; j++) {
      coefficients[j] = view.getFloat32(offset, true);
      offset += 4;
    }

    const timestamp = view.getFloat64(offset, true);
    offset += 8;

    deltas[i] = { gaussianIndex, coefficients, timestamp };
  }

  return deltas;
}

/**
 * Apply a single SH delta to a GaussianSplat's spherical harmonics in place.
 *
 * The delta coefficients are added to the existing SH coefficients.
 * If the delta has more coefficients than the splat, the extra are stored
 * by expanding the SH array (degree upgrade).
 */
export function applySHDeltaToSplat(
  splat: GaussianSplat,
  delta: SphericalHarmonicDelta
): void {
  const numExisting = splat.sphericalHarmonics.length;
  const numDelta = delta.coefficients.length;

  if (numDelta <= numExisting) {
    // Apply delta to existing coefficients (additive)
    for (let i = 0; i < numDelta; i++) {
      splat.sphericalHarmonics[i] += delta.coefficients[i];
    }
  } else {
    // Expand the SH array to accommodate higher-degree terms
    const expanded = new Float32Array(numDelta);
    expanded.set(splat.sphericalHarmonics);
    for (let i = 0; i < numDelta; i++) {
      expanded[i] += delta.coefficients[i];
    }
    splat.sphericalHarmonics = expanded;
  }

  // Also update the DC color (first 3 SH coefficients map to base RGB)
  if (numDelta >= 3) {
    splat.color[0] = Math.max(0, Math.min(1, splat.color[0] + delta.coefficients[0]));
    splat.color[1] = Math.max(0, Math.min(1, splat.color[1] + delta.coefficients[1]));
    splat.color[2] = Math.max(0, Math.min(1, splat.color[2] + delta.coefficients[2]));
  }
}

/**
 * Apply a batch of SH deltas to a GaussianSplat array.
 * Returns the number of deltas successfully applied.
 */
export function applySHDeltaBatch(
  splats: GaussianSplat[],
  deltas: SphericalHarmonicDelta[]
): number {
  let applied = 0;
  for (const delta of deltas) {
    if (delta.gaussianIndex < splats.length) {
      applySHDeltaToSplat(splats[delta.gaussianIndex], delta);
      applied++;
    }
  }
  return applied;
}
