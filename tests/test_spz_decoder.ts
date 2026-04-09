/**
 * Tests for SPZ Decoder
 *
 * Validates header parsing, decompression, coordinate conversion,
 * and edge cases for the SPZ binary format decoder.
 */

import { decodeSPZ, convertCoordinates, GaussianSplat } from "../src/compression/spz_decoder";

// Simple test runner
let passed = 0;
let failed = 0;

function assert(condition: boolean, message: string): void {
  if (condition) {
    passed++;
    console.log(`  PASS: ${message}`);
  } else {
    failed++;
    console.error(`  FAIL: ${message}`);
  }
}

function assertApprox(actual: number, expected: number, epsilon: number, message: string): void {
  assert(Math.abs(actual - expected) < epsilon, `${message} (got ${actual}, expected ${expected})`);
}

// --- Test: convertCoordinates ---

console.log("convertCoordinates:");

(function testConvertNegatesZ() {
  const splat: GaussianSplat = {
    position: new Float32Array([1, 2, 3]),
    alpha: 0.5,
    color: new Float32Array([0.1, 0.2, 0.3]),
    scale: new Float32Array([0.4, 0.5, 0.6]),
    rotation: new Float32Array([1, 0, 0, 0]), // identity quaternion
    sphericalHarmonics: new Float32Array(0),
  };

  const converted = convertCoordinates(splat);

  assertApprox(converted.position[0], 1, 1e-6, "X position unchanged");
  assertApprox(converted.position[1], 2, 1e-6, "Y position unchanged");
  assertApprox(converted.position[2], -3, 1e-6, "Z position negated");
  assertApprox(converted.scale[2], -0.6, 1e-6, "Z scale negated");
  assertApprox(converted.alpha, 0.5, 1e-6, "Alpha preserved");
})();

(function testConvertPreservesColor() {
  const splat: GaussianSplat = {
    position: new Float32Array([0, 0, 0]),
    alpha: 1.0,
    color: new Float32Array([0.9, 0.8, 0.7]),
    scale: new Float32Array([1, 1, 1]),
    rotation: new Float32Array([1, 0, 0, 0]),
    sphericalHarmonics: new Float32Array([0.1, 0.2, 0.3]),
  };

  const converted = convertCoordinates(splat);

  assertApprox(converted.color[0], 0.9, 1e-6, "Red preserved");
  assertApprox(converted.color[1], 0.8, 1e-6, "Green preserved");
  assertApprox(converted.color[2], 0.7, 1e-6, "Blue preserved");
  assert(converted.sphericalHarmonics.length === 3, "SH length preserved");
})();

(function testConvertDoesNotMutateOriginal() {
  const splat: GaussianSplat = {
    position: new Float32Array([1, 2, 3]),
    alpha: 0.5,
    color: new Float32Array([0.1, 0.2, 0.3]),
    scale: new Float32Array([0.4, 0.5, 0.6]),
    rotation: new Float32Array([1, 0, 0, 0]),
    sphericalHarmonics: new Float32Array(0),
  };

  convertCoordinates(splat);

  assertApprox(splat.position[2], 3, 1e-6, "Original Z position unchanged");
  assertApprox(splat.scale[2], 0.6, 1e-6, "Original Z scale unchanged");
})();

// --- Test: decodeSPZ error handling ---

console.log("\ndecodeSPZ error handling:");

(async function testEmptyBuffer() {
  try {
    await decodeSPZ(new ArrayBuffer(0));
    assert(false, "Should throw on empty buffer");
  } catch (e) {
    assert(true, "Throws on empty buffer");
  }
})();

(async function testTooSmallBuffer() {
  try {
    await decodeSPZ(new ArrayBuffer(8));
    assert(false, "Should throw on buffer smaller than header");
  } catch (e) {
    assert(true, "Throws on buffer smaller than header");
  }
})();

(async function testBadMagic() {
  const buf = new ArrayBuffer(16);
  const view = new DataView(buf);
  view.setUint32(0, 0xDEADBEEF, true); // wrong magic
  try {
    await decodeSPZ(buf);
    assert(false, "Should throw on bad magic");
  } catch (e) {
    assert(true, "Throws on bad magic number");
  }
})();

(async function testZeroPoints() {
  const buf = new ArrayBuffer(16);
  const view = new DataView(buf);
  view.setUint32(0, 0x5053_5A00, true); // SPZ magic
  view.setUint32(4, 1, true);           // version
  view.setUint32(8, 0, true);           // numPoints = 0
  view.setUint8(12, 0);                 // shDegree
  view.setUint8(13, 0);                 // flags

  const result = await decodeSPZ(buf);
  assert(result.length === 0, "Returns empty array for zero points");
})();

// --- Summary ---
setTimeout(() => {
  console.log(`\nResults: ${passed} passed, ${failed} failed`);
  if (failed > 0) process.exit(1);
}, 500);
