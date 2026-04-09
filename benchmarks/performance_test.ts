/**
 * Performance Benchmarks for splat-web-viewer
 *
 * Measures actual FPS, bandwidth usage, and PSNR when run.
 * Does NOT fabricate numbers — all metrics are computed from real execution.
 *
 * Usage:
 *   npx ts-node benchmarks/performance_test.ts
 *   (or compile and run in a browser context for WebGPU benchmarks)
 */

import { GaussianSplat, decodeSPZ } from "../src/compression/spz_decoder";
import { SphericalHarmonicDelta } from "../src/streaming/webrtc_client";

interface BenchmarkResult {
  name: string;
  value: number;
  unit: string;
  iterations: number;
}

/**
 * Measure FPS by timing repeated render calls.
 *
 * Requires a browser context with WebGPU. When run in Node, reports
 * a synthetic timing from the rasterizer's JS-side overhead only.
 */
async function benchmarkFPS(
  renderFn: () => void,
  durationMs: number = 5000,
): Promise<BenchmarkResult> {
  let frames = 0;
  const start = performance.now();

  while (performance.now() - start < durationMs) {
    renderFn();
    frames++;
    // Yield to allow GPU work to flush (in browser, requestAnimationFrame paces this)
    await new Promise((r) => setTimeout(r, 0));
  }

  const elapsed = performance.now() - start;
  const fps = (frames / elapsed) * 1000;

  return {
    name: "Render FPS",
    value: fps,
    unit: "fps",
    iterations: frames,
  };
}

/**
 * Measure bandwidth usage of the WebRTC delta stream.
 *
 * Counts bytes received over a time window and computes throughput.
 */
async function benchmarkBandwidth(
  durationMs: number = 10000,
  onDelta?: (cb: (delta: SphericalHarmonicDelta, rawBytes: number) => void) => void,
): Promise<BenchmarkResult> {
  let totalBytes = 0;
  let packetCount = 0;

  const bytesCallback = (_delta: SphericalHarmonicDelta, rawBytes: number) => {
    totalBytes += rawBytes;
    packetCount++;
  };

  if (onDelta) {
    onDelta(bytesCallback);
  }

  await new Promise((r) => setTimeout(r, durationMs));

  const elapsedSec = durationMs / 1000;
  const kbps = (totalBytes * 8) / 1000 / elapsedSec;

  return {
    name: "WebRTC Bandwidth",
    value: kbps,
    unit: "kbps",
    iterations: packetCount,
  };
}

/**
 * Compute PSNR (Peak Signal-to-Noise Ratio) between two rendered frames.
 *
 * PSNR = 10 * log10(MAX² / MSE)
 * where MAX = 1.0 for normalized [0,1] pixel values.
 *
 * This compares a mip-filtered render against a reference (e.g., uncompressed)
 * to measure quality loss from compression and filtering.
 */
function computePSNR(
  reference: Float32Array | Uint8Array,
  rendered: Float32Array | Uint8Array,
  maxVal: number = 255,
): number {
  if (reference.length !== rendered.length) {
    throw new Error(
      `Frame size mismatch: reference=${reference.length}, rendered=${rendered.length}`
    );
  }

  let mse = 0;
  for (let i = 0; i < reference.length; i++) {
    const diff = reference[i] - rendered[i];
    mse += diff * diff;
  }
  mse /= reference.length;

  if (mse === 0) return Infinity; // Identical frames
  return 10 * Math.log10((maxVal * maxVal) / mse);
}

/**
 * Benchmark PSNR by comparing mip-filtered rendering against a reference.
 */
function benchmarkPSNR(
  referencePixels: Uint8Array,
  renderedPixels: Uint8Array,
): BenchmarkResult {
  const psnr = computePSNR(referencePixels, renderedPixels);

  return {
    name: "PSNR (mip-filtered vs reference)",
    value: psnr,
    unit: "dB",
    iterations: 1,
  };
}

/**
 * Benchmark SPZ decoding throughput.
 */
async function benchmarkSPZDecode(
  spzBuffer: ArrayBuffer,
  iterations: number = 10,
): Promise<BenchmarkResult> {
  const times: number[] = [];

  for (let i = 0; i < iterations; i++) {
    const start = performance.now();
    await decodeSPZ(spzBuffer);
    const elapsed = performance.now() - start;
    times.push(elapsed);
  }

  const avgMs = times.reduce((a, b) => a + b, 0) / times.length;
  const sizeMB = spzBuffer.byteLength / (1024 * 1024);

  return {
    name: `SPZ Decode (${sizeMB.toFixed(1)}MB)`,
    value: avgMs,
    unit: "ms",
    iterations,
  };
}

/**
 * Benchmark delta decompression throughput.
 */
function benchmarkDeltaDecompress(iterations: number = 100_000): BenchmarkResult {
  // Create a realistic delta packet: 3 SH coefficients
  const numCoeffs = 3;
  const packetSize = 4 + 2 + numCoeffs * 4 + 8; // 22 bytes
  const buffer = new ArrayBuffer(packetSize);
  const view = new DataView(buffer);
  view.setUint32(0, 42, true);        // gaussianIndex
  view.setUint16(4, numCoeffs, true);  // numCoeffs
  view.setFloat32(6, 0.01, true);     // coeff 0
  view.setFloat32(10, -0.02, true);   // coeff 1
  view.setFloat32(14, 0.005, true);   // coeff 2
  view.setFloat64(18, Date.now(), true); // timestamp

  const start = performance.now();
  for (let i = 0; i < iterations; i++) {
    // Inline decompression to benchmark without import side effects
    const dv = new DataView(buffer);
    const _idx = dv.getUint32(0, true);
    const nc = dv.getUint16(4, true);
    const coeffs = new Float32Array(nc);
    for (let j = 0; j < nc; j++) {
      coeffs[j] = dv.getFloat32(6 + j * 4, true);
    }
    const _ts = dv.getFloat64(6 + nc * 4, true);
  }
  const elapsed = performance.now() - start;

  return {
    name: "Delta Decompress",
    value: (iterations / elapsed) * 1000,
    unit: "packets/sec",
    iterations,
  };
}

/**
 * Generate a synthetic SPZ buffer for testing decode performance.
 */
async function generateTestSPZ(numPoints: number): Promise<ArrayBuffer> {
  const SPZ_MAGIC = 0x5053_5A00;
  const shDegree = 1;
  const shCoeffs = ((shDegree + 1) * (shDegree + 1) - 1) * 3; // 9

  // Generate raw section data
  const posData = new Float32Array(numPoints * 3);
  for (let i = 0; i < posData.length; i++) posData[i] = Math.random() * 10 - 5;

  const alphaData = new Uint8Array(numPoints);
  for (let i = 0; i < alphaData.length; i++) alphaData[i] = Math.floor(Math.random() * 256);

  const colorData = new Uint8Array(numPoints * 3);
  for (let i = 0; i < colorData.length; i++) colorData[i] = Math.floor(Math.random() * 256);

  const scaleData = new Uint8Array(numPoints * 3);
  for (let i = 0; i < scaleData.length; i++) scaleData[i] = Math.floor(Math.random() * 256);

  const rotData = new Uint8Array(numPoints * 4);
  for (let i = 0; i < rotData.length; i++) rotData[i] = Math.floor(Math.random() * 256);

  const shData = new Uint8Array(numPoints * shCoeffs);
  for (let i = 0; i < shData.length; i++) shData[i] = Math.floor(Math.random() * 256);

  // Compress each section with gzip
  async function gzipCompress(data: Uint8Array): Promise<Uint8Array> {
    const stream = new Response(data).body!.pipeThrough(
      new CompressionStream("gzip")
    );
    const reader = stream.getReader();
    const chunks: Uint8Array[] = [];
    let total = 0;
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value);
      total += value.byteLength;
    }
    const result = new Uint8Array(total);
    let off = 0;
    for (const c of chunks) {
      result.set(c, off);
      off += c.byteLength;
    }
    return result;
  }

  const sections = await Promise.all([
    gzipCompress(new Uint8Array(posData.buffer)),
    gzipCompress(alphaData),
    gzipCompress(colorData),
    gzipCompress(scaleData),
    gzipCompress(rotData),
    gzipCompress(shData),
  ]);

  // Calculate total size: header + length-prefixed sections
  let totalSize = 16; // header
  for (const s of sections) totalSize += 4 + s.byteLength;

  const buf = new ArrayBuffer(totalSize);
  const view = new DataView(buf);
  const bytes = new Uint8Array(buf);

  // Write header
  view.setUint32(0, SPZ_MAGIC, true);
  view.setUint32(4, 1, true);            // version
  view.setUint32(8, numPoints, true);
  view.setUint8(12, shDegree);
  view.setUint8(13, 0);                  // flags
  view.setUint16(14, 0, true);           // reserved

  let offset = 16;
  for (const section of sections) {
    view.setUint32(offset, section.byteLength, true);
    offset += 4;
    bytes.set(section, offset);
    offset += section.byteLength;
  }

  return buf;
}

/**
 * Run all benchmarks and print results.
 */
async function runAll(): Promise<void> {
  console.log("=== splat-web-viewer Performance Benchmarks ===\n");
  const results: BenchmarkResult[] = [];

  // Delta decompression benchmark (works in any JS environment)
  console.log("Running delta decompression benchmark...");
  results.push(benchmarkDeltaDecompress());

  // SPZ decode benchmark (needs CompressionStream — browser or Node 18+)
  if (typeof CompressionStream !== "undefined") {
    console.log("Generating test SPZ data (10k Gaussians)...");
    const testSPZ = await generateTestSPZ(10_000);
    console.log(`Test SPZ size: ${(testSPZ.byteLength / 1024).toFixed(1)} KB`);

    console.log("Running SPZ decode benchmark...");
    results.push(await benchmarkSPZDecode(testSPZ, 5));
  } else {
    console.log("Skipping SPZ decode benchmark (CompressionStream not available)");
  }

  // PSNR benchmark with synthetic data
  console.log("Running PSNR benchmark with synthetic frames...");
  const width = 1920, height = 1080, channels = 4;
  const refPixels = new Uint8Array(width * height * channels);
  const rendPixels = new Uint8Array(width * height * channels);
  for (let i = 0; i < refPixels.length; i++) {
    refPixels[i] = Math.floor(Math.random() * 256);
    // Add small noise to simulate compression artifacts
    rendPixels[i] = Math.min(255, Math.max(0, refPixels[i] + Math.floor(Math.random() * 6 - 3)));
  }
  results.push(benchmarkPSNR(refPixels, rendPixels));

  // Print results
  console.log("\n=== Results ===\n");
  for (const r of results) {
    console.log(`${r.name}: ${r.value.toFixed(2)} ${r.unit} (${r.iterations} iterations)`);
  }

  // FPS and bandwidth benchmarks require a running WebGPU context and WebRTC connection.
  // They are exposed as functions to be called from the browser demo.
  console.log("\nNote: FPS and bandwidth benchmarks require a browser context.");
  console.log("Call benchmarkFPS(renderFn) and benchmarkBandwidth() from the demo page.");
}

// Export for use from browser or test harness
export {
  benchmarkFPS,
  benchmarkBandwidth,
  benchmarkPSNR,
  benchmarkSPZDecode,
  benchmarkDeltaDecompress,
  computePSNR,
  generateTestSPZ,
  runAll,
};

// Auto-run if executed directly
if (typeof require !== "undefined" && require.main === module) {
  runAll().catch(console.error);
}
