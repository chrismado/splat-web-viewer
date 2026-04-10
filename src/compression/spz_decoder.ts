/**
 * SPZ Format Decoder
 *
 * SPZ (Niantic Labs): packed binary Gaussian splat format.
 * 90% size reduction vs PLY files while preserving visual fidelity.
 *
 * Format: 16-byte header + gzipped arrays of:
 *   positions, alphas, colors, scales, rotations, spherical harmonics
 *
 * Header layout (16 bytes):
 *   magic:      u32  (0x5053_5A00 = "SPZ\0")
 *   version:    u32
 *   numPoints:  u32
 *   shDegree:   u8
 *   flags:      u8
 *   reserved:   u16
 *
 * Coordinate system: SPZ uses Right-Up-Back internally.
 * Must convert to standard WebGL coordinate frame (Right-Up-Forward).
 *
 * Reference: nianticlabs/spz
 */

export interface GaussianSplat {
  position: Float32Array;     // [x, y, z]
  alpha: number;
  color: Float32Array;        // [r, g, b]
  scale: Float32Array;        // [sx, sy, sz]
  rotation: Float32Array;     // quaternion [w, x, y, z]
  sphericalHarmonics: Float32Array;
}

const SPZ_MAGIC = 0x5053_5A00; // "SPZ\0"
const SPZ_HEADER_SIZE = 16;

interface SPZHeader {
  magic: number;
  version: number;
  numPoints: number;
  shDegree: number;
  flags: number;
}

/**
 * Decompress gzipped data using the browser's DecompressionStream API.
 */
async function decompressGzip(compressed: Uint8Array): Promise<Uint8Array> {
  const payload = new Uint8Array(compressed).buffer;
  const stream = new Response(new Blob([payload])).body!
    .pipeThrough(new DecompressionStream("gzip"));
  const reader = stream.getReader();
  const chunks: Uint8Array[] = [];
  let totalLength = 0;

  let streamDone = false;
  while (!streamDone) {
    const { done, value } = await reader.read();
    if (done) {
      streamDone = true;
      continue;
    }
    chunks.push(value);
    totalLength += value.byteLength;
  }

  const result = new Uint8Array(totalLength);
  let offset = 0;
  for (const chunk of chunks) {
    result.set(chunk, offset);
    offset += chunk.byteLength;
  }
  return result;
}

/**
 * Parse the 16-byte SPZ header.
 */
function parseHeader(view: DataView): SPZHeader {
  const magic = view.getUint32(0, true);
  if (magic !== SPZ_MAGIC) {
    throw new Error(
      `Invalid SPZ magic: 0x${magic.toString(16)}, expected 0x${SPZ_MAGIC.toString(16)}`
    );
  }
  return {
    magic,
    version: view.getUint32(4, true),
    numPoints: view.getUint32(8, true),
    shDegree: view.getUint8(12),
    flags: view.getUint8(13),
  };
}

/**
 * Number of spherical harmonic coefficients per Gaussian for a given SH degree.
 * SH degree 0: 0 extra (DC only, stored in color)
 * SH degree 1: 9 coefficients (3 bands x 3 RGB)
 * SH degree 2: 24 coefficients
 * SH degree 3: 45 coefficients
 */
function shCoeffCount(degree: number): number {
  // (degree + 1)^2 - 1 terms, times 3 for RGB
  if (degree === 0) return 0;
  return ((degree + 1) * (degree + 1) - 1) * 3;
}

/**
 * Decode an SPZ binary buffer into an array of GaussianSplat objects.
 *
 * Binary layout after header (all arrays are gzip-compressed):
 *   positions:  numPoints * 3 * float32 (x, y, z per point)
 *   alphas:     numPoints * uint8       (quantized opacity 0-255)
 *   colors:     numPoints * 3 * uint8   (quantized RGB 0-255)
 *   scales:     numPoints * 3 * uint8   (quantized log-scale 0-255)
 *   rotations:  numPoints * 4 * uint8   (quantized quaternion components)
 *   sh:         numPoints * shCoeffs * uint8 (quantized SH coefficients)
 *
 * Each section is stored as a length-prefixed gzipped block:
 *   u32 compressedSize + compressedSize bytes of gzip data
 */
export async function decodeSPZ(buffer: ArrayBuffer): Promise<GaussianSplat[]> {
  if (buffer.byteLength < SPZ_HEADER_SIZE) {
    throw new Error(`Buffer too small for SPZ header: ${buffer.byteLength} bytes`);
  }

  const view = new DataView(buffer);
  const header = parseHeader(view);
  const numPoints = header.numPoints;
  const shCoeffs = shCoeffCount(header.shDegree);

  if (numPoints === 0) return [];

  let offset = SPZ_HEADER_SIZE;
  const data = new Uint8Array(buffer);

  async function readCompressedSection(expectedBytes: number): Promise<Uint8Array> {
    if (offset + 4 > data.byteLength) {
      throw new Error("Unexpected end of SPZ data reading section length");
    }
    const compressedSize = view.getUint32(offset, true);
    offset += 4;
    if (offset + compressedSize > data.byteLength) {
      throw new Error("Unexpected end of SPZ data reading section payload");
    }
    const compressed = data.subarray(offset, offset + compressedSize);
    offset += compressedSize;
    const decompressed = await decompressGzip(compressed);
    if (decompressed.byteLength !== expectedBytes) {
      throw new Error(
        `Section size mismatch: got ${decompressed.byteLength}, expected ${expectedBytes}`
      );
    }
    return decompressed;
  }

  // Decompress all sections
  const positionsRaw = await readCompressedSection(numPoints * 3 * 4);
  const alphasRaw = await readCompressedSection(numPoints);
  const colorsRaw = await readCompressedSection(numPoints * 3);
  const scalesRaw = await readCompressedSection(numPoints * 3);
  const rotationsRaw = await readCompressedSection(numPoints * 4);
  const shRaw = shCoeffs > 0
    ? await readCompressedSection(numPoints * shCoeffs)
    : new Uint8Array(0);

  const positions = new Float32Array(positionsRaw.buffer, positionsRaw.byteOffset, numPoints * 3);

  // Build GaussianSplat array
  const splats: GaussianSplat[] = new Array(numPoints);

  for (let i = 0; i < numPoints; i++) {
    // Positions: direct float32
    const px = positions[i * 3 + 0];
    const py = positions[i * 3 + 1];
    const pz = positions[i * 3 + 2];

    // Alpha: quantized uint8 → sigmoid-inverse decoded to opacity
    // SPZ stores alpha as uint8 where 0=transparent, 255=opaque
    const alpha = alphasRaw[i] / 255.0;

    // Colors: quantized uint8 → [0, 1] float
    const cr = colorsRaw[i * 3 + 0] / 255.0;
    const cg = colorsRaw[i * 3 + 1] / 255.0;
    const cb = colorsRaw[i * 3 + 2] / 255.0;

    // Scales: quantized uint8 → log-space float
    // SPZ encodes scales as: uint8_val = (log(scale) - log_scale_min) / log_scale_range * 255
    // Decode: scale = exp(uint8_val / 255.0 * log_scale_range + log_scale_min)
    const LOG_SCALE_MIN = -10.0;
    const LOG_SCALE_RANGE = 16.0; // -10 to +6
    const sx = Math.exp(scalesRaw[i * 3 + 0] / 255.0 * LOG_SCALE_RANGE + LOG_SCALE_MIN);
    const sy = Math.exp(scalesRaw[i * 3 + 1] / 255.0 * LOG_SCALE_RANGE + LOG_SCALE_MIN);
    const sz = Math.exp(scalesRaw[i * 3 + 2] / 255.0 * LOG_SCALE_RANGE + LOG_SCALE_MIN);

    // Rotations: quantized uint8 → normalized quaternion [w, x, y, z]
    // SPZ stores quaternion components as uint8 mapped from [-1, 1]
    const rw = (rotationsRaw[i * 4 + 0] / 255.0) * 2.0 - 1.0;
    const rx = (rotationsRaw[i * 4 + 1] / 255.0) * 2.0 - 1.0;
    const ry = (rotationsRaw[i * 4 + 2] / 255.0) * 2.0 - 1.0;
    const rz = (rotationsRaw[i * 4 + 3] / 255.0) * 2.0 - 1.0;
    // Normalize quaternion
    const rlen = Math.sqrt(rw * rw + rx * rx + ry * ry + rz * rz) || 1.0;

    // Spherical harmonics: quantized uint8 → [-1, 1] float
    const sh = new Float32Array(shCoeffs);
    for (let j = 0; j < shCoeffs; j++) {
      sh[j] = (shRaw[i * shCoeffs + j] / 255.0) * 2.0 - 1.0;
    }

    splats[i] = {
      position: new Float32Array([px, py, pz]),
      alpha,
      color: new Float32Array([cr, cg, cb]),
      scale: new Float32Array([sx, sy, sz]),
      rotation: new Float32Array([rw / rlen, rx / rlen, ry / rlen, rz / rlen]),
      sphericalHarmonics: sh,
    };
  }

  // Convert coordinate system from SPZ Right-Up-Back to WebGL Right-Up-Forward
  for (let i = 0; i < splats.length; i++) {
    splats[i] = convertCoordinates(splats[i]);
  }

  return splats;
}

/**
 * Convert a single Gaussian from SPZ coordinate system (Right-Up-Back)
 * to standard WebGL coordinate system (Right-Up-Forward).
 *
 * Transform: negate the Z axis for positions and absorb the handedness
 * flip into the quaternion while keeping Gaussian scales positive.
 */
export function convertCoordinates(splat: GaussianSplat): GaussianSplat {
  const position = new Float32Array(splat.position);
  const scale = new Float32Array(splat.scale);
  const rotation = new Float32Array(splat.rotation);

  // Negate Z position (flip forward/back)
  position[2] *= -1;

  // Keep scales positive and absorb the axis flip into the rotation.
  // Reflecting the basis across Z corresponds to negating the x/y
  // imaginary components of the quaternion in [w, x, y, z] layout.
  rotation[1] *= -1; // qx
  rotation[2] *= -1; // qy

  return {
    position,
    alpha: splat.alpha,
    color: new Float32Array(splat.color),
    scale,
    rotation,
    sphericalHarmonics: new Float32Array(splat.sphericalHarmonics),
  };
}
