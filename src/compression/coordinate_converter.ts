/**
 * Coordinate System Converter
 *
 * Converts Gaussian splat data between different 3D coordinate conventions.
 *
 * Supported systems:
 *   SPZ (Niantic):   Right-Up-Back   (RUB)
 *   WebGL/OpenGL:    Right-Up-Forward (RUF) -- negate Z from SPZ
 *   glTF:            Right-Up-Forward (RUF) -- same as WebGL
 */

import { GaussianSplat } from "./spz_decoder";

export enum CoordinateSystem {
  SPZ = "SPZ",           // Right-Up-Back
  WebGL = "WebGL",       // Right-Up-Forward
  GLTF = "GLTF",         // Right-Up-Forward (same as WebGL)
}

/**
 * Convert a GaussianSplat from SPZ (Right-Up-Back) to WebGL (Right-Up-Forward).
 * Negates the Z axis for positions and adjusts the rotation quaternion.
 */
export function convertSPZToWebGL(splat: GaussianSplat): GaussianSplat {
  const position = new Float32Array(splat.position);
  const scale = new Float32Array(splat.scale);
  const rotation = new Float32Array(splat.rotation);

  position[2] *= -1;
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

/**
 * Convert a GaussianSplat from WebGL (Right-Up-Forward) back to SPZ (Right-Up-Back).
 * Inverse of convertSPZToWebGL (same operation since negation is self-inverse).
 */
export function convertWebGLToSPZ(splat: GaussianSplat): GaussianSplat {
  return convertSPZToWebGL(splat); // Z negation is its own inverse
}

/**
 * Convert a GaussianSplat from glTF coordinates to WebGL.
 * glTF and WebGL share the same Right-Up-Forward convention, so this is identity.
 */
export function convertGLTFToWebGL(splat: GaussianSplat): GaussianSplat {
  return {
    position: new Float32Array(splat.position),
    alpha: splat.alpha,
    color: new Float32Array(splat.color),
    scale: new Float32Array(splat.scale),
    rotation: new Float32Array(splat.rotation),
    sphericalHarmonics: new Float32Array(splat.sphericalHarmonics),
  };
}

/**
 * Generic coordinate conversion between any two supported systems.
 * Converts through WebGL as the canonical intermediate representation.
 */
export function convertCoordinateSystem(
  splat: GaussianSplat,
  from: CoordinateSystem,
  to: CoordinateSystem
): GaussianSplat {
  if (from === to) {
    return {
      position: new Float32Array(splat.position),
      alpha: splat.alpha,
      color: new Float32Array(splat.color),
      scale: new Float32Array(splat.scale),
      rotation: new Float32Array(splat.rotation),
      sphericalHarmonics: new Float32Array(splat.sphericalHarmonics),
    };
  }

  // Step 1: Convert to WebGL (canonical)
  let webgl: GaussianSplat;
  switch (from) {
    case CoordinateSystem.SPZ:
      webgl = convertSPZToWebGL(splat);
      break;
    case CoordinateSystem.WebGL:
    case CoordinateSystem.GLTF:
      webgl = convertGLTFToWebGL(splat);
      break;
    default:
      throw new Error(`Unsupported source coordinate system: ${from}`);
  }

  // Step 2: Convert from WebGL to target
  switch (to) {
    case CoordinateSystem.WebGL:
    case CoordinateSystem.GLTF:
      return webgl;
    case CoordinateSystem.SPZ:
      return convertWebGLToSPZ(webgl);
    default:
      throw new Error(`Unsupported target coordinate system: ${to}`);
  }
}
