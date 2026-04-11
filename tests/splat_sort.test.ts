import { describe, expect, it } from "vitest";

import { GaussianSplat } from "../src/compression/spz_decoder";
import {
  cameraSortSignature,
  distanceSquared,
  sortSplatsBackToFront,
} from "../src/renderer/splat_sort";

function makeSplat(x: number, y: number, z: number): GaussianSplat {
  return {
    position: new Float32Array([x, y, z]),
    alpha: 1,
    color: new Float32Array([1, 1, 1]),
    scale: new Float32Array([1, 1, 1]),
    rotation: new Float32Array([1, 0, 0, 0]),
    sphericalHarmonics: new Float32Array(0),
  };
}

describe("splat_sort", () => {
  it("computes squared distance without a square root", () => {
    const camera = new Float32Array([0, 0, 5]);
    const splat = makeSplat(0, 0, 1);

    expect(distanceSquared(splat.position, camera)).toBeCloseTo(16, 6);
  });

  it("sorts farther splats before nearer splats for alpha blending", () => {
    const camera = new Float32Array([0, 0, 5]);
    const splats = [
      makeSplat(0, 0, 4),
      makeSplat(0, 0, -3),
      makeSplat(0, 0, 1),
    ];

    const sorted = sortSplatsBackToFront(splats, camera);

    expect(sorted[0].position[2]).toBe(-3);
    expect(sorted[1].position[2]).toBe(1);
    expect(sorted[2].position[2]).toBe(4);
  });

  it("builds a stable cache signature from camera coordinates", () => {
    const signature = cameraSortSignature(new Float32Array([1.23456, 0, -4.4444]));
    expect(signature).toBe("1.235|0.000|-4.444");
  });
});
