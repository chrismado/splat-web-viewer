import { describe, expect, it } from "vitest";

import { convertCoordinates, decodeSPZ, GaussianSplat } from "../src/compression/spz_decoder";

describe("convertCoordinates", () => {
  it("negates Z position while keeping scales positive", () => {
    const splat: GaussianSplat = {
      position: new Float32Array([1, 2, 3]),
      alpha: 0.5,
      color: new Float32Array([0.1, 0.2, 0.3]),
      scale: new Float32Array([0.4, 0.5, 0.6]),
      rotation: new Float32Array([1, 0, 0, 0]),
      sphericalHarmonics: new Float32Array(0),
    };

    const converted = convertCoordinates(splat);

    expect(converted.position[0]).toBeCloseTo(1, 6);
    expect(converted.position[1]).toBeCloseTo(2, 6);
    expect(converted.position[2]).toBeCloseTo(-3, 6);
    expect(converted.scale[0]).toBeCloseTo(0.4, 6);
    expect(converted.scale[1]).toBeCloseTo(0.5, 6);
    expect(converted.scale[2]).toBeCloseTo(0.6, 6);
    expect(converted.alpha).toBeCloseTo(0.5, 6);
  });

  it("preserves color and spherical harmonics", () => {
    const splat: GaussianSplat = {
      position: new Float32Array([0, 0, 0]),
      alpha: 1.0,
      color: new Float32Array([0.9, 0.8, 0.7]),
      scale: new Float32Array([1, 1, 1]),
      rotation: new Float32Array([1, 0, 0, 0]),
      sphericalHarmonics: new Float32Array([0.1, 0.2, 0.3]),
    };

    const converted = convertCoordinates(splat);

    expect(converted.color[0]).toBeCloseTo(0.9, 6);
    expect(converted.color[1]).toBeCloseTo(0.8, 6);
    expect(converted.color[2]).toBeCloseTo(0.7, 6);
    expect(converted.sphericalHarmonics.length).toBe(3);
  });

  it("does not mutate the original splat", () => {
    const splat: GaussianSplat = {
      position: new Float32Array([1, 2, 3]),
      alpha: 0.5,
      color: new Float32Array([0.1, 0.2, 0.3]),
      scale: new Float32Array([0.4, 0.5, 0.6]),
      rotation: new Float32Array([1, 0, 0, 0]),
      sphericalHarmonics: new Float32Array(0),
    };

    convertCoordinates(splat);

    expect(splat.position[2]).toBeCloseTo(3, 6);
    expect(splat.scale[2]).toBeCloseTo(0.6, 6);
  });
});

describe("decodeSPZ", () => {
  it("throws on an empty buffer", async () => {
    await expect(decodeSPZ(new ArrayBuffer(0))).rejects.toThrow(/Buffer too small/);
  });

  it("throws on a buffer smaller than the header", async () => {
    await expect(decodeSPZ(new ArrayBuffer(8))).rejects.toThrow(/Buffer too small/);
  });

  it("throws on a bad magic number", async () => {
    const buf = new ArrayBuffer(16);
    const view = new DataView(buf);
    view.setUint32(0, 0xDEADBEEF, true);

    await expect(decodeSPZ(buf)).rejects.toThrow(/Invalid SPZ magic/);
  });

  it("returns an empty array for zero points", async () => {
    const buf = new ArrayBuffer(16);
    const view = new DataView(buf);
    view.setUint32(0, 0x5053_5A00, true);
    view.setUint32(4, 1, true);
    view.setUint32(8, 0, true);
    view.setUint8(12, 0);
    view.setUint8(13, 0);

    await expect(decodeSPZ(buf)).resolves.toEqual([]);
  });
});
