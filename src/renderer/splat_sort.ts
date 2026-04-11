import { GaussianSplat } from "../compression/spz_decoder";

export function distanceSquared(position: Float32Array, cameraPosition: Float32Array): number {
  const dx = position[0] - cameraPosition[0];
  const dy = position[1] - cameraPosition[1];
  const dz = position[2] - cameraPosition[2];
  return dx * dx + dy * dy + dz * dz;
}

export function cameraSortSignature(cameraPosition: Float32Array): string {
  return Array.from(cameraPosition)
    .map((value) => value.toFixed(3))
    .join("|");
}

export function sortSplatsBackToFront(
  splats: GaussianSplat[],
  cameraPosition: Float32Array,
): GaussianSplat[] {
  return [...splats].sort(
    (a, b) => distanceSquared(b.position, cameraPosition) - distanceSquared(a.position, cameraPosition),
  );
}
