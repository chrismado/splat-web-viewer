import { readFile } from "node:fs/promises";
import { join } from "node:path";

import { generateTestSPZ } from "../benchmarks/performance_test";
import { decodeSPZ } from "../src/compression/spz_decoder";

async function main(): Promise<void> {
  const artifactPath = join(process.cwd(), "examples", "artifacts", "synthetic.spz");

  let payload: Uint8Array;
  try {
    payload = await readFile(artifactPath);
  } catch {
    payload = new Uint8Array(await generateTestSPZ(256));
  }

  const splats = await decodeSPZ(payload.buffer.slice(payload.byteOffset, payload.byteOffset + payload.byteLength));
  console.log(`Decoded ${splats.length} splats`);
  if (splats[0]) {
    console.log("First splat alpha:", splats[0].alpha.toFixed(4));
  }
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
