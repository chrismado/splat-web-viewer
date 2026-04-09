import { writeFile } from "node:fs/promises";
import { join } from "node:path";

import { generateTestSPZ } from "../benchmarks/performance_test";

async function main(): Promise<void> {
  const outputPath = join(process.cwd(), "examples", "artifacts", "synthetic.spz");
  const buffer = await generateTestSPZ(2_000);
  await writeFile(outputPath, new Uint8Array(buffer));
  console.log(`Wrote synthetic SPZ fixture to ${outputPath}`);
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
