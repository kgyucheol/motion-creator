// Usage: node --experimental-transform-types scripts/cylinder-sweep.ts variants.json results.json
// variants.json: { scene, variants: [{ name, params, waypoints }] }  (written by scripts/cylinder_scoop_sweep.py)
import { readFileSync, writeFileSync } from 'node:fs';
import { simulateScoop, NSE_SCOOP_PROFILE } from '../lib/cylinder-sim.ts';

const [input, output] = process.argv.slice(2);
const { scene, variants, profile, options } = JSON.parse(readFileSync(input, 'utf8'));
const results = variants.map((variant: { name: string; params: Record<string, number>; blades: never[] }) => {
  const { frames, ...result } = simulateScoop(scene, variant.blades, profile ?? NSE_SCOOP_PROFILE, options ?? {});
  return { name: variant.name, params: variant.params, ...result, frameCount: frames.length };
});
writeFileSync(output, JSON.stringify(results));
