// Usage: node --experimental-transform-types scripts/cylinder-settle.ts scene.json settled.json
import { readFileSync, writeFileSync } from 'node:fs';
import { settleScene } from '../lib/cylinder-sim.ts';

const [input, output] = process.argv.slice(2);
writeFileSync(output, JSON.stringify(settleScene(JSON.parse(readFileSync(input, 'utf8')))));
