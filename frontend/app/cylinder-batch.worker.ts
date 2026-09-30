// Runs the all-cylinder test off the page's main thread and reports each attempt as it finishes.
import { runBatch } from '../lib/lab-batch.ts';
import type { BladePath, Scene, SimOptions } from '../lib/cylinder-sim.ts';

type Request = { scene: Scene; paths: BladePath[]; referenceId: string; options: Partial<SimOptions> };

self.onmessage = (event: MessageEvent<Request>) => {
  const { scene, paths, referenceId, options } = event.data;
  try {
    const report = runBatch(scene, paths, referenceId, options, undefined, (attempt, index, total) =>
      self.postMessage({ type: 'attempt', attempt, index, total }));
    self.postMessage({ type: 'done', extracted: report.extracted, total: report.total, stoppedAt: report.stoppedAt });
  } catch (error) {
    self.postMessage({ type: 'error', message: error instanceof Error ? error.message : String(error) });
  }
};
