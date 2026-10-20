// Runs extraction attempts off the page's main thread.
import { runAttempt } from '../lib/lab-batch.ts';
import type { BladePath, BladeProfile, Scene, SimOptions } from '../lib/cylinder-sim.ts';

type Request = { pile: Scene; targetId: string; paths: BladePath[]; referenceCenter: number[]; order: number;
  options: Partial<SimOptions>; settle: boolean; profile?: BladeProfile | BladeProfile[]; referenceRimUp?: number | null };

self.onmessage = (event: MessageEvent<Request>) => {
  const { pile, targetId, paths, referenceCenter, order, options, settle, profile, referenceRimUp } = event.data;
  try {
    self.postMessage({ type: 'attempt', attempt: runAttempt(pile, targetId, paths, referenceCenter, order, options, settle, profile,
      referenceRimUp ?? null) });
  } catch (error) {
    self.postMessage({ type: 'error', message: error instanceof Error ? error.message : String(error) });
  }
};
