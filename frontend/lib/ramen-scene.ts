import type { SceneObject } from './scene-objects';

/** Four horizontal bundles per layer, three layers, in an open five-panel carton. */
export function ramenScene(prefix: string): SceneObject[] {
  const diameter = .10, length = .23, gap = .003, wall = .01;
  const width = 4 * diameter + 5 * gap, depth = length + 2 * gap;
  const height = 3 * diameter + 4 * gap, floor = .62, x = .48;
  const objects: SceneObject[] = [];
  function add(name: string, shape: SceneObject['shape'], position: number[], size: number[], fixed = false) {
    objects.push({ id: `${prefix}-${objects.length}`, name, shape, position, size,
      quaternion_xyzw: shape === 'cylinder' ? [0, Math.SQRT1_2, 0, Math.SQRT1_2] : [0, 0, 0, 1],
      mass_kg: fixed ? 1 : .15, friction: .7, color: fixed ? '#b78a57' : '#e7d8be',
      opacity: fixed ? .45 : 1, visible: true, fixed,
      placement: { prevent_overlap: false, surface_snap: false, ground_lock: false } });
  }
  add('상자 바닥 (고정)', 'box', [x, 0, floor - wall / 2], [depth + 2 * wall, width + 2 * wall, wall], true);
  for (const sign of [-1, 1]) {
    add('상자 앞뒤 벽 (고정)', 'box', [x + sign * (depth + wall) / 2, 0, floor + height / 2], [wall, width + 2 * wall, height], true);
    add('상자 옆벽 (고정)', 'box', [x, sign * (width + wall) / 2, floor + height / 2], [depth, wall, height], true);
  }
  for (let layer = 0; layer < 3; layer++) for (let column = 0; column < 4; column++) {
    add(`용기 묶음 ${layer + 1}층-${column + 1}`, 'cylinder',
      [x, (column - 1.5) * (diameter + gap), floor + gap + diameter / 2 + layer * (diameter + gap)],
      [diameter, diameter, length]);
  }
  return objects;
}
