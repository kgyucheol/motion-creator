export type Keyframe = {
  name: string;
  duration: number;
  qpos: number[];
  pins: string[];
  samples?: number[][];
};

export function duplicateKeyframeAfter(keyframes: Keyframe[], selectedIndex: number) {
  if (selectedIndex < 0 || selectedIndex >= keyframes.length) return { keyframes, index: selectedIndex };
  const index = selectedIndex + 1;
  const duplicate = structuredClone(keyframes[selectedIndex]);
  return { keyframes: [...keyframes.slice(0, index), duplicate, ...keyframes.slice(index)], index };
}
