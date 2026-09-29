const completeNumber = /^[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?$/;

export function isCompleteNumericText(text: string) {
  return completeNumber.test(text) && Number.isFinite(Number(text));
}
