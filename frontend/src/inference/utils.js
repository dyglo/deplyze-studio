export function sigmoid(value) {
  return 1 / (1 + Math.exp(-value));
}

export function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

export function softmax(values) {
  const max = Math.max(...values);
  const exps = values.map((value) => Math.exp(value - max));
  const total = exps.reduce((sum, value) => sum + value, 0);
  return exps.map((value) => value / total);
}

export function getColorForIndex(index, alpha = 1) {
  const palette = [
    [239, 68, 68], [16, 185, 129], [59, 130, 246], [245, 158, 11],
    [139, 92, 246], [6, 182, 212], [132, 204, 22], [249, 115, 22],
  ];
  const [r, g, b] = palette[index % palette.length];
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

export function toPercent(value) {
  return `${(value * 100).toFixed(1)}%`;
}

export function topK(values, k) {
  return values
    .map((value, index) => ({ value, index }))
    .sort((left, right) => right.value - left.value)
    .slice(0, k);
}

export function computeMaskCoverage(mask) {
  let active = 0;
  for (let index = 0; index < mask.data.length; index += 1) {
    if (mask.data[index] > 0) active += 1;
  }
  return active / mask.data.length;
}
