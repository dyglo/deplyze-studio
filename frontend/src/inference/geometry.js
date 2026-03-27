import { clamp } from './utils';

export function xywhToXyxy(x, y, width, height) {
  return {
    x1: x - width / 2,
    y1: y - height / 2,
    x2: x + width / 2,
    y2: y + height / 2,
  };
}

export function scaleBoxFromLetterbox(box, meta) {
  const x1 = clamp((box.x1 - meta.padX) / meta.scale, 0, meta.originalWidth);
  const y1 = clamp((box.y1 - meta.padY) / meta.scale, 0, meta.originalHeight);
  const x2 = clamp((box.x2 - meta.padX) / meta.scale, 0, meta.originalWidth);
  const y2 = clamp((box.y2 - meta.padY) / meta.scale, 0, meta.originalHeight);
  return {
    x1,
    y1,
    x2,
    y2,
    width: Math.max(0, x2 - x1),
    height: Math.max(0, y2 - y1),
    centerX: (x1 + x2) / 2,
    centerY: (y1 + y2) / 2,
  };
}

export function computeIoU(left, right) {
  const x1 = Math.max(left.x1, right.x1);
  const y1 = Math.max(left.y1, right.y1);
  const x2 = Math.min(left.x2, right.x2);
  const y2 = Math.min(left.y2, right.y2);
  const intersection = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  const leftArea = Math.max(0, left.x2 - left.x1) * Math.max(0, left.y2 - left.y1);
  const rightArea = Math.max(0, right.x2 - right.x1) * Math.max(0, right.y2 - right.y1);
  const union = leftArea + rightArea - intersection;
  return union <= 0 ? 0 : intersection / union;
}

export function nonMaxSuppression(items, iouThreshold = 0.45) {
  const ordered = [...items].sort((left, right) => right.confidence - left.confidence);
  const kept = [];

  while (ordered.length) {
    const candidate = ordered.shift();
    kept.push(candidate);
    for (let index = ordered.length - 1; index >= 0; index -= 1) {
      if (ordered[index].classId !== candidate.classId) continue;
      if (computeIoU(candidate.box, ordered[index].box) > iouThreshold) {
        ordered.splice(index, 1);
      }
    }
  }

  return kept;
}

export function rotatedBoxToCorners(centerX, centerY, width, height, angleRadians) {
  const cosine = Math.cos(angleRadians);
  const sine = Math.sin(angleRadians);
  const halfWidth = width / 2;
  const halfHeight = height / 2;
  const baseCorners = [
    [-halfWidth, -halfHeight],
    [halfWidth, -halfHeight],
    [halfWidth, halfHeight],
    [-halfWidth, halfHeight],
  ];

  return baseCorners.map(([x, y]) => ({
    x: centerX + x * cosine - y * sine,
    y: centerY + x * sine + y * cosine,
  }));
}

export function rotatedCornersToAabb(corners) {
  const xs = corners.map((corner) => corner.x);
  const ys = corners.map((corner) => corner.y);
  return {
    x1: Math.min(...xs),
    y1: Math.min(...ys),
    x2: Math.max(...xs),
    y2: Math.max(...ys),
  };
}
