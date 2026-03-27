import { nonMaxSuppression, rotatedBoxToCorners, rotatedCornersToAabb, scaleBoxFromLetterbox, xywhToXyxy } from './geometry';
import { createFallbackLabels } from './labels';
import { clamp, computeMaskCoverage, sigmoid, softmax, topK } from './utils';

function getPrimaryOutput(outputs) {
  const entries = Object.values(outputs);
  if (!entries.length) {
    throw new Error('ONNX model returned no outputs');
  }
  return entries[0];
}

function tensorToRows(tensor) {
  const dims = tensor.dims;
  if (dims.length !== 3) {
    throw new Error(`Unsupported tensor rank ${dims.length}`);
  }

  const [, dimA, dimB] = dims;
  const rows = [];

  if (dimA < dimB) {
    for (let column = 0; column < dimB; column += 1) {
      const row = [];
      for (let channel = 0; channel < dimA; channel += 1) {
        row.push(tensor.data[channel * dimB + column]);
      }
      rows.push(row);
    }
    return rows;
  }

  for (let rowIndex = 0; rowIndex < dimA; rowIndex += 1) {
    const start = rowIndex * dimB;
    rows.push(Array.from(tensor.data.slice(start, start + dimB)));
  }
  return rows;
}

function assertCompatibleYolo26(output, task, family) {
  if (family !== 'yolo26') return;
  const dims = output.dims;
  if (dims.length === 3 && (dims.includes(300) || dims[dims.length - 1] === 6)) {
    throw new Error(`Unsupported YOLO26 end-to-end export for ${task}. Export with end2end=False.`);
  }
}

function getClassNames(modelConfig, count, prefix) {
  if (modelConfig.classNames?.length) return modelConfig.classNames;
  return createFallbackLabels(count, prefix);
}

function parseEndToEndRow(row) {
  return {
    x1: row[0],
    y1: row[1],
    x2: row[2],
    y2: row[3],
    confidence: row[4],
    classId: Math.round(row[5]),
    extras: row.slice(6),
  };
}

function scaleDirectBox(box, meta) {
  return {
    x1: clamp((box.x1 - meta.padX) / meta.scale, 0, meta.originalWidth),
    y1: clamp((box.y1 - meta.padY) / meta.scale, 0, meta.originalHeight),
    x2: clamp((box.x2 - meta.padX) / meta.scale, 0, meta.originalWidth),
    y2: clamp((box.y2 - meta.padY) / meta.scale, 0, meta.originalHeight),
    width: clamp((box.x2 - box.x1) / meta.scale, 0, meta.originalWidth),
    height: clamp((box.y2 - box.y1) / meta.scale, 0, meta.originalHeight),
    centerX: clamp(((box.x1 + box.x2) / 2 - meta.padX) / meta.scale, 0, meta.originalWidth),
    centerY: clamp(((box.y1 + box.y2) / 2 - meta.padY) / meta.scale, 0, meta.originalHeight),
  };
}

export function parseDetectionOutputs(outputs, meta, modelConfig, options = {}) {
  const output = getPrimaryOutput(outputs);
  const rows = tensorToRows(output);
  const classCount = modelConfig.classNames?.length ?? Math.max(1, rows[0].length - 4);
  const scoreThreshold = options.scoreThreshold ?? 0.25;
  const candidates = [];

  if (rows[0].length === 6) {
    rows.forEach((row) => {
      const parsed = parseEndToEndRow(row);
      if (parsed.confidence < scoreThreshold) return;
      const box = scaleDirectBox(parsed, meta);
      candidates.push({
        classId: parsed.classId,
        confidence: parsed.confidence,
        className: getClassNames(modelConfig, classCount, 'Class')[parsed.classId] ?? `Class ${parsed.classId}`,
        box,
      });
    });
    const detections = nonMaxSuppression(candidates, options.iouThreshold ?? 0.45).map((item, index) => ({
      id: index,
      classId: item.classId,
      className: item.className,
      confidence: item.confidence,
      bbox: item.box,
    }));
    return {
      task: 'detect',
      detections,
      image: { width: meta.originalWidth, height: meta.originalHeight },
      inferenceTimeMs: options.inferenceTimeMs ?? 0,
      summary: {
        totalDetections: detections.length,
        classes: [...new Set(detections.map((item) => item.className))],
      },
    };
  }

  rows.forEach((row) => {
    if (row.length < 4 + classCount) return;
    const scores = row.slice(4, 4 + classCount);
    let bestScore = -Infinity;
    let bestClass = 0;
    scores.forEach((score, classId) => {
      if (score > bestScore) {
        bestScore = score;
        bestClass = classId;
      }
    });
    if (bestScore < scoreThreshold) return;
    const rawBox = xywhToXyxy(row[0], row[1], row[2], row[3]);
    const box = scaleBoxFromLetterbox(rawBox, meta);
    candidates.push({
      classId: bestClass,
      confidence: bestScore,
      className: getClassNames(modelConfig, classCount, 'Class')[bestClass] ?? `Class ${bestClass}`,
      box,
    });
  });

  const detections = nonMaxSuppression(candidates, options.iouThreshold ?? 0.45).map((item, index) => ({
    id: index,
    classId: item.classId,
    className: item.className,
    confidence: item.confidence,
    bbox: item.box,
  }));

  return {
    task: 'detect',
    detections,
    image: { width: meta.originalWidth, height: meta.originalHeight },
    inferenceTimeMs: options.inferenceTimeMs ?? 0,
    summary: {
      totalDetections: detections.length,
      classes: [...new Set(detections.map((item) => item.className))],
    },
  };
}

function rebuildMask(protoTensor, coefficients, box, meta) {
  const [, channels, protoHeight, protoWidth] = protoTensor.dims;
  const flattenedProto = protoTensor.data;
  const maskLowRes = new Float32Array(protoHeight * protoWidth);

  for (let channel = 0; channel < channels; channel += 1) {
    const channelOffset = channel * protoHeight * protoWidth;
    for (let index = 0; index < protoHeight * protoWidth; index += 1) {
      maskLowRes[index] += coefficients[channel] * flattenedProto[channelOffset + index];
    }
  }

  const mask = new Uint8ClampedArray(meta.originalWidth * meta.originalHeight);
  const xStart = Math.max(0, Math.floor(box.x1));
  const xEnd = Math.min(meta.originalWidth, Math.ceil(box.x2));
  const yStart = Math.max(0, Math.floor(box.y1));
  const yEnd = Math.min(meta.originalHeight, Math.ceil(box.y2));

  for (let y = yStart; y < yEnd; y += 1) {
    for (let x = xStart; x < xEnd; x += 1) {
      const letterboxedX = x * meta.scale + meta.padX;
      const letterboxedY = y * meta.scale + meta.padY;
      const px = clamp(Math.floor((letterboxedX / meta.inputWidth) * protoWidth), 0, protoWidth - 1);
      const py = clamp(Math.floor((letterboxedY / meta.inputHeight) * protoHeight), 0, protoHeight - 1);
      const lowResIndex = py * protoWidth + px;
      if (sigmoid(maskLowRes[lowResIndex]) > 0.5) {
        mask[y * meta.originalWidth + x] = 255;
      }
    }
  }

  return {
    width: meta.originalWidth,
    height: meta.originalHeight,
    data: mask,
  };
}

export function parseSegmentationOutputs(outputs, meta, modelConfig, options = {}) {
  const tensors = Object.values(outputs);
  if (tensors.length < 2) {
    throw new Error('Segmentation models must return detection and proto outputs.');
  }
  const detectionTensor = tensors.find((tensor) => tensor.dims.length === 3);
  const protoTensor = tensors.find((tensor) => tensor.dims.length === 4);
  if (!detectionTensor || !protoTensor) {
    throw new Error('Unsupported segmentation output signature.');
  }
  const rows = tensorToRows(detectionTensor);
  const maskChannels = protoTensor.dims[1];
  const isEndToEnd = rows[0].length === maskChannels + 6;
  const classCount = isEndToEnd ? (modelConfig.classNames?.length ?? 80) : (modelConfig.classNames?.length ?? Math.max(1, rows[0].length - 4 - maskChannels));
  const candidates = [];

  rows.forEach((row) => {
    let bestScore;
    let bestClass;
    let coefficients;
    let box;

    if (isEndToEnd) {
      const parsed = parseEndToEndRow(row);
      bestScore = parsed.confidence;
      bestClass = parsed.classId;
      coefficients = parsed.extras.slice(0, maskChannels);
      box = scaleDirectBox(parsed, meta);
    } else {
      if (row.length < 4 + classCount + maskChannels) return;
      const scores = row.slice(4, 4 + classCount);
      bestScore = -Infinity;
      bestClass = 0;
      scores.forEach((score, classId) => {
        if (score > bestScore) {
          bestScore = score;
          bestClass = classId;
        }
      });
      coefficients = row.slice(4 + classCount, 4 + classCount + maskChannels);
      box = scaleBoxFromLetterbox(xywhToXyxy(row[0], row[1], row[2], row[3]), meta);
    }

    if (bestScore < (options.scoreThreshold ?? 0.25)) return;
    candidates.push({
      classId: bestClass,
      className: getClassNames(modelConfig, classCount, 'Class')[bestClass] ?? `Class ${bestClass}`,
      confidence: bestScore,
      box,
      coefficients,
    });
  });

  const detections = nonMaxSuppression(candidates, options.iouThreshold ?? 0.5).map((item, index) => {
    const mask = rebuildMask(protoTensor, item.coefficients, item.box, meta);
    return {
      id: index,
      classId: item.classId,
      className: item.className,
      confidence: item.confidence,
      bbox: item.box,
      mask,
      maskCoverage: computeMaskCoverage(mask),
    };
  });

  return {
    task: 'segment',
    detections,
    image: { width: meta.originalWidth, height: meta.originalHeight },
    inferenceTimeMs: options.inferenceTimeMs ?? 0,
    summary: {
      totalDetections: detections.length,
      classes: [...new Set(detections.map((item) => item.className))],
    },
  };
}

export function parsePoseOutputs(outputs, meta, modelConfig, options = {}) {
  const output = getPrimaryOutput(outputs);
  const rows = tensorToRows(output);
  const keypointCount = modelConfig.keypointNames?.length ?? 17;
  const isEndToEnd = rows[0].length === 6 + keypointCount * 3;
  const classCount = isEndToEnd ? 1 : (modelConfig.classNames?.length ?? 1);
  const expectedMin = isEndToEnd ? 6 + keypointCount * 3 : 4 + classCount + keypointCount * 3;
  if (rows[0].length < expectedMin) {
    throw new Error('Unsupported pose output signature.');
  }

  const candidates = [];
  rows.forEach((row) => {
    const confidence = isEndToEnd ? row[4] : Math.max(...row.slice(4, 4 + classCount));
    if (confidence < (options.scoreThreshold ?? 0.25)) return;
    const box = isEndToEnd
      ? scaleDirectBox(parseEndToEndRow(row), meta)
      : scaleBoxFromLetterbox(xywhToXyxy(row[0], row[1], row[2], row[3]), meta);
    const keypoints = [];
    const offset = isEndToEnd ? 6 : 4 + classCount;
    for (let index = 0; index < keypointCount; index += 1) {
      const pointOffset = offset + index * 3;
      keypoints.push({
        name: modelConfig.keypointNames?.[index] ?? `kp_${index}`,
        x: clamp((row[pointOffset] - meta.padX) / meta.scale, 0, meta.originalWidth),
        y: clamp((row[pointOffset + 1] - meta.padY) / meta.scale, 0, meta.originalHeight),
        confidence: row[pointOffset + 2],
      });
    }
    candidates.push({
      classId: 0,
      className: 'person',
      confidence,
      box,
      keypoints,
    });
  });

  const detections = nonMaxSuppression(candidates, options.iouThreshold ?? 0.45).map((item, index) => ({
    id: index,
    classId: item.classId,
    className: item.className,
    confidence: item.confidence,
    bbox: item.box,
    keypoints: item.keypoints,
    skeleton: modelConfig.keypointSkeleton ?? [],
  }));

  return {
    task: 'pose',
    detections,
    image: { width: meta.originalWidth, height: meta.originalHeight },
    inferenceTimeMs: options.inferenceTimeMs ?? 0,
    summary: { totalDetections: detections.length },
  };
}

export function parseClassificationOutputs(outputs, meta, modelConfig, options = {}) {
  const output = getPrimaryOutput(outputs);
  const rawValues = Array.from(output.data);
  const probabilities = softmax(rawValues);
  const labels = modelConfig.classNames?.length ? modelConfig.classNames : createFallbackLabels(probabilities.length, 'Class');
  const predictions = topK(probabilities, options.topK ?? 5).map((item) => ({
    classId: item.index,
    className: labels[item.index] ?? `Class ${item.index}`,
    confidence: item.value,
  }));

  return {
    task: 'classify',
    predictions,
    topPrediction: predictions[0] ?? null,
    image: { width: meta.originalWidth, height: meta.originalHeight },
    inferenceTimeMs: options.inferenceTimeMs ?? 0,
    summary: { totalClasses: probabilities.length },
  };
}

export function parseObbOutputs(outputs, meta, modelConfig, options = {}) {
  const output = getPrimaryOutput(outputs);
  const rows = tensorToRows(output);
  const isEndToEnd = rows[0].length === 7;
  const classCount = isEndToEnd ? (modelConfig.classNames?.length ?? 15) : (modelConfig.classNames?.length ?? Math.max(1, rows[0].length - 5));
  const candidates = [];

  rows.forEach((row) => {
    let bestScore;
    let bestClass;
    let centerX;
    let centerY;
    let width;
    let height;
    let angle;

    if (isEndToEnd) {
      bestScore = row[4];
      bestClass = Math.round(row[5]);
      angle = row[6];
      const directBox = scaleDirectBox({
        x1: row[0],
        y1: row[1],
        x2: row[2],
        y2: row[3],
      }, meta);
      centerX = directBox.centerX;
      centerY = directBox.centerY;
      width = directBox.width;
      height = directBox.height;
    } else {
      if (row.length < 5 + classCount) return;
      const scores = row.slice(4, 4 + classCount);
      bestScore = -Infinity;
      bestClass = 0;
      scores.forEach((score, classId) => {
        if (score > bestScore) {
          bestScore = score;
          bestClass = classId;
        }
      });
      centerX = (row[0] - meta.padX) / meta.scale;
      centerY = (row[1] - meta.padY) / meta.scale;
      width = row[2] / meta.scale;
      height = row[3] / meta.scale;
      angle = row[4 + classCount];
    }

    if (bestScore < (options.scoreThreshold ?? 0.25)) return;
    const corners = rotatedBoxToCorners(centerX, centerY, width, height, angle);
    const box = rotatedCornersToAabb(corners);
    candidates.push({
      classId: bestClass,
      className: getClassNames(modelConfig, classCount, 'Class')[bestClass] ?? `Class ${bestClass}`,
      confidence: bestScore,
      box,
      corners,
      angle,
      width,
      height,
      centerX,
      centerY,
    });
  });

  const detections = nonMaxSuppression(candidates, options.iouThreshold ?? 0.45).map((item, index) => ({
    id: index,
    classId: item.classId,
    className: item.className,
    confidence: item.confidence,
    bbox: item.box,
    corners: item.corners,
    angle: item.angle,
    centerX: item.centerX,
    centerY: item.centerY,
    width: item.width,
    height: item.height,
  }));

  return {
    task: 'obb',
    detections,
    image: { width: meta.originalWidth, height: meta.originalHeight },
    inferenceTimeMs: options.inferenceTimeMs ?? 0,
    summary: {
      totalDetections: detections.length,
      classes: [...new Set(detections.map((item) => item.className))],
    },
  };
}
