import { rotatedBoxToCorners, scaleBoxFromLetterbox, xywhToXyxy } from '../geometry';
import { createFallbackLabels, COCO_SKELETON } from '../labels';
import { parseClassificationOutputs, parseObbOutputs, parsePoseOutputs, parseSegmentationOutputs } from '../parsers';
import { clamp, softmax, topK } from '../utils';

describe('inference helpers', () => {
  test('letterbox scaling restores original box coordinates', () => {
    const meta = { padX: 0, padY: 80, scale: 0.5, originalWidth: 1280, originalHeight: 960 };
    const raw = xywhToXyxy(320, 320, 200, 100);
    const scaled = scaleBoxFromLetterbox(raw, meta);
    expect(scaled.x1).toBeCloseTo(440);
    expect(scaled.y1).toBeCloseTo(380);
    expect(scaled.width).toBeCloseTo(400);
    expect(scaled.height).toBeCloseTo(200);
  });

  test('softmax preserves ordering and sums to one', () => {
    const result = softmax([1, 2, 4]);
    expect(result[2]).toBeGreaterThan(result[1]);
    expect(result.reduce((sum, value) => sum + value, 0)).toBeCloseTo(1, 6);
  });

  test('topK returns highest values first', () => {
    expect(topK([0.2, 0.9, 0.5], 2)).toEqual([
      { index: 1, value: 0.9 },
      { index: 2, value: 0.5 },
    ]);
  });

  test('rotated boxes create four corners', () => {
    const corners = rotatedBoxToCorners(100, 100, 40, 20, Math.PI / 4);
    expect(corners).toHaveLength(4);
    expect(corners[0].x).not.toBe(corners[1].x);
  });

  test('fallback labels and skeleton metadata are available', () => {
    expect(createFallbackLabels(3)).toEqual(['Class 0', 'Class 1', 'Class 2']);
    expect(COCO_SKELETON.length).toBeGreaterThan(0);
    expect(clamp(15, 0, 10)).toBe(10);
  });

  test('classification parser returns top prediction', () => {
    const result = parseClassificationOutputs(
      { output0: { data: new Float32Array([0.1, 0.9, 0.2]), dims: [1, 3] } },
      { originalWidth: 224, originalHeight: 224 },
      { classNames: ['a', 'b', 'c'] },
      {},
    );
    expect(result.topPrediction.className).toBe('b');
    expect(result.predictions).toHaveLength(3);
  });

  test('yolo26 pose end-to-end rows are parsed', () => {
    const data = new Float32Array(300 * 57);
    const row = data.subarray(0, 57);
    row[0] = 10; row[1] = 20; row[2] = 110; row[3] = 220; row[4] = 0.9; row[5] = 0;
    for (let i = 0; i < 17; i += 1) {
      const offset = 6 + i * 3;
      row[offset] = 30 + i;
      row[offset + 1] = 40 + i;
      row[offset + 2] = 0.8;
    }
    const result = parsePoseOutputs(
      { output0: { data, dims: [1, 300, 57] } },
      { padX: 0, padY: 0, scale: 1, originalWidth: 640, originalHeight: 640 },
      { classNames: ['person'], keypointNames: Array.from({ length: 17 }, (_, i) => `k${i}`), keypointSkeleton: COCO_SKELETON },
      {},
    );
    expect(result.detections).toHaveLength(1);
    expect(result.detections[0].keypoints).toHaveLength(17);
  });

  test('yolo26 segmentation end-to-end rows are parsed', () => {
    const detectionData = new Float32Array(300 * 38);
    const detection = detectionData.subarray(0, 38);
    detection[0] = 10; detection[1] = 20; detection[2] = 110; detection[3] = 120; detection[4] = 0.95; detection[5] = 1;
    detection.fill(1, 6);
    const protoData = new Float32Array(32 * 4 * 4).fill(1);
    const result = parseSegmentationOutputs(
      {
        output0: { data: detectionData, dims: [1, 300, 38] },
        output1: { data: protoData, dims: [1, 32, 4, 4] },
      },
      { padX: 0, padY: 0, scale: 1, originalWidth: 128, originalHeight: 128, inputWidth: 128, inputHeight: 128 },
      { classNames: ['a', 'b'] },
      {},
    );
    expect(result.detections).toHaveLength(1);
    expect(result.detections[0].maskCoverage).toBeGreaterThan(0);
  });

  test('obb parser handles end-to-end rows', () => {
    const data = new Float32Array(300 * 7);
    data.set([10, 20, 110, 120, 0.8, 2, Math.PI / 6], 0);
    const result = parseObbOutputs(
      { output0: { data, dims: [1, 300, 7] } },
      { padX: 0, padY: 0, scale: 1, originalWidth: 640, originalHeight: 640 },
      { classNames: ['a', 'b', 'c'] },
      {},
    );
    expect(result.detections).toHaveLength(1);
    expect(result.detections[0].corners).toHaveLength(4);
  });
});
