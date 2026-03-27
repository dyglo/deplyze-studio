import { getColorForIndex, toPercent } from './utils';

function drawLabel(context, x, y, label, color) {
  context.font = 'bold 13px sans-serif';
  const metrics = context.measureText(label);
  const width = metrics.width + 12;
  const height = 22;
  const drawY = Math.max(0, y - height);
  context.fillStyle = color;
  context.fillRect(x, drawY, width, height);
  context.fillStyle = '#ffffff';
  context.fillText(label, x + 6, drawY + 15);
}

function drawPill(context, x, y, text, color = 'rgba(17, 24, 39, 0.92)') {
  context.save();
  context.font = 'bold 16px sans-serif';
  const metrics = context.measureText(text);
  const width = metrics.width + 20;
  const height = 32;
  context.fillStyle = color;
  context.beginPath();
  context.roundRect(x, y, width, height, 999);
  context.fill();
  context.fillStyle = '#ffffff';
  context.fillText(text, x + 10, y + 21);
  context.restore();
  return { width, height };
}

function parseColor(color) {
  const match = color.match(/rgba\((\d+), (\d+), (\d+),/);
  if (!match) return [255, 255, 255];
  return [Number(match[1]), Number(match[2]), Number(match[3])];
}

function getStrokeWidth(canvas) {
  return Math.max(3, Math.min(4, Math.round(Math.max(canvas.width, canvas.height) / 700)));
}

function getBoxStrokeWidth(canvas) {
  return getStrokeWidth(canvas) * 6;
}

function drawMask(context, mask, rgb, alpha) {
  const maskCanvas = document.createElement('canvas');
  maskCanvas.width = mask.width;
  maskCanvas.height = mask.height;
  const maskContext = maskCanvas.getContext('2d');
  const imageData = maskContext.createImageData(mask.width, mask.height);
  for (let pixel = 0; pixel < mask.data.length; pixel += 1) {
    const maskValue = mask.data[pixel];
    if (!maskValue) continue;
    const offset = pixel * 4;
    imageData.data[offset] = rgb[0];
    imageData.data[offset + 1] = rgb[1];
    imageData.data[offset + 2] = rgb[2];
    imageData.data[offset + 3] = alpha;
  }
  maskContext.putImageData(imageData, 0, 0);
  context.drawImage(maskCanvas, 0, 0);
}

function drawOutlinedLine(context, start, end, color, width) {
  context.save();
  context.lineCap = 'round';
  context.strokeStyle = 'rgba(15, 23, 42, 0.85)';
  context.lineWidth = width + 2;
  context.beginPath();
  context.moveTo(start.x, start.y);
  context.lineTo(end.x, end.y);
  context.stroke();
  context.strokeStyle = color;
  context.lineWidth = width;
  context.beginPath();
  context.moveTo(start.x, start.y);
  context.lineTo(end.x, end.y);
  context.stroke();
  context.restore();
}

function drawOutlinedPoint(context, point, color, radius) {
  context.save();
  context.fillStyle = 'rgba(15, 23, 42, 0.9)';
  context.beginPath();
  context.arc(point.x, point.y, radius + 1.5, 0, Math.PI * 2);
  context.fill();
  context.fillStyle = color;
  context.beginPath();
  context.arc(point.x, point.y, radius, 0, Math.PI * 2);
  context.fill();
  context.restore();
}

export function drawInferenceResult(canvas, imageElement, result, hoveredIndex = null) {
  if (!canvas || !imageElement || !result) return;
  const context = canvas.getContext('2d');
  canvas.width = imageElement.naturalWidth;
  canvas.height = imageElement.naturalHeight;
  context.clearRect(0, 0, canvas.width, canvas.height);
  const strokeWidth = getStrokeWidth(canvas);
  const boxStrokeWidth = getBoxStrokeWidth(canvas);

  if (result.task === 'classify') {
    if (!result.predictions?.length) return;
    const topPrediction = `${result.topPrediction.className} • ${toPercent(result.topPrediction.confidence)}`;
    const pill = drawPill(context, 20, 20, topPrediction);
    const barX = 20;
    let barY = 20 + pill.height + 14;
    result.predictions.slice(0, 5).forEach((prediction) => {
      context.save();
      context.fillStyle = 'rgba(17, 24, 39, 0.86)';
      context.beginPath();
      context.roundRect(barX, barY, 260, 28, 999);
      context.fill();
      context.fillStyle = '#ffffff';
      context.font = '12px sans-serif';
      context.fillText(prediction.className, barX + 10, barY + 18);
      context.fillStyle = 'rgba(255,255,255,0.18)';
      context.beginPath();
      context.roundRect(barX + 150, barY + 9, 96, 8, 999);
      context.fill();
      context.fillStyle = 'rgba(198, 93, 60, 1)';
      context.beginPath();
      context.roundRect(barX + 150, barY + 9, Math.max(6, 96 * prediction.confidence), 8, 999);
      context.fill();
      context.fillStyle = '#ffffff';
      context.textAlign = 'right';
      context.fillText(toPercent(prediction.confidence), barX + 255, barY + 18);
      context.restore();
      barY += 34;
    });
    return;
  }

  if (!result.detections?.length) return;

  result.detections.forEach((detection, index) => {
    const color = getColorForIndex(index, hoveredIndex === index ? 0.9 : 0.65);
    const opaqueColor = getColorForIndex(index, 1);
    const [r, g, b] = parseColor(opaqueColor);

    if (result.task === 'segment' && detection.mask) {
      drawMask(context, detection.mask, [r, g, b], hoveredIndex === index ? 160 : 110);
      context.strokeStyle = opaqueColor;
      context.lineWidth = hoveredIndex === index ? boxStrokeWidth + 2 : boxStrokeWidth;
      context.strokeRect(detection.bbox.x1, detection.bbox.y1, detection.bbox.width, detection.bbox.height);
      drawLabel(context, detection.bbox.x1, detection.bbox.y1, `${detection.className} ${toPercent(detection.confidence)}`, opaqueColor);
      return;
    }

    if (result.task === 'pose') {
      context.strokeStyle = opaqueColor;
      context.lineWidth = hoveredIndex === index ? boxStrokeWidth + 2 : boxStrokeWidth;
      context.strokeRect(detection.bbox.x1, detection.bbox.y1, detection.bbox.width, detection.bbox.height);
      detection.skeleton.forEach(([startIndex, endIndex]) => {
        const start = detection.keypoints[startIndex];
        const end = detection.keypoints[endIndex];
        if (!start || !end || start.confidence < 0.35 || end.confidence < 0.35) return;
        drawOutlinedLine(context, start, end, opaqueColor, strokeWidth);
      });
      detection.keypoints.forEach((point) => {
        if (point.confidence < 0.35) return;
        drawOutlinedPoint(context, point, opaqueColor, hoveredIndex === index ? 5 : 4);
      });
      drawLabel(context, detection.bbox.x1, detection.bbox.y1, `${detection.className} ${toPercent(detection.confidence)}`, opaqueColor);
      return;
    }

    if (result.task === 'obb') {
      context.strokeStyle = opaqueColor;
      context.lineWidth = hoveredIndex === index ? boxStrokeWidth + 2 : boxStrokeWidth;
      context.beginPath();
      detection.corners.forEach((corner, cornerIndex) => {
        if (cornerIndex === 0) context.moveTo(corner.x, corner.y);
        else context.lineTo(corner.x, corner.y);
      });
      context.closePath();
      context.stroke();
      const labelAnchor = [...detection.corners].sort((left, right) => left.y - right.y)[0];
      drawLabel(context, Math.max(0, labelAnchor.x), Math.max(24, labelAnchor.y), `${detection.className} ${toPercent(detection.confidence)}`, opaqueColor);
      return;
    }

    context.strokeStyle = opaqueColor;
    context.lineWidth = hoveredIndex === index ? boxStrokeWidth + 2 : boxStrokeWidth;
    context.strokeRect(detection.bbox.x1, detection.bbox.y1, detection.bbox.width, detection.bbox.height);
    drawLabel(context, detection.bbox.x1, detection.bbox.y1, `${detection.className} ${toPercent(detection.confidence)}`, opaqueColor);
  });
}
