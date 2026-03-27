import * as ort from 'onnxruntime-web';

function createCanvas(width, height) {
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  return canvas;
}

export function loadImageElement(sourceUrl) {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = reject;
    image.src = sourceUrl;
  });
}

export function prepareImageTensor(image, inputSize, mode = 'letterbox') {
  const canvas = createCanvas(inputSize, inputSize);
  const context = canvas.getContext('2d', { willReadFrequently: true });

  let drawWidth = inputSize;
  let drawHeight = inputSize;
  let padX = 0;
  let padY = 0;
  let scaleX = inputSize / image.naturalWidth;
  let scaleY = inputSize / image.naturalHeight;
  let scale = Math.min(scaleX, scaleY);

  context.fillStyle = '#000000';
  context.fillRect(0, 0, inputSize, inputSize);

  if (mode === 'letterbox') {
    drawWidth = Math.round(image.naturalWidth * scale);
    drawHeight = Math.round(image.naturalHeight * scale);
    padX = Math.floor((inputSize - drawWidth) / 2);
    padY = Math.floor((inputSize - drawHeight) / 2);
    context.drawImage(image, padX, padY, drawWidth, drawHeight);
  } else {
    scale = inputSize / image.naturalWidth;
    scaleX = inputSize / image.naturalWidth;
    scaleY = inputSize / image.naturalHeight;
    context.drawImage(image, 0, 0, inputSize, inputSize);
  }

  const imageData = context.getImageData(0, 0, inputSize, inputSize).data;
  const floatData = new Float32Array(3 * inputSize * inputSize);

  for (let index = 0; index < inputSize * inputSize; index += 1) {
    const pixelOffset = index * 4;
    floatData[index] = imageData[pixelOffset] / 255;
    floatData[inputSize * inputSize + index] = imageData[pixelOffset + 1] / 255;
    floatData[inputSize * inputSize * 2 + index] = imageData[pixelOffset + 2] / 255;
  }

  return {
    tensor: new ort.Tensor('float32', floatData, [1, 3, inputSize, inputSize]),
    meta: {
      padX,
      padY,
      scale,
      scaleX,
      scaleY,
      inputWidth: inputSize,
      inputHeight: inputSize,
      originalWidth: image.naturalWidth,
      originalHeight: image.naturalHeight,
      preprocessMode: mode,
    },
  };
}
