import { parseClassificationOutputs, parseDetectionOutputs, parseObbOutputs, parsePoseOutputs, parseSegmentationOutputs } from './parsers';
import { prepareImageTensor } from './preprocess';
import { getOrCreateSession } from './session';

const PARSERS = {
  detect: parseDetectionOutputs,
  segment: parseSegmentationOutputs,
  pose: parsePoseOutputs,
  classify: parseClassificationOutputs,
  obb: parseObbOutputs,
};

export async function runInference({ task, image, modelConfig, scoreThreshold = 0.25, iouThreshold = 0.45 }) {
  const preprocessMode = task === 'classify' ? 'resize' : 'letterbox';
  const { tensor, meta } = prepareImageTensor(image, modelConfig.inputSize, preprocessMode);
  const session = await getOrCreateSession(modelConfig);
  const inputName = session.inputNames[0];
  const start = performance.now();
  const outputs = await session.run({ [inputName]: tensor });
  const elapsed = performance.now() - start;

  const parser = PARSERS[task];
  if (!parser) {
    throw new Error(`No parser registered for task: ${task}`);
  }

  return parser(outputs, meta, modelConfig, {
    scoreThreshold,
    iouThreshold,
    inferenceTimeMs: elapsed,
  });
}
