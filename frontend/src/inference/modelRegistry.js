import { COCO_CLASSES, COCO_SKELETON, DOTA_CLASSES, POSE_KEYPOINT_NAMES } from './labels';

export const TASK_OPTIONS = [
  { id: 'detect', label: 'Detection', description: 'Current backend detection flow' },
  { id: 'segment', label: 'Instance Segmentation', description: 'Browser ONNX masks and inspector overlays' },
  { id: 'pose', label: 'Pose Estimation', description: 'Browser ONNX skeleton rendering' },
  { id: 'classify', label: 'Image Classification', description: 'Browser ONNX ranked predictions' },
  { id: 'obb', label: 'OBB', description: 'Browser ONNX rotated boxes for aerial imagery' },
];

export const TASK_SAMPLE_IMAGES = {
  segment: [
    { id: 'bus', label: 'Bus Sample', url: 'https://ultralytics.com/images/bus.jpg' },
  ],
  pose: [
    { id: 'zidane', label: 'Pose Sample', url: 'https://ultralytics.com/images/zidane.jpg' },
  ],
  classify: [
    { id: 'bus', label: 'Bus Sample', url: 'https://ultralytics.com/images/bus.jpg' },
  ],
  obb: [
    { id: 'boats', label: 'Boats Sample', url: 'https://ultralytics.com/images/boats.jpg' },
  ],
};

const browserModels = [
  {
    id: 'yolo11n-seg',
    task: 'segment',
    family: 'yolo11',
    displayName: 'YOLO11n-seg',
    inputSize: 640,
    classNames: COCO_CLASSES,
    defaultUrl: 'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-seg.onnx',
    cdnUrl: 'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-seg.onnx',
  },
  {
    id: 'yolo26n-seg',
    task: 'segment',
    family: 'yolo26',
    displayName: 'YOLO26n-seg',
    inputSize: 640,
    classNames: COCO_CLASSES,
    defaultUrl: 'https://huggingface.co/zwh20081/yolo26-onnx/resolve/main/yolo26n-seg.onnx?download=true',
    cdnUrl: 'https://huggingface.co/zwh20081/yolo26-onnx/resolve/main/yolo26n-seg.onnx?download=true',
  },
  {
    id: 'yolo11n-pose',
    task: 'pose',
    family: 'yolo11',
    displayName: 'YOLO11n-pose',
    inputSize: 640,
    classNames: ['person'],
    keypointNames: POSE_KEYPOINT_NAMES,
    keypointSkeleton: COCO_SKELETON,
    defaultUrl: 'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-pose.onnx',
    cdnUrl: 'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-pose.onnx',
  },
  {
    id: 'yolo26n-pose',
    task: 'pose',
    family: 'yolo26',
    displayName: 'YOLO26n-pose',
    inputSize: 640,
    classNames: ['person'],
    keypointNames: POSE_KEYPOINT_NAMES,
    keypointSkeleton: COCO_SKELETON,
    defaultUrl: 'https://huggingface.co/zwh20081/yolo26-onnx/resolve/main/yolo26n-pose.onnx?download=true',
    cdnUrl: 'https://huggingface.co/zwh20081/yolo26-onnx/resolve/main/yolo26n-pose.onnx?download=true',
  },
  {
    id: 'yolo11n-cls',
    task: 'classify',
    family: 'yolo11',
    displayName: 'YOLO11n-cls',
    inputSize: 224,
    classNames: [],
    labelsUrl: 'https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt',
    defaultUrl: 'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-cls.onnx',
    cdnUrl: 'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-cls.onnx',
  },
  {
    id: 'yolo26n-cls',
    task: 'classify',
    family: 'yolo26',
    displayName: 'YOLO26n-cls',
    inputSize: 640,
    classNames: [],
    labelsUrl: 'https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt',
    defaultUrl: 'https://huggingface.co/zwh20081/yolo26-onnx/resolve/main/yolo26n-cls.onnx?download=true',
    cdnUrl: 'https://huggingface.co/zwh20081/yolo26-onnx/resolve/main/yolo26n-cls.onnx?download=true',
  },
  {
    id: 'yolo11n-obb',
    task: 'obb',
    family: 'yolo11',
    displayName: 'YOLO11n-obb',
    inputSize: 1024,
    classNames: DOTA_CLASSES,
    defaultUrl: 'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-obb.onnx',
    cdnUrl: 'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-obb.onnx',
  },
  {
    id: 'yolo26n-obb',
    task: 'obb',
    family: 'yolo26',
    displayName: 'YOLO26n-obb',
    inputSize: 1024,
    classNames: DOTA_CLASSES,
    available: false,
    unavailableReason: 'No official YOLO26 ONNX CDN asset was found in Ultralytics assets releases.',
  },
];

export const BROWSER_MODEL_REGISTRY = browserModels;

export function getModelsForTask(task) {
  return browserModels.filter((model) => model.task === task);
}

export function getModelById(modelId) {
  return browserModels.find((model) => model.id === modelId) ?? null;
}
