import { getModelsForTask, TASK_OPTIONS } from './modelRegistry';

export const CUSTOM_MODEL_PREFIX = 'custom:';

export function getCustomModelSelection(task) {
  return `${CUSTOM_MODEL_PREFIX}${task}`;
}

export function isCustomModelSelection(value) {
  return typeof value === 'string' && value.startsWith(CUSTOM_MODEL_PREFIX);
}

export function getInitialTaskModelSelections() {
  const selections = {};
  TASK_OPTIONS.forEach((option) => {
    if (option.id === 'detect') {
      selections[option.id] = 'active-backend';
      return;
    }
    selections[option.id] = getModelsForTask(option.id).find((model) => model.available !== false)?.id ?? getCustomModelSelection(option.id);
  });
  return selections;
}

export function makeBrowserModelConfig(baseModel, assetState) {
  if (!baseModel && !assetState) return null;
  return {
    ...(baseModel ?? {}),
    ...(assetState ?? {}),
    displayName: assetState?.modelName ?? baseModel?.displayName ?? 'Custom Model',
    modelUrl: assetState?.modelUrl ?? baseModel?.defaultUrl,
    sessionKey: assetState?.sessionKey ?? baseModel?.id,
    classNames: assetState?.labels?.length ? assetState.labels : (baseModel?.classNames ?? []),
  };
}
