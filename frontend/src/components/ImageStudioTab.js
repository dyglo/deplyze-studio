import React, { useEffect, useMemo, useRef, useState } from 'react';
import axios from 'axios';
import { Download, Eye, Layers, RefreshCcw, Upload, Zap } from 'lucide-react';
import { toast } from 'sonner';
import { Badge } from './ui/badge';
import { Button } from './ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card';
import { Input } from './ui/input';
import { Label } from './ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from './ui/select';
import StudioInspector from './StudioInspector';
import { getModelById, getModelsForTask, TASK_OPTIONS, TASK_SAMPLE_IMAGES } from '../inference/modelRegistry';
import { createFallbackLabels } from '../inference/labels';
import { runInference } from '../inference';
import { loadImageElement } from '../inference/preprocess';
import { drawInferenceResult } from '../inference/render';
import { ensureBrowserModelDownloaded, fetchTextCached } from '../inference/session';
import { CUSTOM_MODEL_PREFIX, getCustomModelSelection, isCustomModelSelection, makeBrowserModelConfig } from '../inference/studio';

function base64ToObjectUrl(base64Data, mimeType = 'application/octet-stream') {
  const binary = window.atob(base64Data);
  const bytes = new Uint8Array(binary.length);
  for (let index = 0; index < binary.length; index += 1) {
    bytes[index] = binary.charCodeAt(index);
  }
  return URL.createObjectURL(new Blob([bytes], { type: mimeType }));
}

function fileToText(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onerror = () => reject(new Error('Failed to read label file'));
    reader.onload = () => resolve(String(reader.result ?? ''));
    reader.readAsText(file);
  });
}

export default function ImageStudioTab({
  apiBaseUrl,
  detectionModelName,
  modelInfo,
  studioTask,
  onStudioTaskChange,
  taskModelSelections,
  onTaskModelChange,
  browserAssets,
  setBrowserAssets,
  customLabelsByTask,
  setCustomLabelsByTask,
  switchModel,
  loadModelInfo,
}) {
  const fileInputRef = useRef(null);
  const customModelInputRef = useRef(null);
  const labelsInputRef = useRef(null);
  const previewImageRef = useRef(null);
  const overlayCanvasRef = useRef(null);

  const [imageFile, setImageFile] = useState(null);
  const [imageUrl, setImageUrl] = useState(null);
  const [annotatedImage, setAnnotatedImage] = useState(null);
  const [result, setResult] = useState(null);
  const [hoveredIndex, setHoveredIndex] = useState(null);
  const [isRunning, setIsRunning] = useState(false);
  const [scoreThreshold, setScoreThreshold] = useState(0.25);
  const [iouThreshold, setIouThreshold] = useState(0.45);
  const [pendingAutoRun, setPendingAutoRun] = useState(false);
  const [uploadingCustomModel, setUploadingCustomModel] = useState(false);

  const availableModels = useMemo(() => getModelsForTask(studioTask), [studioTask]);
  const selectedModelId = taskModelSelections[studioTask] ?? (studioTask === 'detect' ? detectionModelName : '');
  const selectedBrowserModel = useMemo(
    () => (!isCustomModelSelection(selectedModelId) ? getModelById(selectedModelId) : null),
    [selectedModelId],
  );
  const activeBrowserAsset = browserAssets[selectedModelId];
  const resolvedBrowserModel = useMemo(
    () => makeBrowserModelConfig(selectedBrowserModel, activeBrowserAsset),
    [selectedBrowserModel, activeBrowserAsset],
  );
  const sampleImages = TASK_SAMPLE_IMAGES[studioTask] ?? [];
  const labelsLoaded = activeBrowserAsset?.labels?.length ?? selectedBrowserModel?.classNames?.length ?? customLabelsByTask[studioTask]?.length ?? 0;
  const modelStatus = studioTask === 'detect' ? 'ready' : activeBrowserAsset?.status ?? 'idle';
  const detectModelOptions = modelInfo?.available_models ? Object.keys(modelInfo.available_models) : [];

  useEffect(() => () => {
    if (imageUrl) URL.revokeObjectURL(imageUrl);
  }, [imageUrl]);

  useEffect(() => {
    if (!previewImageRef.current || !overlayCanvasRef.current || !result || studioTask === 'detect') return;
    const redraw = () => drawInferenceResult(overlayCanvasRef.current, previewImageRef.current, result, hoveredIndex);
    if (previewImageRef.current.complete) redraw();
    else previewImageRef.current.onload = redraw;
  }, [result, studioTask, hoveredIndex]);

  useEffect(() => {
    if (studioTask === 'detect' || !pendingAutoRun || !imageFile || modelStatus !== 'ready' || isRunning) {
      return;
    }
    setPendingAutoRun(false);
    runCurrentTask(imageFile);
  }, [studioTask, pendingAutoRun, imageFile, modelStatus, isRunning]);

  useEffect(() => {
    let isMounted = true;

    async function warmModel() {
      if (studioTask === 'detect' || isCustomModelSelection(selectedModelId)) {
        return;
      }
      if (!selectedBrowserModel) return;
      if (selectedBrowserModel.available === false) {
        setBrowserAssets((current) => ({
          ...current,
          [selectedBrowserModel.id]: {
            status: 'error',
            error: selectedBrowserModel.unavailableReason,
          },
        }));
        return;
      }
      if (activeBrowserAsset?.status === 'ready' || activeBrowserAsset?.status === 'downloading') {
        return;
      }

      setBrowserAssets((current) => ({
        ...current,
        [selectedBrowserModel.id]: {
          ...current[selectedBrowserModel.id],
          status: 'downloading',
          error: null,
        },
      }));

      try {
        const modelUrl = await ensureBrowserModelDownloaded(selectedBrowserModel);
        let labels = activeBrowserAsset?.labels;
        if ((!labels || !labels.length) && selectedBrowserModel.labelsUrl) {
          const labelText = await fetchTextCached(selectedBrowserModel.labelsUrl);
          labels = labelText.split(/\r?\n/).map((line) => line.trim()).filter(Boolean);
        }
        if (!isMounted) return;
        setBrowserAssets((current) => ({
          ...current,
          [selectedBrowserModel.id]: {
            ...current[selectedBrowserModel.id],
            modelUrl,
            sessionKey: `${selectedBrowserModel.id}:cdn`,
            modelName: selectedBrowserModel.displayName,
            labels,
            status: 'ready',
            error: null,
          },
        }));
      } catch (error) {
        if (!isMounted) return;
        setBrowserAssets((current) => ({
          ...current,
          [selectedBrowserModel.id]: {
            ...current[selectedBrowserModel.id],
            status: 'error',
            error: error.message || `Failed to download ${selectedBrowserModel.displayName}`,
          },
        }));
      }
    }

    warmModel();
    return () => {
      isMounted = false;
    };
  }, [studioTask, selectedBrowserModel, selectedModelId, activeBrowserAsset, setBrowserAssets]);

  const clearResultState = () => {
    setResult(null);
    setAnnotatedImage(null);
    setHoveredIndex(null);
  };

  const getRuntimeLabel = () => (studioTask === 'detect' ? 'FastAPI' : 'Browser ONNX');

  const getInspectorTaskLabel = () => TASK_OPTIONS.find((option) => option.id === studioTask)?.label ?? studioTask;

  const getActiveModelLabel = () => {
    if (studioTask === 'detect') return detectionModelName || selectedModelId || 'Backend model';
    return resolvedBrowserModel?.displayName ?? 'Unconfigured';
  };

  const runDetectionTask = async (file) => {
    const formData = new FormData();
    formData.append('file', file);
    const [detectResponse, annotatedResponse] = await Promise.all([
      axios.post(`${apiBaseUrl}/detect/image`, formData, { headers: { 'Content-Type': 'multipart/form-data' } }),
      axios.post(`${apiBaseUrl}/detect/image/annotated`, formData, { headers: { 'Content-Type': 'multipart/form-data' } }),
    ]);

    setResult({
      ...detectResponse.data,
      task: 'detect',
      inferenceTimeMs: (detectResponse.data.inference_time ?? 0) * 1000,
      detections: (detectResponse.data.detections ?? []).map((detection) => ({
        ...detection,
        className: detection.class_name,
      })),
    });
    setAnnotatedImage(`data:image/jpeg;base64,${annotatedResponse.data.annotated_image}`);
    toast.success(`Found ${(detectResponse.data.detections ?? []).length} detections in ${detectResponse.data.inference_time.toFixed(2)}s`);
  };

  const runBrowserTask = async (file) => {
    if (!resolvedBrowserModel?.modelUrl) {
      throw new Error(activeBrowserAsset?.error || `Model for ${studioTask} is not ready yet.`);
    }
    const sourceUrl = URL.createObjectURL(file);
    const image = await loadImageElement(sourceUrl);
    try {
      const browserResult = await runInference({
        task: studioTask,
        image,
        modelConfig: resolvedBrowserModel,
        scoreThreshold,
        iouThreshold,
      });
      setResult(browserResult);
      toast.success(`${getInspectorTaskLabel()} completed in ${browserResult.inferenceTimeMs.toFixed(0)}ms`);
    } finally {
      URL.revokeObjectURL(sourceUrl);
    }
  };

  const runCurrentTask = async (overrideFile = null) => {
    const fileToUse = overrideFile ?? imageFile;
    if (!fileToUse) {
      toast.error('Choose or load a sample image first.');
      return;
    }
    setIsRunning(true);
    clearResultState();
    try {
      if (studioTask === 'detect') await runDetectionTask(fileToUse);
      else await runBrowserTask(fileToUse);
    } catch (error) {
      console.error(`Error running ${studioTask}:`, error);
      toast.error(error.message || `Failed to run ${studioTask}`);
    } finally {
      setIsRunning(false);
    }
  };

  const handleImageUpload = (event) => {
    const file = event.target.files?.[0];
    if (!file) return;
    if (imageUrl) URL.revokeObjectURL(imageUrl);
    const nextUrl = URL.createObjectURL(file);
    setImageFile(file);
    setImageUrl(nextUrl);
    clearResultState();
    if (studioTask === 'detect' || modelStatus === 'ready') {
      setTimeout(() => {
        runCurrentTask(file);
      }, 0);
    } else {
      setPendingAutoRun(true);
    }
  };

  const handleSampleSelect = async (sample) => {
    try {
      const response = await fetch(sample.url);
      if (!response.ok) throw new Error(`Failed to fetch sample image: ${response.status}`);
      const blob = await response.blob();
      const file = new File([blob], `${sample.id}.jpg`, { type: blob.type || 'image/jpeg' });
      if (imageUrl) URL.revokeObjectURL(imageUrl);
      const nextUrl = URL.createObjectURL(file);
      setImageFile(file);
      setImageUrl(nextUrl);
      clearResultState();
      if (studioTask === 'detect' || modelStatus === 'ready') {
        setTimeout(() => {
          runCurrentTask(file);
        }, 0);
      } else {
        setPendingAutoRun(true);
      }
      toast.success(`Loaded sample image: ${sample.label}`);
    } catch (error) {
      toast.error(error.message || 'Failed to load sample image');
    }
  };

  const handleDownload = async () => {
    if (!result) return;
    if (studioTask === 'detect' && result.id) {
      try {
        const response = await axios.get(`${apiBaseUrl}/download/image/${result.id}`, { responseType: 'blob' });
        const url = window.URL.createObjectURL(new Blob([response.data]));
        const link = document.createElement('a');
        link.href = url;
        link.download = `visionflow_annotated_${result.id}.jpg`;
        link.click();
        window.URL.revokeObjectURL(url);
      } catch (error) {
        toast.error('Failed to download annotated image');
      }
      return;
    }

    if (!previewImageRef.current) return;
    const exportCanvas = document.createElement('canvas');
    exportCanvas.width = previewImageRef.current.naturalWidth;
    exportCanvas.height = previewImageRef.current.naturalHeight;
    const context = exportCanvas.getContext('2d');
    context.drawImage(previewImageRef.current, 0, 0);
    if (overlayCanvasRef.current) context.drawImage(overlayCanvasRef.current, 0, 0);
    const link = document.createElement('a');
    link.href = exportCanvas.toDataURL('image/png');
    link.download = `${studioTask}_${Date.now()}.png`;
    link.click();
  };

  const handleTaskChange = (nextTask) => {
    onStudioTaskChange(nextTask);
    clearResultState();
  };

  const handleModelChange = async (value) => {
    clearResultState();
    onTaskModelChange(studioTask, value);
    if (studioTask === 'detect' && !isCustomModelSelection(value)) {
      await switchModel(value);
      await loadModelInfo();
    }
  };

  const handleCustomModelUpload = async (event) => {
    const file = event.target.files?.[0];
    if (!file) return;
    setUploadingCustomModel(true);
    try {
      if (studioTask === 'detect') {
        const formData = new FormData();
        formData.append('file', file);
        formData.append('model_name', file.name.split('.')[0]);
        const response = await axios.post(`${apiBaseUrl}/model/upload`, formData, {
          headers: { 'Content-Type': 'multipart/form-data' },
        });
        await loadModelInfo();
        await switchModel(response.data.model_name);
        onTaskModelChange('detect', response.data.model_name);
        toast.success(`Custom backend model ready: ${response.data.model_name}`);
        return;
      }

      const selectionKey = getCustomModelSelection(studioTask);
      let modelUrl = null;
      let sourceType = 'uploaded-onnx';
      if (file.name.toLowerCase().endsWith('.onnx')) {
        modelUrl = URL.createObjectURL(file);
      } else if (file.name.toLowerCase().endsWith('.pt')) {
        const formData = new FormData();
        formData.append('file', file);
        formData.append('task', studioTask);
        const response = await axios.post(`${apiBaseUrl}/model/convert`, formData, {
          headers: { 'Content-Type': 'multipart/form-data' },
          timeout: 300000,
        });
        modelUrl = base64ToObjectUrl(response.data.onnx_model, 'application/octet-stream');
        sourceType = 'converted-pt';
      } else {
        throw new Error('Custom models must be .onnx or .pt');
      }

      const nextLabels = customLabelsByTask[studioTask] ?? createFallbackLabels(1);
      setBrowserAssets((current) => ({
        ...current,
        [selectionKey]: {
          ...(current[selectionKey] ?? {}),
          id: selectionKey,
          task: studioTask,
          modelName: file.name,
          displayName: file.name,
          modelUrl,
          sessionKey: `${selectionKey}:${Date.now()}`,
          labels: nextLabels,
          sourceType,
          status: 'ready',
          error: null,
          inputSize: studioTask === 'classify' ? 224 : resolvedBrowserModel?.inputSize ?? 640,
          family: 'custom',
          keypointNames: studioTask === 'pose' ? resolvedBrowserModel?.keypointNames : undefined,
          keypointSkeleton: studioTask === 'pose' ? resolvedBrowserModel?.keypointSkeleton : undefined,
        },
      }));
      onTaskModelChange(studioTask, selectionKey);
      toast.success(`Custom ${studioTask} model loaded for this session`);
    } catch (error) {
      console.error('Error loading custom model:', error);
      toast.error(error.response?.data?.detail || error.message || 'Failed to load custom model');
    } finally {
      setUploadingCustomModel(false);
      event.target.value = '';
    }
  };

  const handleLabelsUpload = async (event) => {
    const file = event.target.files?.[0];
    if (!file) return;
    try {
      const text = await fileToText(file);
      const labels = text.split(/\r?\n/).map((line) => line.trim()).filter(Boolean);
      if (!labels.length) throw new Error('No labels found in file');
      setCustomLabelsByTask((current) => ({ ...current, [studioTask]: labels }));
      if (isCustomModelSelection(selectedModelId)) {
        setBrowserAssets((current) => ({
          ...current,
          [selectedModelId]: {
            ...(current[selectedModelId] ?? {}),
            labels,
          },
        }));
      }
      toast.success(`Loaded ${labels.length} labels for ${getInspectorTaskLabel()}`);
    } catch (error) {
      toast.error(error.message || 'Failed to load labels');
    } finally {
      event.target.value = '';
    }
  };

  const footerNote = studioTask !== 'detect' ? (
    <>
      <div>Registered browser models: {availableModels.length}</div>
      <div>Status: {modelStatus}</div>
      <div>Labels: {labelsLoaded || 'none'}</div>
      <div>Source: {activeBrowserAsset?.sourceType ?? (selectedBrowserModel?.cdnUrl ? 'CDN auto-download' : 'Unavailable')}</div>
    </>
  ) : (
    <>
      <div>Backend model registry: {detectModelOptions.length}</div>
      <div>Active detector: {detectionModelName}</div>
    </>
  );

  return (
    <div className="space-y-6">
      <div className="grid lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2 space-y-6 flex flex-col h-full min-h-[500px]">
          <Card className="bg-card border-border shadow-sm">
            <CardHeader className="border-b bg-muted/10">
              <CardTitle className="flex items-center gap-2 text-base"><Layers className="w-4 h-4 text-primary" />Single-Image Studio</CardTitle>
              <CardDescription>Detection stays on the current backend path. Phase 2 browser tasks keep using ONNX runtime, including session-scoped custom models.</CardDescription>
            </CardHeader>
            <CardContent className="p-4 space-y-4">
              <div className="grid md:grid-cols-2 xl:grid-cols-4 gap-3">
                <div className="space-y-2">
                  <div className="text-[11px] font-bold uppercase tracking-widest text-muted-foreground">Task</div>
                  <Select value={studioTask} onValueChange={handleTaskChange}>
                    <SelectTrigger><SelectValue /></SelectTrigger>
                    <SelectContent>
                      {TASK_OPTIONS.map((option) => <SelectItem key={option.id} value={option.id}>{option.label}</SelectItem>)}
                    </SelectContent>
                  </Select>
                </div>

                <div className="space-y-2">
                  <div className="text-[11px] font-bold uppercase tracking-widest text-muted-foreground">Model</div>
                  <Select value={selectedModelId} onValueChange={handleModelChange}>
                    <SelectTrigger><SelectValue /></SelectTrigger>
                    <SelectContent>
                      {studioTask === 'detect' ? detectModelOptions.map((modelName) => (
                        <SelectItem key={modelName} value={modelName}>{modelName}</SelectItem>
                      )) : availableModels.map((model) => (
                        <SelectItem key={model.id} value={model.id} disabled={model.available === false}>
                          {model.displayName}{model.available === false ? ' (Unavailable)' : ''}
                        </SelectItem>
                      ))}
                      <SelectItem value={getCustomModelSelection(studioTask)}>Custom Model</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                <div className="space-y-2">
                  <div className="text-[11px] font-bold uppercase tracking-widest text-muted-foreground">Score</div>
                  <input type="range" min="0.1" max="0.9" step="0.05" value={scoreThreshold} onChange={(event) => setScoreThreshold(Number(event.target.value))} className="w-full" />
                  <div className="text-xs text-muted-foreground">{(scoreThreshold * 100).toFixed(0)}%</div>
                </div>

                <div className="space-y-2">
                  <div className="text-[11px] font-bold uppercase tracking-widest text-muted-foreground">IoU</div>
                  <input type="range" min="0.2" max="0.7" step="0.05" value={iouThreshold} onChange={(event) => setIouThreshold(Number(event.target.value))} className="w-full" disabled={studioTask === 'classify'} />
                  <div className="text-xs text-muted-foreground">{(iouThreshold * 100).toFixed(0)}%</div>
                </div>
              </div>

              <div className="flex flex-wrap items-center gap-3 rounded-lg border bg-muted/10 p-3">
                <Badge variant="outline" className="uppercase tracking-widest">{studioTask === 'detect' ? 'BACKEND' : (resolvedBrowserModel?.family?.toUpperCase() ?? 'CUSTOM')}</Badge>
                <Badge variant="outline">{getActiveModelLabel()}</Badge>
                <Badge variant={modelStatus === 'ready' ? 'default' : 'outline'}>
                  {uploadingCustomModel ? 'Preparing Model' : modelStatus === 'downloading' ? 'Downloading Model' : modelStatus === 'ready' ? 'Model Ready' : modelStatus === 'error' ? 'Model Error' : 'Waiting'}
                </Badge>
                <Badge variant="outline">{labelsLoaded || createFallbackLabels(1).length} labels</Badge>
                {activeBrowserAsset?.error && <span className="text-xs text-red-500">{activeBrowserAsset.error}</span>}
              </div>

              {isCustomModelSelection(selectedModelId) && (
                <div className="grid md:grid-cols-2 gap-4 rounded-xl border bg-card/60 p-4">
                  <div className="space-y-3">
                    <Label className="text-[11px] font-bold uppercase tracking-widest text-muted-foreground">Custom Model</Label>
                    <Button type="button" variant="outline" className="w-full justify-start" onClick={() => customModelInputRef.current?.click()}>
                      <Upload className="w-4 h-4 mr-2" />
                      {studioTask === 'detect' ? 'Upload .pt or .onnx to backend' : 'Upload .onnx or convert .pt'}
                    </Button>
                    <p className="text-xs text-muted-foreground">Custom models are cached for this browser session and reused across tabs for the same task.</p>
                    <input ref={customModelInputRef} type="file" accept=".onnx,.pt" onChange={handleCustomModelUpload} className="hidden" />
                  </div>
                  <div className="space-y-3">
                    <Label className="text-[11px] font-bold uppercase tracking-widest text-muted-foreground">Custom Labels</Label>
                    <Button type="button" variant="outline" className="w-full justify-start" onClick={() => labelsInputRef.current?.click()}>
                      <Download className="w-4 h-4 mr-2" />
                      Upload label file (.txt)
                    </Button>
                    <Input value={(customLabelsByTask[studioTask] ?? []).join(', ')} readOnly className="text-xs" placeholder="One class per line" />
                    <input ref={labelsInputRef} type="file" accept=".txt" onChange={handleLabelsUpload} className="hidden" />
                  </div>
                </div>
              )}

              {studioTask !== 'detect' && sampleImages.length > 0 && (
                <div className="flex flex-wrap gap-2">
                  {sampleImages.map((sample) => (
                    <Button key={sample.id} type="button" variant="outline" size="sm" onClick={() => handleSampleSelect(sample)}>
                      {sample.label}
                    </Button>
                  ))}
                </div>
              )}
            </CardContent>
          </Card>

          <Card className="bg-card border-border shadow-sm flex-1 flex flex-col">
            <CardHeader className="flex flex-row items-center justify-between py-3 px-4 border-b shrink-0">
              <div className="space-y-1">
                <CardTitle className="text-lg flex items-center gap-2"><Eye className="w-4 h-4 text-primary" />Image Analysis</CardTitle>
                <CardDescription>{studioTask === 'detect' ? `Backend detection using active model ${detectionModelName || 'unknown'}` : `${getActiveModelLabel()} on uploaded image`}</CardDescription>
              </div>
              <div className="flex items-center gap-2">
                <Button variant="secondary" size="sm" onClick={() => fileInputRef.current?.click()}><Upload className="w-4 h-4 mr-2" />Upload Image</Button>
                <Button variant="default" size="sm" disabled={!imageFile || isRunning || (studioTask !== 'detect' && modelStatus !== 'ready')} onClick={() => { setPendingAutoRun(false); runCurrentTask(); }}>
                  <Zap className="w-4 h-4 mr-2" />
                  Run Inference
                </Button>
                <Button variant="outline" size="sm" disabled={!imageFile || isRunning} onClick={() => { setPendingAutoRun(false); clearResultState(); setImageFile(null); if (imageUrl) URL.revokeObjectURL(imageUrl); setImageUrl(null); }}>
                  <RefreshCcw className="w-4 h-4 mr-2" />
                  Reset
                </Button>
              </div>
            </CardHeader>
            <CardContent className="p-0 flex items-center justify-center bg-muted/10 flex-1 overflow-hidden relative group/canvas min-h-[420px]">
              {imageUrl ? (
                <div className="relative w-full h-full flex items-center justify-center">
                  {studioTask === 'detect' && annotatedImage ? (
                    <img src={annotatedImage} alt="Annotated result" className="max-w-full max-h-full object-contain" />
                  ) : (
                    <>
                      <img ref={previewImageRef} src={imageUrl} alt="Uploaded preview" className="max-w-full max-h-full object-contain" crossOrigin="anonymous" />
                      {studioTask !== 'detect' && <canvas ref={overlayCanvasRef} className="absolute inset-0 w-full h-full pointer-events-none" />}
                    </>
                  )}
                  {isRunning && (
                    <div className="absolute inset-0 bg-background/65 backdrop-blur-sm flex flex-col items-center justify-center gap-3">
                      <div className="animate-spin rounded-full w-10 h-10 border-2 border-primary border-t-transparent"></div>
                      <div className="text-sm font-medium">Running {getInspectorTaskLabel()}...</div>
                    </div>
                  )}
                </div>
              ) : (
                <div className="w-full h-full min-h-[420px] border-2 border-dashed border-border rounded-lg m-6 text-center hover:border-primary transition-colors cursor-pointer flex flex-col items-center justify-center" onClick={() => fileInputRef.current?.click()}>
                  <Upload className="w-12 h-12 mx-auto text-muted-foreground mb-3" />
                  <p className="text-secondary-foreground font-medium">Click to upload an image</p>
                  <p className="text-sm text-muted-foreground">PNG, JPG, GIF up to 10MB</p>
                </div>
              )}

              <input type="file" ref={fileInputRef} onChange={handleImageUpload} accept="image/*" className="hidden" />
            </CardContent>
          </Card>
        </div>

        <StudioInspector
          task={getInspectorTaskLabel()}
          runtimeLabel={getRuntimeLabel()}
          modelLabel={getActiveModelLabel()}
          result={result}
          hoveredIndex={hoveredIndex}
          setHoveredIndex={setHoveredIndex}
          isRunning={isRunning}
          footerNote={footerNote}
          onDownload={handleDownload}
          disableDownload={!result}
        />
      </div>
    </div>
  );
}
