import React, { useEffect, useMemo, useRef, useState } from 'react';
import axios from 'axios';
import {
  ArrowLeft,
  BarChart3,
  BookOpen,
  Database,
  ExternalLink,
  Eye,
  EyeOff,
  Grid2X2,
  Image as ImageIcon,
  KeyRound,
  LayoutGrid,
  List,
  Search,
  Sparkles,
  Target,
  X,
  ChevronDown,
  ChevronUp,
} from 'lucide-react';
import { toast } from 'sonner';
import { Badge } from './ui/badge';
import { Button } from './ui/button';
import { Card, CardContent } from './ui/card';
import { Dialog, DialogContent, DialogHeader, DialogTitle } from './ui/dialog';
import { Input } from './ui/input';
import { runInference } from '../inference';
import { loadImageElement } from '../inference/preprocess';
import { getModelById, TASK_OPTIONS } from '../inference/modelRegistry';
import { isCustomModelSelection, makeBrowserModelConfig } from '../inference/studio';

const STORAGE_KEY = 'deplyze:roboflow-api-key';
const DATASET_SECTIONS = [
  { id: 'overview', label: 'Overview', icon: LayoutGrid },
  { id: 'images', label: 'Images', icon: ImageIcon },
  { id: 'dataset', label: 'Dataset', icon: Database },
  { id: 'analytics', label: 'Analytics', icon: BarChart3 },
  { id: 'model', label: 'Model', icon: Target },
  { id: 'api-docs', label: 'API Docs', icon: BookOpen },
];

function clamp(value, min, max) {
  return Math.min(Math.max(value, min), max);
}

function formatCount(value, suffix) {
  const numeric = Number(value ?? 0);
  return `${numeric.toLocaleString()} ${suffix}`;
}

function base64ToDataUrl(base64, mime = 'image/jpeg') {
  return `data:${mime};base64,${base64}`;
}

function inferSampleSplit(sample) {
  const path = `${sample?.path ?? ''}`.toLowerCase();
  if (path.includes('/train/')) return 'train';
  if (path.includes('/valid/')) return 'valid';
  if (path.includes('/test/')) return 'test';
  return 'all';
}

function getSampleSource(sample) {
  if (!sample) return '';
  return sample.image_url || (sample.image_base64 ? base64ToDataUrl(sample.image_base64, sample.mime_type) : '');
}

function getResultCount(result) {
  if (!result) return 0;
  if (result.task === 'classify') {
    return result.predictions?.length ?? 0;
  }
  return result.detections?.length ?? 0;
}

function normalizeDetectResult(payload) {
  return {
    ...payload,
    task: 'detect',
    inferenceTimeMs: (payload.inference_time ?? 0) * 1000,
    detections: (payload.detections ?? []).map((item, index) => ({
      id: item.id ?? index,
      ...item,
      className: item.class_name ?? item.className,
    })),
  };
}

function guessMimeType(filename, fallback = 'image/jpeg') {
  const lower = `${filename ?? ''}`.toLowerCase();
  if (lower.endsWith('.png')) return 'image/png';
  if (lower.endsWith('.webp')) return 'image/webp';
  if (lower.endsWith('.gif')) return 'image/gif';
  return fallback;
}

async function dataUrlToFile(dataUrl, filename, preferredMimeType) {
  const response = await fetch(dataUrl);
  const blob = await response.blob();
  const fileType = preferredMimeType || blob.type || guessMimeType(filename);
  return new File([blob], filename, { type: fileType });
}

function resolveRuntimeSampleSource(sample, apiBaseUrl) {
  const sourceUrl = sample?.sourceUrl ?? getSampleSource(sample);
  if (!sourceUrl) return '';
  if (sourceUrl.startsWith('data:') || sourceUrl.startsWith('blob:')) {
    return sourceUrl;
  }
  if (sourceUrl.startsWith('http://') || sourceUrl.startsWith('https://')) {
    return `${apiBaseUrl}/proxy/model?url=${encodeURIComponent(sourceUrl)}`;
  }
  return sourceUrl;
}

function renderResultDetails(result, hoveredIndex, setHoveredIndex) {
  if (!result) {
    return (
      <div className="h-full flex items-center justify-center rounded-[1.5rem] border border-dashed border-border bg-muted/10 p-8 text-center text-sm text-muted-foreground">
        Run a dataset sample to inspect detections, classifications, or segmentation results here.
      </div>
    );
  }

  if (result.task === 'classify') {
    return (
      <div className="rounded-[1.5rem] border bg-card overflow-hidden">
        <div className="flex items-center justify-between border-b bg-muted/10 px-5 py-4">
          <div>
            <div className="text-sm font-semibold text-foreground">{result.topPrediction?.className ?? 'No top prediction'}</div>
            <div className="text-[11px] text-muted-foreground">Top-ranked class for the current sample.</div>
          </div>
          <Badge variant="outline" className="font-mono text-xs">
            {result.inferenceTimeMs?.toFixed?.(0) ?? 0}ms
          </Badge>
        </div>
        <div className="space-y-2 p-4">
          {(result.predictions ?? []).map((prediction, index) => (
            <div key={`${prediction.classId}-${index}`} className="flex items-center justify-between rounded-2xl border bg-muted/5 px-4 py-3">
              <div>
                <div className="text-sm font-medium text-foreground">{prediction.className}</div>
                <div className="text-[10px] uppercase tracking-[0.2em] text-muted-foreground">Rank {index + 1}</div>
              </div>
              <Badge variant={index === 0 ? 'default' : 'secondary'} className="font-mono text-xs">
                {(prediction.confidence * 100).toFixed(1)}%
              </Badge>
            </div>
          ))}
        </div>
      </div>
    );
  }

  return (
    <div className="rounded-[1.5rem] border bg-card overflow-hidden">
      <div className="flex items-center justify-between border-b bg-muted/10 px-5 py-4">
        <div>
          <div className="text-sm font-semibold text-foreground">{result.detections?.length ?? 0} results</div>
          <div className="text-[11px] text-muted-foreground">Interactive result list for the active sample.</div>
        </div>
        <Badge variant="outline" className="font-mono text-xs">
          {result.inferenceTimeMs?.toFixed?.(0) ?? 0}ms
        </Badge>
      </div>
      <div className="space-y-2 p-4">
        {(result.detections?.length ?? 0) > 0 ? (
          result.detections.map((item, index) => (
            <div
              key={item.id ?? index}
              onMouseEnter={() => setHoveredIndex(index)}
              onMouseLeave={() => setHoveredIndex(null)}
              className={`rounded-2xl border px-4 py-3 transition-all ${index === hoveredIndex ? 'border-primary/30 bg-primary/5' : 'bg-muted/5'}`}
            >
              <div className="flex items-center justify-between gap-3">
                <div className="min-w-0">
                  <div className="truncate text-sm font-medium text-foreground">{item.className}</div>
                  <div className="text-[10px] uppercase tracking-[0.18em] text-muted-foreground">
                    {result.task === 'pose' && `${(item.keypoints ?? []).filter((point) => point.confidence >= 0.35).length} visible keypoints`}
                    {result.task === 'segment' && `mask ${((item.maskCoverage ?? 0) * 100).toFixed(1)}% coverage`}
                    {result.task === 'obb' && `${(((item.angle ?? 0) * 180) / Math.PI).toFixed(1)}° rotation`}
                    {result.task === 'detect' && `${item.bbox?.width?.toFixed?.(0) ?? 0}×${item.bbox?.height?.toFixed?.(0) ?? 0}`}
                  </div>
                </div>
                <Badge variant="secondary" className="font-mono text-xs">
                  {((item.confidence ?? 0) * 100).toFixed(1)}%
                </Badge>
              </div>
            </div>
          ))
        ) : (
          <div className="rounded-2xl border border-dashed border-border bg-muted/10 p-6 text-center text-sm text-muted-foreground">
            No results were found above the current confidence threshold.
          </div>
        )}
      </div>
    </div>
  );
}

function SectionNav({ activeSection, onChange }) {
  return (
    <div className="rounded-[1.75rem] border bg-card p-3 shadow-sm">
      <div className="mb-3 px-3 pt-2">
        <div className="text-[10px] font-semibold uppercase tracking-[0.28em] text-muted-foreground">Dataset Workspace</div>
      </div>
      <div className="space-y-1.5">
        {DATASET_SECTIONS.map((section) => {
          const Icon = section.icon;
          const isActive = activeSection === section.id;
          return (
            <button
              key={section.id}
              type="button"
              onClick={() => onChange(section.id)}
              className={`flex w-full items-center gap-3 rounded-2xl px-3 py-3 text-left text-sm transition-all ${
                isActive ? 'bg-primary text-primary-foreground shadow-lg shadow-primary/20' : 'text-muted-foreground hover:bg-muted/60 hover:text-foreground'
              }`}
            >
              <Icon className="h-4 w-4 shrink-0" />
              <span className="font-medium">{section.label}</span>
            </button>
          );
        })}
      </div>
    </div>
  );
}

function SampleModal({
  sample,
  isOpen,
  onOpenChange,
  onRunSample,
  onNext,
  onPrevious,
  hasNext,
  hasPrevious,
  isSelected,
  onToggleSelection,
  selectedDataset,
}) {
  if (!sample) return null;

  return (
    <Dialog open={isOpen} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-5xl overflow-hidden rounded-[2rem] border border-border/80 p-0">
        <DialogHeader className="border-b bg-muted/10 px-6 py-5">
          <DialogTitle className="flex flex-col gap-2 text-left md:flex-row md:items-center md:justify-between">
            <div>
              <div className="text-xl font-semibold">{sample.file_name}</div>
              <div className="mt-1 text-sm font-normal text-muted-foreground">
                {selectedDataset?.name} · {selectedDataset?.workspace}/{selectedDataset?.project}
              </div>
            </div>
            <div className="flex flex-wrap gap-2">
              <Badge variant="outline">{inferSampleSplit(sample) === 'all' ? 'Preview' : inferSampleSplit(sample)}</Badge>
              <Badge variant="outline">{sample.mime_type ?? 'image/jpeg'}</Badge>
            </div>
          </DialogTitle>
        </DialogHeader>
        <div className="grid gap-0 lg:grid-cols-[1.5fr_0.75fr]">
          <div className="bg-black/95 p-4 md:p-6">
            <div className="overflow-hidden rounded-[1.5rem] border border-white/10 bg-black">
              <img src={getSampleSource(sample)} alt={sample.file_name} className="h-full max-h-[68vh] w-full object-contain" />
            </div>
          </div>
          <div className="space-y-6 bg-card p-6">
            <div className="space-y-3">
              <div className="text-[10px] font-semibold uppercase tracking-[0.24em] text-muted-foreground">Actions</div>
              <Button className="w-full rounded-2xl" onClick={() => onRunSample(sample)}>
                <Sparkles className="mr-2 h-4 w-4" />
                Run This Image
              </Button>
              <Button variant={isSelected ? 'default' : 'outline'} className="w-full rounded-2xl" onClick={() => onToggleSelection(sample.sample_id)}>
                {isSelected ? 'Selected for Batch Run' : 'Select for Batch Run'}
              </Button>
            </div>

            <div className="space-y-3 rounded-[1.5rem] border bg-muted/10 p-4">
              <div className="text-[10px] font-semibold uppercase tracking-[0.24em] text-muted-foreground">Metadata</div>
              <div className="space-y-2 text-sm text-foreground">
                <div className="flex justify-between gap-4">
                  <span className="text-muted-foreground">Split</span>
                  <span className="font-medium capitalize">{inferSampleSplit(sample)}</span>
                </div>
                <div className="flex justify-between gap-4">
                  <span className="text-muted-foreground">Sample ID</span>
                  <span className="font-medium">{sample.sample_id}</span>
                </div>
                <div className="flex justify-between gap-4">
                  <span className="text-muted-foreground">Classes</span>
                  <span className="font-medium">{selectedDataset?.class_count ?? 0}</span>
                </div>
              </div>
            </div>

            <div className="space-y-3">
              <div className="text-[10px] font-semibold uppercase tracking-[0.24em] text-muted-foreground">Navigate</div>
              <div className="flex gap-2">
                <Button variant="outline" className="flex-1 rounded-2xl" onClick={onPrevious} disabled={!hasPrevious}>
                  Previous
                </Button>
                <Button variant="outline" className="flex-1 rounded-2xl" onClick={onNext} disabled={!hasNext}>
                  Next
                </Button>
              </div>
            </div>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}

export default function DatasetsTab({
  apiBaseUrl,
  studioTask,
  taskModelSelections,
  browserAssets,
  customLabelsByTask,
  detectionModelName,
}) {
  const [apiKey, setApiKey] = useState('');
  const [draftKey, setDraftKey] = useState('');
  const [query, setQuery] = useState('');
  const [searching, setSearching] = useState(false);
  const [results, setResults] = useState([]);
  const [selectedDataset, setSelectedDataset] = useState(null);
  const [loadingDetail, setLoadingDetail] = useState(false);
  const [runningInference, setRunningInference] = useState(false);
  const [datasetSection, setDatasetSection] = useState('overview');
  const [imageSearch, setImageSearch] = useState('');
  const [splitFilter, setSplitFilter] = useState('all');
  const [sortOrder, setSortOrder] = useState('featured');
  const [imageLayout, setImageLayout] = useState('grid');
  const [showAnnotations, setShowAnnotations] = useState(true);
  const [selectedSampleIds, setSelectedSampleIds] = useState([]);
  const [activeImageIndex, setActiveImageIndex] = useState(null);
  const [hoveredIndex, setHoveredIndex] = useState(null);
  const [dockOpen, setDockOpen] = useState(false);
  const [dockCollapsed, setDockCollapsed] = useState(false);
  const [dockHeight, setDockHeight] = useState(360);
  const [dockResults, setDockResults] = useState([]);
  const [activeResultIndex, setActiveResultIndex] = useState(0);
  const resizingRef = useRef(false);
  const resizeStartRef = useRef({ y: 0, height: 360 });

  useEffect(() => {
    const stored = window.localStorage.getItem(STORAGE_KEY) ?? '';
    setApiKey(stored);
    setDraftKey(stored);
  }, []);

  useEffect(() => {
    if (!resizingRef.current) return undefined;

    const handleMove = (event) => {
      const nextHeight = resizeStartRef.current.height + (resizeStartRef.current.y - event.clientY);
      setDockHeight(clamp(nextHeight, 240, 620));
    };

    const handleUp = () => {
      resizingRef.current = false;
      document.body.style.cursor = '';
      document.body.style.userSelect = '';
      window.removeEventListener('mousemove', handleMove);
      window.removeEventListener('mouseup', handleUp);
    };

    window.addEventListener('mousemove', handleMove);
    window.addEventListener('mouseup', handleUp);

    return () => {
      window.removeEventListener('mousemove', handleMove);
      window.removeEventListener('mouseup', handleUp);
    };
  }, [dockOpen, dockCollapsed]);

  const selectedModelId = taskModelSelections[studioTask];
  const selectedBrowserModel = useMemo(
    () => (!isCustomModelSelection(selectedModelId) ? getModelById(selectedModelId) : null),
    [selectedModelId],
  );
  const resolvedBrowserModel = useMemo(
    () => makeBrowserModelConfig(selectedBrowserModel, browserAssets[selectedModelId]),
    [selectedBrowserModel, browserAssets, selectedModelId],
  );

  const taskLabel = TASK_OPTIONS.find((option) => option.id === studioTask)?.label ?? studioTask;
  const runtimeLabel = studioTask === 'detect' ? 'FastAPI' : 'Browser ONNX';
  const modelLabel = studioTask === 'detect' ? detectionModelName : (resolvedBrowserModel?.displayName ?? 'Unavailable');
  const workspaceMode = selectedDataset ? 'dataset-workspace' : 'discovery';

  const datasetSamples = useMemo(() => {
    const sourceSamples = (selectedDataset?.samples ?? []).length
      ? (selectedDataset?.samples ?? [])
      : (selectedDataset?.thumbnail ? [{
          sample_id: `${selectedDataset.dataset_id ?? selectedDataset.project ?? 'dataset'}-thumbnail-preview`,
          file_name: `${selectedDataset.project ?? 'dataset'}-preview.jpg`,
          mime_type: 'image/jpeg',
          image_url: selectedDataset.thumbnail,
          path: 'preview/thumbnail',
          isFallbackPreview: true,
        }] : []);

    return sourceSamples.map((sample, index) => ({
      ...sample,
      split: inferSampleSplit(sample),
      sourceUrl: getSampleSource(sample),
      sortIndex: index,
    }));
  }, [selectedDataset]);

  const filteredSamples = useMemo(() => {
    const lowered = imageSearch.trim().toLowerCase();
    const list = datasetSamples.filter((sample) => {
      const matchesSearch = !lowered || `${sample.file_name} ${sample.sample_id}`.toLowerCase().includes(lowered);
      const matchesSplit = splitFilter === 'all' || sample.split === splitFilter;
      return matchesSearch && matchesSplit;
    });

    if (sortOrder === 'az') {
      return [...list].sort((left, right) => left.file_name.localeCompare(right.file_name));
    }
    if (sortOrder === 'za') {
      return [...list].sort((left, right) => right.file_name.localeCompare(left.file_name));
    }
    return [...list].sort((left, right) => left.sortIndex - right.sortIndex);
  }, [datasetSamples, imageSearch, splitFilter, sortOrder]);

  const activeImageSample = activeImageIndex === null ? null : filteredSamples[activeImageIndex] ?? null;
  const selectedSamples = datasetSamples.filter((sample) => selectedSampleIds.includes(sample.sample_id));
  const activeDockEntry = dockResults[activeResultIndex] ?? null;
  const activeDockResult = activeDockEntry?.result ?? null;

  const saveApiKey = () => {
    const nextValue = draftKey.trim();
    window.localStorage.setItem(STORAGE_KEY, nextValue);
    setApiKey(nextValue);
    toast.success(nextValue ? 'Roboflow API key saved in this browser' : 'Roboflow API key cleared from this browser');
  };

  const searchDatasets = async () => {
    if (!query.trim()) {
      toast.error('Enter a search query first');
      return;
    }

    setSearching(true);
    try {
      const response = await axios.get(`${apiBaseUrl}/roboflow/search`, {
        params: { q: query.trim(), api_key: apiKey || undefined },
      });
      const nextResults = response.data.results ?? [];
      setResults(nextResults);
      if (nextResults.length === 0) {
        toast.info('No datasets found for that query');
      }
    } catch (error) {
      console.error('Dataset search failed', error);
      toast.error(error.response?.data?.detail || 'Failed to search Roboflow Universe');
    } finally {
      setSearching(false);
    }
  };

  const openDataset = async (dataset) => {
    setLoadingDetail(true);
    setSelectedDataset(dataset);
    setDatasetSection('overview');
    setSelectedSampleIds([]);
    setActiveImageIndex(null);
    setDockResults([]);
    setDockOpen(false);
    setDockCollapsed(false);
    setHoveredIndex(null);
    setActiveResultIndex(0);
    setImageSearch('');
    setSplitFilter('all');
    setSortOrder('featured');
    setImageLayout('grid');

    try {
      const response = await axios.get(`${apiBaseUrl}/roboflow/dataset/${dataset.workspace}/${dataset.project}`, {
        params: { api_key: apiKey || undefined },
      });
      const mergedDataset = { ...dataset, ...response.data };
      if (!(mergedDataset.samples ?? []).length && mergedDataset.thumbnail) {
        toast.info('Roboflow did not return preview samples for this dataset. Using the dataset cover image as a fallback preview.');
      }
      setSelectedDataset(mergedDataset);
    } catch (error) {
      console.error('Dataset detail failed', error);
      toast.error(error.response?.data?.detail || 'Failed to load dataset details');
    } finally {
      setLoadingDetail(false);
    }
  };

  const closeWorkspace = () => {
    setSelectedDataset(null);
    setDockOpen(false);
    setDockCollapsed(false);
    setDockResults([]);
    setActiveResultIndex(0);
    setSelectedSampleIds([]);
    setActiveImageIndex(null);
    setHoveredIndex(null);
  };

  const toggleSampleSelection = (sampleId) => {
    setSelectedSampleIds((current) => (
      current.includes(sampleId) ? current.filter((item) => item !== sampleId) : [...current, sampleId]
    ));
  };

  const runSamplesInStudio = async (samples, originLabel) => {
    if (!samples.length) {
      toast.info('Select one or more samples first');
      return;
    }

    setRunningInference(true);
    setDockOpen(true);
    setDockCollapsed(false);
    setHoveredIndex(null);
    setDockResults([]);
    setActiveResultIndex(0);

    try {
      let nextResults = [];

      if (studioTask === 'detect') {
        nextResults = await Promise.all(samples.map(async (sample) => {
          const runtimeSource = resolveRuntimeSampleSource(sample, apiBaseUrl);
          const file = await dataUrlToFile(
            runtimeSource,
            sample.file_name || `${sample.sample_id}.jpg`,
            sample.mime_type || guessMimeType(sample.file_name),
          );
          const formData = new FormData();
          formData.append('file', file);
          const response = await axios.post(`${apiBaseUrl}/detect/image`, formData, {
            headers: { 'Content-Type': 'multipart/form-data' },
          });

          return {
            sample,
            result: normalizeDetectResult(response.data),
          };
        }));
      } else {
        if (!resolvedBrowserModel?.modelUrl) {
          throw new Error('The current browser model is not ready yet.');
        }

        nextResults = await Promise.all(samples.map(async (sample) => {
          const runtimeSource = resolveRuntimeSampleSource(sample, apiBaseUrl);
          const image = await loadImageElement(runtimeSource);
          const browserResult = await runInference({
            task: studioTask,
            image,
            modelConfig: {
              ...resolvedBrowserModel,
              classNames: customLabelsByTask[studioTask]?.length ? customLabelsByTask[studioTask] : resolvedBrowserModel.classNames,
            },
          });

          return {
            sample,
            result: browserResult,
          };
        }));
      }

      setDockResults(nextResults);
      setActiveResultIndex(0);
      toast.success(`Ran ${taskLabel} on ${nextResults.length} dataset sample${nextResults.length === 1 ? '' : 's'} from ${originLabel}`);
    } catch (error) {
      console.error('Dataset inference failed', error);
      toast.error(error.response?.data?.detail || error.message || 'Failed to run dataset inference');
    } finally {
      setRunningInference(false);
    }
  };

  const runCuratedBatch = () => {
    const samples = datasetSamples.slice(0, Math.min(datasetSamples.length, 4));
    runSamplesInStudio(samples, 'Overview');
  };

  const runSelectedBatch = () => {
    if (!selectedSamples.length) {
      toast.info('Select one or more images in the grid first');
      return;
    }
    runSamplesInStudio(selectedSamples, 'Images');
  };

  const startDockResize = (event) => {
    resizingRef.current = true;
    resizeStartRef.current = { y: event.clientY, height: dockHeight };
    document.body.style.cursor = 'ns-resize';
    document.body.style.userSelect = 'none';
  };

  const detailStats = selectedDataset ? [
    { label: 'Images', value: formatCount(selectedDataset.image_count, 'images') },
    { label: 'Classes', value: formatCount(selectedDataset.class_count, 'classes') },
    { label: 'Version', value: `v${selectedDataset.version ?? selectedDataset.version_number ?? 'latest'}` },
    { label: 'Task', value: selectedDataset.type ?? 'Dataset' },
  ] : [];

  const renderDiscovery = () => (
    <div className="space-y-6">
      <div className="overflow-hidden rounded-[2rem] border border-border/70 bg-card shadow-[0_24px_80px_-40px_rgba(198,93,60,0.35)]">
        <div className="border-b border-border/70 bg-[radial-gradient(circle_at_top_left,_rgba(198,93,60,0.16),_transparent_32%),linear-gradient(135deg,rgba(255,255,255,0.98),rgba(251,246,243,0.96))] px-6 py-7 md:px-8">
          <div className="flex flex-col gap-5 xl:flex-row xl:items-end xl:justify-between">
            <div className="max-w-3xl space-y-3">
              <Badge variant="outline" className="rounded-full border-primary/25 bg-primary/5 px-3 py-1 text-[10px] font-semibold uppercase tracking-[0.26em] text-primary">
                Roboflow Dataset Explorer
              </Badge>
              <div>
                <h2 className="font-outfit text-3xl font-semibold tracking-tight text-foreground md:text-5xl">
                  Search public datasets and open them as a working surface.
                </h2>
                <p className="mt-3 max-w-2xl text-sm leading-6 text-muted-foreground md:text-base">
                  Move from discovery into a dedicated dataset workspace, inspect curated previews, and run the active Deplyze Studio task without leaving the app.
                </p>
              </div>
            </div>
            <div className="grid gap-3 sm:grid-cols-3 xl:min-w-[360px]">
              <div className="rounded-[1.4rem] border border-border/70 bg-white/80 px-4 py-4">
                <div className="text-[10px] font-semibold uppercase tracking-[0.22em] text-muted-foreground">Task</div>
                <div className="mt-2 text-base font-semibold text-foreground">{taskLabel}</div>
              </div>
              <div className="rounded-[1.4rem] border border-border/70 bg-white/80 px-4 py-4">
                <div className="text-[10px] font-semibold uppercase tracking-[0.22em] text-muted-foreground">Runtime</div>
                <div className="mt-2 text-base font-semibold text-foreground">{runtimeLabel}</div>
              </div>
              <div className="rounded-[1.4rem] border border-border/70 bg-white/80 px-4 py-4">
                <div className="text-[10px] font-semibold uppercase tracking-[0.22em] text-muted-foreground">Model</div>
                <div className="mt-2 truncate text-base font-semibold text-foreground">{modelLabel}</div>
              </div>
            </div>
          </div>
        </div>

        <div className="space-y-4 px-6 py-5 md:px-8">
          <div className="rounded-[1.6rem] border border-border/70 bg-muted/10 px-4 py-4">
            <div className="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
              <div className="space-y-1">
                <div className="flex items-center gap-2 text-sm font-semibold text-foreground">
                  <KeyRound className="h-4 w-4 text-primary" />
                  Roboflow Key
                </div>
                <div className="text-sm text-muted-foreground">
                  Stored locally in this browser. Requests fall back to backend <code>ROBOFLOW_API_KEY</code> when empty.
                </div>
              </div>
              <div className="flex flex-1 flex-col gap-3 lg:max-w-2xl lg:flex-row">
                <Input
                  type="password"
                  value={draftKey}
                  onChange={(event) => setDraftKey(event.target.value)}
                  placeholder="Paste Roboflow API key"
                  className="h-12 rounded-2xl bg-white"
                />
                <Button onClick={saveApiKey} className="h-12 rounded-2xl px-6">
                  {apiKey ? 'Update Key' : 'Save Key'}
                </Button>
              </div>
            </div>
            <div className="mt-4 flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
              <Badge variant={apiKey ? 'secondary' : 'outline'} className="rounded-full px-3 py-1">
                {apiKey ? 'Using browser key' : 'Using backend fallback'}
              </Badge>
              <span>{apiKey ? 'Roboflow requests will use the saved browser key.' : 'No browser key saved in localStorage.'}</span>
            </div>
          </div>

          <div className="rounded-[1.8rem] border border-border/70 bg-white px-4 py-4 shadow-sm">
            <div className="flex flex-col gap-4 xl:flex-row xl:items-center">
              <div className="relative flex-1">
                <Search className="pointer-events-none absolute left-4 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
                <Input
                  value={query}
                  onChange={(event) => setQuery(event.target.value)}
                  onKeyDown={(event) => {
                    if (event.key === 'Enter') searchDatasets();
                  }}
                  placeholder='Search Roboflow Universe datasets, for example "pothole detection"'
                  className="h-14 rounded-2xl border-border bg-background pl-11 text-base"
                />
              </div>
              <div className="flex items-center gap-3">
                <div className="hidden rounded-2xl border border-border bg-muted/10 px-4 py-3 text-sm text-muted-foreground md:block">
                  <span className="font-semibold text-foreground">{results.length}</span> results
                </div>
                <Button onClick={searchDatasets} disabled={searching} className="h-14 rounded-2xl px-7 text-sm font-semibold">
                  {searching ? 'Searching...' : 'Search'}
                </Button>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="space-y-5">
        <div className="flex flex-col gap-2 md:flex-row md:items-end md:justify-between">
          <div>
            <div className="text-[10px] font-semibold uppercase tracking-[0.26em] text-muted-foreground">Discovery</div>
            <h3 className="mt-1 text-2xl font-semibold tracking-tight text-foreground">Search results</h3>
          </div>
          <div className="text-sm text-muted-foreground">
            {searching ? 'Looking for datasets in Roboflow Universe…' : `${results.length} dataset${results.length === 1 ? '' : 's'} available`}
          </div>
        </div>

        {results.length === 0 ? (
          <Card className="overflow-hidden rounded-[2rem] border border-dashed border-border/80 bg-card/70 shadow-none">
            <CardContent className="flex min-h-[360px] flex-col items-center justify-center px-6 py-16 text-center">
              <div className="mb-5 rounded-full bg-primary/10 p-5 text-primary">
                <Database className="h-8 w-8" />
              </div>
              <h4 className="text-xl font-semibold text-foreground">Start with a dataset search</h4>
              <p className="mt-3 max-w-xl text-sm leading-6 text-muted-foreground">
                Use a topic like potholes, hard hats, retail shelves, or vehicle counts. Opening a dataset will bring you into a workspace designed for browsing images and running your active studio task.
              </p>
            </CardContent>
          </Card>
        ) : (
          <div className="grid gap-5 lg:grid-cols-2 2xl:grid-cols-3">
            {results.map((dataset) => (
              <button
                key={dataset.dataset_id}
                type="button"
                onClick={() => openDataset(dataset)}
                className="group overflow-hidden rounded-[2rem] border border-border/70 bg-card text-left shadow-sm transition-all hover:-translate-y-1 hover:border-primary/30 hover:shadow-[0_30px_80px_-45px_rgba(198,93,60,0.55)]"
              >
                <div className="relative aspect-[16/10] overflow-hidden bg-muted/20">
                  {dataset.thumbnail ? (
                    <img src={dataset.thumbnail} alt={dataset.name} className="h-full w-full object-cover transition-transform duration-500 group-hover:scale-105" />
                  ) : (
                    <div className="flex h-full w-full items-center justify-center text-muted-foreground">
                      <ImageIcon className="h-10 w-10" />
                    </div>
                  )}
                  <div className="absolute inset-x-0 bottom-0 h-32 bg-gradient-to-t from-black/65 via-black/10 to-transparent" />
                  <div className="absolute bottom-4 left-4 right-4 flex items-center justify-between gap-3">
                    <Badge className="rounded-full border-none bg-white/90 px-3 py-1 text-[10px] font-semibold uppercase tracking-[0.22em] text-foreground">
                      {dataset.type ?? 'Dataset'}
                    </Badge>
                    {(dataset.version ?? dataset.version_number) && (
                      <Badge variant="secondary" className="rounded-full bg-white/85 px-3 py-1 text-xs text-foreground">
                        v{dataset.version ?? dataset.version_number}
                      </Badge>
                    )}
                  </div>
                </div>
                <div className="space-y-4 px-5 py-5">
                  <div>
                    <div className="line-clamp-2 text-2xl font-semibold tracking-tight text-foreground">{dataset.name}</div>
                    <div className="mt-2 text-sm text-muted-foreground">
                      {dataset.workspace}/{dataset.project}
                    </div>
                  </div>
                  <div className="flex flex-wrap gap-2">
                    <Badge variant="outline" className="rounded-full px-3 py-1">{formatCount(dataset.image_count, 'images')}</Badge>
                    <Badge variant="outline" className="rounded-full px-3 py-1">{formatCount(dataset.class_count, 'classes')}</Badge>
                  </div>
                  <div className="flex items-center justify-between border-t border-border/70 pt-4 text-sm">
                    <span className="text-muted-foreground">Open the full dataset workspace</span>
                    <span className="font-semibold text-primary">Open Dataset</span>
                  </div>
                </div>
              </button>
            ))}
          </div>
        )}
      </div>
    </div>
  );

  const renderOverviewSection = () => (
    <div className="space-y-6">
      <div className="overflow-hidden rounded-[2rem] border border-border/70 bg-card shadow-sm">
        <div className="grid gap-0 xl:grid-cols-[1.3fr_0.7fr]">
          <div className="border-b border-border/70 p-5 xl:border-b-0 xl:border-r">
            <div className="grid gap-3 md:grid-cols-5">
              {datasetSamples.slice(0, 5).map((sample, index) => (
                <button
                  key={sample.sample_id}
                  type="button"
                  onClick={() => {
                    setDatasetSection('images');
                    setImageLayout('grid');
                    setActiveImageIndex(index);
                  }}
                  className={`overflow-hidden rounded-[1.2rem] bg-muted/10 ${index === 0 ? 'md:col-span-2 md:row-span-2 aspect-[1.15/1]' : 'aspect-[1.1/1]'}`}
                >
                  <img src={sample.sourceUrl} alt={sample.file_name} className="h-full w-full object-cover transition-transform duration-500 hover:scale-105" />
                </button>
              ))}
            </div>
          </div>
          <div className="space-y-5 px-5 py-5">
            <div>
              <div className="text-[10px] font-semibold uppercase tracking-[0.24em] text-muted-foreground">Dataset Overview</div>
              <h3 className="mt-2 text-3xl font-semibold tracking-tight text-foreground">{selectedDataset.name}</h3>
              <div className="mt-3 flex flex-wrap items-center gap-2 text-sm text-muted-foreground">
                <span>{selectedDataset.workspace}/{selectedDataset.project}</span>
                {(selectedDataset.version ?? selectedDataset.version_number) && (
                  <>
                    <span>·</span>
                    <span>Updated in v{selectedDataset.version ?? selectedDataset.version_number}</span>
                  </>
                )}
              </div>
            </div>

            <div className="flex flex-wrap gap-2">
              <Badge variant="outline" className="rounded-full px-3 py-1">{selectedDataset.type ?? 'Dataset'}</Badge>
              <Badge variant="outline" className="rounded-full px-3 py-1">{formatCount(selectedDataset.image_count, 'images')}</Badge>
              <Badge variant="outline" className="rounded-full px-3 py-1">{formatCount(selectedDataset.class_count, 'classes')}</Badge>
            </div>

            <div className="flex flex-col gap-3 sm:flex-row">
              <Button className="rounded-2xl px-5" onClick={runCuratedBatch} disabled={runningInference || loadingDetail}>
                <Sparkles className="mr-2 h-4 w-4" />
                {runningInference ? 'Running…' : 'Run in Studio'}
              </Button>
              <Button variant="outline" className="rounded-2xl px-5" onClick={() => setDatasetSection('images')}>
                Select Samples
              </Button>
            </div>
          </div>
        </div>
      </div>

      <div className="grid gap-4 md:grid-cols-4">
        {detailStats.map((item) => (
          <div key={item.label} className="rounded-[1.6rem] border border-border/70 bg-card px-5 py-5 shadow-sm">
            <div className="text-[10px] font-semibold uppercase tracking-[0.24em] text-muted-foreground">{item.label}</div>
            <div className="mt-3 text-lg font-semibold text-foreground">{item.value}</div>
          </div>
        ))}
      </div>

      <div className="grid gap-6 xl:grid-cols-[1.15fr_0.85fr]">
        <div className="rounded-[2rem] border border-border/70 bg-card p-6 shadow-sm">
          <div className="flex items-center justify-between gap-3">
            <div>
              <div className="text-[10px] font-semibold uppercase tracking-[0.24em] text-muted-foreground">About</div>
              <h4 className="mt-2 text-xl font-semibold tracking-tight text-foreground">How this dataset plugs into Deplyze Studio</h4>
            </div>
          </div>
          <p className="mt-4 max-w-3xl text-sm leading-7 text-muted-foreground">
            Use overview to launch a curated sample batch, then switch to Images when you want to inspect individual previews, pick a subset, and compare inference outputs without leaving the dataset context.
          </p>

          <div className="mt-6 rounded-[1.5rem] border border-border/70 bg-muted/10 p-4">
            <div className="text-[10px] font-semibold uppercase tracking-[0.24em] text-muted-foreground">Active Studio Session</div>
            <div className="mt-3 grid gap-3 sm:grid-cols-3">
              <div className="rounded-[1.2rem] border bg-card px-4 py-4">
                <div className="text-[10px] uppercase tracking-[0.22em] text-muted-foreground">Task</div>
                <div className="mt-2 font-semibold text-foreground">{taskLabel}</div>
              </div>
              <div className="rounded-[1.2rem] border bg-card px-4 py-4">
                <div className="text-[10px] uppercase tracking-[0.22em] text-muted-foreground">Runtime</div>
                <div className="mt-2 font-semibold text-foreground">{runtimeLabel}</div>
              </div>
              <div className="rounded-[1.2rem] border bg-card px-4 py-4">
                <div className="text-[10px] uppercase tracking-[0.22em] text-muted-foreground">Model</div>
                <div className="mt-2 truncate font-semibold text-foreground">{modelLabel}</div>
              </div>
            </div>
          </div>
        </div>

        <div className="rounded-[2rem] border border-border/70 bg-card p-6 shadow-sm">
          <div className="text-[10px] font-semibold uppercase tracking-[0.24em] text-muted-foreground">Classes</div>
          <h4 className="mt-2 text-xl font-semibold tracking-tight text-foreground">Labels in this dataset</h4>
          <div className="mt-5 flex flex-wrap gap-2">
            {(selectedDataset.classes ?? []).map((className) => (
              <Badge key={className} variant="secondary" className="rounded-full px-3 py-1.5 text-sm">
                {className}
              </Badge>
            ))}
            {(selectedDataset.classes ?? []).length === 0 && (
              <div className="rounded-[1.2rem] border border-dashed border-border bg-muted/10 px-4 py-8 text-center text-sm text-muted-foreground">
                No classes were returned for this dataset.
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );

  const renderImagesSection = () => (
    <div className="space-y-5">
      <div className="rounded-[2rem] border border-border/70 bg-card p-5 shadow-sm">
        <div className="flex flex-col gap-4 xl:flex-row xl:items-center xl:justify-between">
          <div>
            <div className="text-[10px] font-semibold uppercase tracking-[0.24em] text-muted-foreground">Images</div>
            <h4 className="mt-2 text-2xl font-semibold tracking-tight text-foreground">Browse preview samples</h4>
          </div>
          <div className="flex flex-wrap items-center gap-2">
            <Button variant="outline" className="rounded-2xl" onClick={() => setSelectedSampleIds([])} disabled={!selectedSampleIds.length}>
              Clear Selection
            </Button>
            <Button className="rounded-2xl" onClick={runSelectedBatch} disabled={runningInference || !selectedSampleIds.length}>
              <Sparkles className="mr-2 h-4 w-4" />
              Run Selected
            </Button>
          </div>
        </div>

        <div className="mt-5 grid gap-3 xl:grid-cols-[1.5fr_repeat(4,minmax(0,1fr))]">
          <div className="relative">
            <Search className="pointer-events-none absolute left-4 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
            <Input
              value={imageSearch}
              onChange={(event) => setImageSearch(event.target.value)}
              placeholder="Search images"
              className="h-12 rounded-2xl bg-background pl-11"
            />
          </div>
          <select value={splitFilter} onChange={(event) => setSplitFilter(event.target.value)} className="h-12 rounded-2xl border border-input bg-background px-4 text-sm">
            <option value="all">All splits</option>
            <option value="train">Train</option>
            <option value="valid">Valid</option>
            <option value="test">Test</option>
          </select>
          <select value={sortOrder} onChange={(event) => setSortOrder(event.target.value)} className="h-12 rounded-2xl border border-input bg-background px-4 text-sm">
            <option value="featured">Featured</option>
            <option value="az">Name A-Z</option>
            <option value="za">Name Z-A</option>
          </select>
          <Button variant="outline" className="h-12 rounded-2xl justify-start gap-2" onClick={() => setShowAnnotations((current) => !current)}>
            {showAnnotations ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
            {showAnnotations ? 'Hide overlay' : 'Show overlay'}
          </Button>
          <div className="flex h-12 items-center gap-2 rounded-2xl border bg-background p-1">
            <button
              type="button"
              onClick={() => setImageLayout('grid')}
              className={`flex h-full flex-1 items-center justify-center rounded-[0.95rem] transition-colors ${imageLayout === 'grid' ? 'bg-primary text-primary-foreground' : 'text-muted-foreground hover:bg-muted'}`}
              aria-label="Grid view"
            >
              <Grid2X2 className="h-4 w-4" />
            </button>
            <button
              type="button"
              onClick={() => setImageLayout('list')}
              className={`flex h-full flex-1 items-center justify-center rounded-[0.95rem] transition-colors ${imageLayout === 'list' ? 'bg-primary text-primary-foreground' : 'text-muted-foreground hover:bg-muted'}`}
              aria-label="List view"
            >
              <List className="h-4 w-4" />
            </button>
          </div>
        </div>

        <div className="mt-4 flex flex-wrap items-center gap-2 text-sm text-muted-foreground">
          <Badge variant="outline" className="rounded-full px-3 py-1">{filteredSamples.length} visible</Badge>
          <Badge variant={selectedSampleIds.length ? 'secondary' : 'outline'} className="rounded-full px-3 py-1">
            {selectedSampleIds.length} selected
          </Badge>
        </div>
      </div>

      {filteredSamples.length === 0 ? (
        <div className="rounded-[2rem] border border-dashed border-border bg-card/70 px-6 py-20 text-center text-sm text-muted-foreground">
          No preview images match the current filters.
        </div>
      ) : imageLayout === 'grid' ? (
        <div className="grid gap-4 md:grid-cols-2 2xl:grid-cols-3">
          {filteredSamples.map((sample, index) => {
            const isSelected = selectedSampleIds.includes(sample.sample_id);
            return (
              <div key={sample.sample_id} className="group overflow-hidden rounded-[1.8rem] border border-border/70 bg-card shadow-sm">
                <button type="button" onClick={() => setActiveImageIndex(index)} className="relative block aspect-[1.18/1] w-full overflow-hidden bg-muted/10">
                  <img src={sample.sourceUrl} alt={sample.file_name} className="h-full w-full object-cover transition-transform duration-500 group-hover:scale-105" />
                  <div className="absolute inset-x-0 bottom-0 h-24 bg-gradient-to-t from-black/65 via-black/15 to-transparent" />
                  {showAnnotations && (
                    <div className="absolute inset-x-0 bottom-0 flex items-end justify-between gap-3 p-4">
                      <div className="min-w-0">
                        <div className="truncate text-sm font-semibold text-white">{sample.file_name}</div>
                        <div className="mt-1 text-[11px] uppercase tracking-[0.2em] text-white/70">
                          {sample.split === 'all' ? 'Preview sample' : sample.split}
                        </div>
                      </div>
                      <Badge className="rounded-full border-none bg-white/90 px-3 py-1 text-[10px] font-semibold uppercase tracking-[0.18em] text-foreground">
                        Open
                      </Badge>
                    </div>
                  )}
                </button>
                <div className="flex items-center justify-between gap-3 px-4 py-4">
                  <div className="min-w-0">
                    <div className="truncate text-sm font-semibold text-foreground">{sample.file_name}</div>
                    <div className="mt-1 text-xs text-muted-foreground">{sample.sample_id}</div>
                  </div>
                  <Button
                    variant={isSelected ? 'default' : 'outline'}
                    className="rounded-2xl"
                    onClick={() => toggleSampleSelection(sample.sample_id)}
                  >
                    {isSelected ? 'Selected' : 'Select'}
                  </Button>
                </div>
              </div>
            );
          })}
        </div>
      ) : (
        <div className="space-y-3">
          {filteredSamples.map((sample, index) => {
            const isSelected = selectedSampleIds.includes(sample.sample_id);
            return (
              <div key={sample.sample_id} className="flex flex-col gap-4 rounded-[1.7rem] border border-border/70 bg-card p-4 shadow-sm md:flex-row md:items-center">
                <button type="button" onClick={() => setActiveImageIndex(index)} className="h-24 w-full overflow-hidden rounded-[1.2rem] bg-muted/10 md:w-36">
                  <img src={sample.sourceUrl} alt={sample.file_name} className="h-full w-full object-cover" />
                </button>
                <div className="min-w-0 flex-1">
                  <div className="truncate text-lg font-semibold text-foreground">{sample.file_name}</div>
                  <div className="mt-1 text-sm text-muted-foreground">{sample.sample_id}</div>
                  <div className="mt-3 flex flex-wrap gap-2">
                    <Badge variant="outline" className="rounded-full px-3 py-1 capitalize">{sample.split === 'all' ? 'Preview' : sample.split}</Badge>
                    <Badge variant="outline" className="rounded-full px-3 py-1">{sample.mime_type ?? 'image/jpeg'}</Badge>
                  </div>
                </div>
                <div className="flex gap-2">
                  <Button variant="outline" className="rounded-2xl" onClick={() => setActiveImageIndex(index)}>
                    Open
                  </Button>
                  <Button variant={isSelected ? 'default' : 'outline'} className="rounded-2xl" onClick={() => toggleSampleSelection(sample.sample_id)}>
                    {isSelected ? 'Selected' : 'Select'}
                  </Button>
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );

  const renderDatasetSection = () => (
    <div className="grid gap-6 xl:grid-cols-[1.15fr_0.85fr]">
      <div className="rounded-[2rem] border border-border/70 bg-card p-6 shadow-sm">
        <div className="text-[10px] font-semibold uppercase tracking-[0.24em] text-muted-foreground">Dataset Metadata</div>
        <div className="mt-5 space-y-3">
          {[
            ['Name', selectedDataset.name],
            ['Workspace', selectedDataset.workspace],
            ['Project', selectedDataset.project],
            ['Version', `v${selectedDataset.version ?? selectedDataset.version_number ?? 'latest'}`],
            ['Type', selectedDataset.type ?? 'Dataset'],
            ['Image Count', formatCount(selectedDataset.image_count, 'images')],
            ['Class Count', formatCount(selectedDataset.class_count, 'classes')],
          ].map(([label, value]) => (
            <div key={label} className="flex items-center justify-between gap-5 rounded-[1.2rem] border bg-muted/10 px-4 py-3">
              <span className="text-sm text-muted-foreground">{label}</span>
              <span className="text-sm font-semibold text-foreground">{value}</span>
            </div>
          ))}
        </div>
      </div>
      <div className="rounded-[2rem] border border-border/70 bg-card p-6 shadow-sm">
        <div className="text-[10px] font-semibold uppercase tracking-[0.24em] text-muted-foreground">External Links</div>
        <div className="mt-4 space-y-3">
          <Button
            variant="outline"
            className="w-full justify-between rounded-2xl"
            onClick={() => window.open(selectedDataset.dataset_url, '_blank', 'noopener,noreferrer')}
          >
            Open Universe Page
            <ExternalLink className="h-4 w-4" />
          </Button>
          <div className="rounded-[1.5rem] border bg-muted/10 p-4 text-sm leading-7 text-muted-foreground">
            This section stays read-only inside Deplyze Studio and exposes the main dataset metadata you need before running samples in the app.
          </div>
        </div>
      </div>
    </div>
  );

  const renderAnalyticsSection = () => (
    <div className="grid gap-6 xl:grid-cols-2">
      <div className="rounded-[2rem] border border-border/70 bg-card p-6 shadow-sm">
        <div className="text-[10px] font-semibold uppercase tracking-[0.24em] text-muted-foreground">Coverage</div>
        <h4 className="mt-2 text-xl font-semibold tracking-tight text-foreground">Sample preview density</h4>
        <div className="mt-6 space-y-5">
          <div>
            <div className="mb-2 flex items-center justify-between text-sm">
              <span className="text-muted-foreground">Preview images loaded</span>
              <span className="font-semibold text-foreground">{datasetSamples.length}</span>
            </div>
            <div className="h-3 overflow-hidden rounded-full bg-muted">
              <div className="h-full rounded-full bg-primary" style={{ width: `${Math.min((datasetSamples.length / Math.max(selectedDataset.image_count || 1, 1)) * 100, 100)}%` }} />
            </div>
          </div>
          <div>
            <div className="mb-2 flex items-center justify-between text-sm">
              <span className="text-muted-foreground">Class breadth</span>
              <span className="font-semibold text-foreground">{selectedDataset.class_count ?? 0}</span>
            </div>
            <div className="h-3 overflow-hidden rounded-full bg-muted">
              <div className="h-full rounded-full bg-primary/70" style={{ width: `${Math.min(((selectedDataset.class_count ?? 0) / 20) * 100, 100)}%` }} />
            </div>
          </div>
        </div>
      </div>
      <div className="rounded-[2rem] border border-border/70 bg-card p-6 shadow-sm">
        <div className="text-[10px] font-semibold uppercase tracking-[0.24em] text-muted-foreground">Readiness</div>
        <h4 className="mt-2 text-xl font-semibold tracking-tight text-foreground">Inference workflow status</h4>
        <div className="mt-5 rounded-[1.5rem] border bg-muted/10 p-5">
          <div className="space-y-3 text-sm text-muted-foreground">
            <div className="flex items-center justify-between rounded-[1rem] bg-card px-4 py-3">
              <span>Active task is mapped</span>
              <span className="font-semibold text-foreground">{taskLabel}</span>
            </div>
            <div className="flex items-center justify-between rounded-[1rem] bg-card px-4 py-3">
              <span>Preview samples cached</span>
              <span className="font-semibold text-foreground">{datasetSamples.length > 0 ? 'Ready' : 'Waiting'}</span>
            </div>
            <div className="flex items-center justify-between rounded-[1rem] bg-card px-4 py-3">
              <span>Result dock</span>
              <span className="font-semibold text-foreground">{dockResults.length > 0 ? 'Populated' : 'Idle'}</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );

  const renderModelSection = () => (
    <div className="grid gap-6 xl:grid-cols-[0.95fr_1.05fr]">
      <div className="rounded-[2rem] border border-border/70 bg-card p-6 shadow-sm">
        <div className="text-[10px] font-semibold uppercase tracking-[0.24em] text-muted-foreground">Active Model</div>
        <h4 className="mt-2 text-xl font-semibold tracking-tight text-foreground">Current studio inference contract</h4>
        <div className="mt-5 space-y-3">
          <div className="rounded-[1.2rem] border bg-muted/10 px-4 py-4">
            <div className="text-[10px] uppercase tracking-[0.22em] text-muted-foreground">Task</div>
            <div className="mt-2 font-semibold text-foreground">{taskLabel}</div>
          </div>
          <div className="rounded-[1.2rem] border bg-muted/10 px-4 py-4">
            <div className="text-[10px] uppercase tracking-[0.22em] text-muted-foreground">Runtime</div>
            <div className="mt-2 font-semibold text-foreground">{runtimeLabel}</div>
          </div>
          <div className="rounded-[1.2rem] border bg-muted/10 px-4 py-4">
            <div className="text-[10px] uppercase tracking-[0.22em] text-muted-foreground">Model</div>
            <div className="mt-2 font-semibold text-foreground">{modelLabel}</div>
          </div>
        </div>
      </div>
      <div className="rounded-[2rem] border border-border/70 bg-card p-6 shadow-sm">
        <div className="text-[10px] font-semibold uppercase tracking-[0.24em] text-muted-foreground">Execution Notes</div>
        <div className="mt-4 rounded-[1.5rem] border bg-muted/10 p-5 text-sm leading-7 text-muted-foreground">
          Deplyze keeps the dataset in context and runs the active Studio configuration directly against preview samples. Use Overview for a curated run, or jump into Images when you want to control exactly which samples enter the batch.
        </div>
        <div className="mt-5 flex flex-wrap gap-3">
          <Button className="rounded-2xl" onClick={runCuratedBatch} disabled={runningInference || loadingDetail}>
            <Sparkles className="mr-2 h-4 w-4" />
            Run Curated Batch
          </Button>
          <Button variant="outline" className="rounded-2xl" onClick={() => setDatasetSection('images')}>
            Open Images
          </Button>
        </div>
      </div>
    </div>
  );

  const renderApiDocsSection = () => (
    <div className="grid gap-4 xl:grid-cols-2">
      {[
        { method: 'GET', path: '/api/roboflow/search?q={query}&api_key={key}', note: 'Search public Roboflow datasets.' },
        { method: 'GET', path: `/api/roboflow/dataset/${selectedDataset.workspace}/${selectedDataset.project}?api_key={key}`, note: 'Load dataset metadata and preview samples.' },
        { method: 'POST', path: '/api/roboflow/infer', note: 'Original backend dataset inference endpoint.' },
        { method: 'POST', path: '/api/detect/image', note: 'Used by the dataset workspace for per-sample backend detection.' },
      ].map((endpoint) => (
        <div key={endpoint.path} className="rounded-[1.8rem] border border-border/70 bg-card p-5 shadow-sm">
          <div className="flex items-center gap-2">
            <Badge variant="secondary" className="rounded-full px-3 py-1 text-[10px] font-semibold uppercase tracking-[0.22em]">
              {endpoint.method}
            </Badge>
            <span className="text-sm font-semibold text-foreground">{endpoint.path}</span>
          </div>
          <div className="mt-3 text-sm leading-6 text-muted-foreground">{endpoint.note}</div>
        </div>
      ))}
    </div>
  );

  const renderWorkspaceContent = () => {
    if (loadingDetail) {
      return (
        <div className="rounded-[2rem] border border-border/70 bg-card px-6 py-24 text-center text-sm text-muted-foreground">
          Loading dataset details and preview cache…
        </div>
      );
    }

    switch (datasetSection) {
      case 'images':
        return renderImagesSection();
      case 'dataset':
        return renderDatasetSection();
      case 'analytics':
        return renderAnalyticsSection();
      case 'model':
        return renderModelSection();
      case 'api-docs':
        return renderApiDocsSection();
      case 'overview':
      default:
        return renderOverviewSection();
    }
  };

  const renderWorkspace = () => (
    <div className="space-y-6">
      <div className="flex items-center justify-between gap-3">
        <Button variant="ghost" className="rounded-2xl px-0 text-muted-foreground hover:bg-transparent hover:text-foreground" onClick={closeWorkspace}>
          <ArrowLeft className="mr-2 h-4 w-4" />
          Back to results
        </Button>
        {selectedDataset?.dataset_url && (
          <Button
            variant="outline"
            className="rounded-2xl"
            onClick={() => window.open(selectedDataset.dataset_url, '_blank', 'noopener,noreferrer')}
          >
            Universe
            <ExternalLink className="ml-2 h-4 w-4" />
          </Button>
        )}
      </div>

      <div className="overflow-hidden rounded-[2rem] border border-border/70 bg-card shadow-[0_24px_80px_-40px_rgba(198,93,60,0.35)]">
        <div className="border-b border-border/70 bg-[radial-gradient(circle_at_top_left,_rgba(198,93,60,0.14),_transparent_28%),linear-gradient(180deg,rgba(255,255,255,0.98),rgba(251,246,243,0.94))] px-6 py-7 md:px-8">
          <div className="flex flex-col gap-6 xl:flex-row xl:items-end xl:justify-between">
            <div className="max-w-3xl">
              <div className="text-[10px] font-semibold uppercase tracking-[0.26em] text-primary">Dataset Workspace</div>
              <h2 className="mt-3 font-outfit text-3xl font-semibold tracking-tight text-foreground md:text-5xl">
                {selectedDataset.name}
              </h2>
              <div className="mt-3 flex flex-wrap items-center gap-2 text-sm text-muted-foreground">
                <span>{selectedDataset.workspace}/{selectedDataset.project}</span>
                <span>·</span>
                <span>{selectedDataset.type ?? 'Dataset'}</span>
                <span>·</span>
                <span>{formatCount(selectedDataset.image_count, 'images')}</span>
                <span>·</span>
                <span>{formatCount(selectedDataset.class_count, 'classes')}</span>
              </div>
            </div>
            <div className="flex flex-wrap gap-3">
              <Button className="rounded-2xl px-5" onClick={runCuratedBatch} disabled={runningInference || loadingDetail}>
                <Sparkles className="mr-2 h-4 w-4" />
                {runningInference ? 'Running…' : 'Run in Studio'}
              </Button>
              <Button variant="outline" className="rounded-2xl px-5" onClick={() => setDatasetSection('images')}>
                Select Samples
              </Button>
            </div>
          </div>
        </div>

        <div className="grid gap-6 p-4 md:p-6 xl:grid-cols-[240px_minmax(0,1fr)]">
          <div className="space-y-4">
            <div className="xl:hidden">
              <div className="flex gap-2 overflow-x-auto pb-2">
                {DATASET_SECTIONS.map((section) => {
                  const isActive = section.id === datasetSection;
                  return (
                    <button
                      key={section.id}
                      type="button"
                      onClick={() => setDatasetSection(section.id)}
                      className={`whitespace-nowrap rounded-full px-4 py-2 text-sm transition-colors ${
                        isActive ? 'bg-primary text-primary-foreground' : 'border border-border bg-card text-muted-foreground'
                      }`}
                    >
                      {section.label}
                    </button>
                  );
                })}
              </div>
            </div>
            <div className="hidden xl:block">
              <SectionNav activeSection={datasetSection} onChange={setDatasetSection} />
            </div>
            <div className="rounded-[1.75rem] border border-border/70 bg-card p-4 shadow-sm">
              <div className="text-[10px] font-semibold uppercase tracking-[0.24em] text-muted-foreground">Session</div>
              <div className="mt-3 space-y-3 text-sm text-muted-foreground">
                <div className="rounded-[1.1rem] border bg-muted/10 px-4 py-3">
                  {apiKey ? 'Browser Roboflow key active' : 'Backend fallback key active'}
                </div>
                <div className="rounded-[1.1rem] border bg-muted/10 px-4 py-3">
                  {selectedSampleIds.length} sample{selectedSampleIds.length === 1 ? '' : 's'} selected
                </div>
              </div>
            </div>
          </div>

          <div className="min-w-0 space-y-6">
            {renderWorkspaceContent()}
            {dockOpen && (
              <div className="overflow-hidden rounded-[2rem] border border-border/80 bg-card shadow-[0_24px_80px_-45px_rgba(15,23,42,0.28)]">
                <div
                  onMouseDown={startDockResize}
                  className="hidden h-3 cursor-ns-resize items-center justify-center border-b border-border bg-muted/40 md:flex"
                >
                  <div className="h-1.5 w-16 rounded-full bg-muted-foreground/30" />
                </div>

                <div className="border-b border-border/70 bg-[linear-gradient(180deg,rgba(255,255,255,0.98),rgba(248,244,241,0.96))] px-5 py-4">
                  <div className="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
                    <div>
                      <div className="text-[10px] font-semibold uppercase tracking-[0.26em] text-primary">Inference Results</div>
                      <h4 className="mt-1 text-xl font-semibold tracking-tight text-foreground">
                        {runningInference ? 'Running active studio task…' : 'Dataset results in context'}
                      </h4>
                      <div className="mt-2 flex flex-wrap gap-2 text-xs text-muted-foreground">
                        <Badge variant="outline" className="rounded-full px-3 py-1">{taskLabel}</Badge>
                        <Badge variant="outline" className="rounded-full px-3 py-1">{runtimeLabel}</Badge>
                        <Badge variant="outline" className="rounded-full px-3 py-1">{modelLabel}</Badge>
                      </div>
                    </div>
                    <div className="flex items-center gap-2">
                      <Button variant="outline" size="icon" className="rounded-2xl" onClick={() => setDockCollapsed((current) => !current)}>
                        {dockCollapsed ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />}
                      </Button>
                      <Button variant="outline" size="icon" className="rounded-2xl" onClick={() => setDockOpen(false)}>
                        <X className="h-4 w-4" />
                      </Button>
                    </div>
                  </div>
                </div>

                {!dockCollapsed && (
                  <div className="overflow-hidden" style={{ height: `${dockHeight}px` }}>
                    {runningInference ? (
                      <div className="flex h-full flex-col items-center justify-center gap-4 bg-muted/10 px-6 text-center">
                        <div className="h-10 w-10 animate-spin rounded-full border-2 border-primary border-t-transparent" />
                        <div>
                          <div className="font-semibold text-foreground">Analyzing selected dataset samples</div>
                          <div className="mt-1 text-sm text-muted-foreground">Results stay anchored beneath the current dataset section while inference runs.</div>
                        </div>
                      </div>
                    ) : (
                      <div className="grid h-full gap-0 lg:grid-cols-[300px_minmax(0,1fr)]">
                        <div className="border-b border-border/70 bg-muted/10 lg:border-b-0 lg:border-r">
                          <div className="flex items-center justify-between border-b border-border/70 px-5 py-4">
                            <div>
                              <div className="text-[10px] font-semibold uppercase tracking-[0.24em] text-muted-foreground">Queue</div>
                              <div className="mt-1 text-sm font-semibold text-foreground">{dockResults.length} sample{dockResults.length === 1 ? '' : 's'}</div>
                            </div>
                            <Badge variant="secondary" className="rounded-full px-3 py-1">
                              {activeResultIndex + 1}/{Math.max(dockResults.length, 1)}
                            </Badge>
                          </div>
                          <div className="max-h-full space-y-2 overflow-y-auto p-3">
                            {dockResults.map((entry, index) => (
                              <button
                                key={entry.sample.sample_id}
                                type="button"
                                onClick={() => setActiveResultIndex(index)}
                                className={`flex w-full items-center gap-3 rounded-[1.3rem] border p-3 text-left transition-all ${
                                  index === activeResultIndex ? 'border-primary/30 bg-primary/10 shadow-sm' : 'bg-card hover:border-primary/20'
                                }`}
                              >
                                <div className="h-16 w-16 overflow-hidden rounded-[1rem] bg-muted/10">
                                  <img src={entry.sample.sourceUrl} alt={entry.sample.file_name} className="h-full w-full object-cover" />
                                </div>
                                <div className="min-w-0 flex-1">
                                  <div className="truncate text-sm font-semibold text-foreground">{entry.sample.file_name}</div>
                                  <div className="mt-1 text-xs text-muted-foreground">{getResultCount(entry.result)} results</div>
                                </div>
                              </button>
                            ))}
                          </div>
                        </div>

                        <div className="grid h-full gap-0 xl:grid-cols-[1.1fr_0.9fr]">
                          <div className="border-b border-border/70 bg-slate-950 p-4 xl:border-b-0 xl:border-r">
                            {activeDockEntry ? (
                              <div className="flex h-full flex-col">
                                <div className="mb-4 flex items-center justify-between text-white/80">
                                  <div>
                                    <div className="text-sm font-semibold">{activeDockEntry.sample.file_name}</div>
                                    <div className="mt-1 text-xs uppercase tracking-[0.2em] text-white/45">{inferSampleSplit(activeDockEntry.sample)}</div>
                                  </div>
                                  <Badge className="rounded-full border-none bg-white/12 px-3 py-1 text-white">
                                    {getResultCount(activeDockEntry.result)} results
                                  </Badge>
                                </div>
                                <div className="flex-1 overflow-hidden rounded-[1.6rem] border border-white/10 bg-black">
                                  <img src={activeDockEntry.sample.sourceUrl} alt={activeDockEntry.sample.file_name} className="h-full w-full object-contain" />
                                </div>
                              </div>
                            ) : (
                              <div className="flex h-full items-center justify-center text-center text-sm text-white/60">
                                Run inference to populate the comparison panel.
                              </div>
                            )}
                          </div>

                          <div className="h-full overflow-y-auto bg-card p-4">
                            <div className="mb-4 grid gap-3 sm:grid-cols-3">
                              <div className="rounded-[1.2rem] border bg-muted/10 px-4 py-4">
                                <div className="text-[10px] uppercase tracking-[0.22em] text-muted-foreground">Task</div>
                                <div className="mt-2 text-sm font-semibold text-foreground">{taskLabel}</div>
                              </div>
                              <div className="rounded-[1.2rem] border bg-muted/10 px-4 py-4">
                                <div className="text-[10px] uppercase tracking-[0.22em] text-muted-foreground">Runtime</div>
                                <div className="mt-2 text-sm font-semibold text-foreground">{runtimeLabel}</div>
                              </div>
                              <div className="rounded-[1.2rem] border bg-muted/10 px-4 py-4">
                                <div className="text-[10px] uppercase tracking-[0.22em] text-muted-foreground">Model</div>
                                <div className="mt-2 truncate text-sm font-semibold text-foreground">{modelLabel}</div>
                              </div>
                            </div>
                            {renderResultDetails(activeDockResult, hoveredIndex, setHoveredIndex)}
                          </div>
                        </div>
                      </div>
                    )}
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );

  return (
    <div className="relative space-y-6 pb-8">
      {workspaceMode === 'discovery' ? renderDiscovery() : renderWorkspace()}

      <SampleModal
        sample={activeImageSample}
        isOpen={activeImageIndex !== null}
        onOpenChange={(open) => {
          if (!open) setActiveImageIndex(null);
        }}
        onRunSample={(sample) => runSamplesInStudio([sample], 'Image Modal')}
        onNext={() => setActiveImageIndex((current) => clamp((current ?? 0) + 1, 0, Math.max(filteredSamples.length - 1, 0)))}
        onPrevious={() => setActiveImageIndex((current) => clamp((current ?? 0) - 1, 0, Math.max(filteredSamples.length - 1, 0)))}
        hasNext={activeImageIndex !== null && activeImageIndex < filteredSamples.length - 1}
        hasPrevious={activeImageIndex !== null && activeImageIndex > 0}
        isSelected={activeImageSample ? selectedSampleIds.includes(activeImageSample.sample_id) : false}
        onToggleSelection={toggleSampleSelection}
        selectedDataset={selectedDataset}
      />
    </div>
  );
}
