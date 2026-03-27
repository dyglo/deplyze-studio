import React from 'react';
import { Activity, Download, Target, Zap } from 'lucide-react';
import { Badge } from './ui/badge';
import { Button } from './ui/button';

function renderInspectorBody(result, hoveredIndex, setHoveredIndex) {
  if (!result) {
    return <div className="h-full flex items-center justify-center p-8 text-center text-muted-foreground text-sm">Choose a task, load an image source, and run inference to inspect results.</div>;
  }

  if (result.task === 'classify') {
    return (
      <div className="flex flex-col h-full">
        <div className="px-4 py-3 border-b bg-card shrink-0 flex items-center justify-between">
          <span className="text-sm font-medium text-foreground">{result.topPrediction?.className ?? 'No prediction'}</span>
          <Badge variant="outline" className="text-xs font-mono">{result.inferenceTimeMs?.toFixed?.(0) ?? 0}ms</Badge>
        </div>
        <div className="flex-1 overflow-y-auto p-2 space-y-1">
          {(result.predictions ?? []).map((prediction, index) => (
            <div key={`${prediction.classId}-${index}`} className="flex items-center justify-between px-3 py-2 rounded hover:bg-muted transition-all">
              <div>
                <div className="text-sm font-medium">{prediction.className}</div>
                <div className="text-[10px] uppercase tracking-wider text-muted-foreground">Rank #{index + 1}</div>
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
    <div className="flex flex-col h-full">
      <div className="px-4 py-3 border-b bg-card shrink-0 flex items-center justify-between">
        <span className="text-sm font-medium text-foreground">{result.detections?.length ?? 0} results</span>
        <Badge variant="outline" className="text-xs font-mono">{result.inferenceTimeMs?.toFixed?.(0) ?? 0}ms</Badge>
      </div>
      <div className="flex-1 overflow-y-auto p-2 space-y-1">
        {(result.detections?.length ?? 0) > 0 ? result.detections.map((item, index) => (
          <div
            key={item.id ?? index}
            onMouseEnter={() => setHoveredIndex?.(index)}
            onMouseLeave={() => setHoveredIndex?.(null)}
            onClick={() => setHoveredIndex?.(index === hoveredIndex ? null : index)}
            className={`px-3 py-3 rounded cursor-pointer transition-all ${index === hoveredIndex ? 'bg-primary/10 shadow-inner' : 'hover:bg-muted'}`}
          >
            <div className="flex items-center justify-between gap-3">
              <div className="space-y-1 overflow-hidden">
                <div className="text-sm font-medium truncate">{item.className}</div>
                <div className="text-[10px] uppercase tracking-widest text-muted-foreground">
                  {result.task === 'pose' && `${(item.keypoints ?? []).filter((point) => point.confidence >= 0.35).length} visible keypoints`}
                  {result.task === 'segment' && `mask ${((item.maskCoverage ?? 0) * 100).toFixed(1)}% coverage`}
                  {result.task === 'obb' && `${(((item.angle ?? 0) * 180) / Math.PI).toFixed(1)}° rotation`}
                  {result.task === 'detect' && `${item.bbox?.width?.toFixed?.(0) ?? 0}×${item.bbox?.height?.toFixed?.(0) ?? 0}`}
                </div>
              </div>
              <Badge variant="secondary" className="font-mono text-xs">{((item.confidence ?? 0) * 100).toFixed(1)}%</Badge>
            </div>
            {item.trackId !== undefined && item.trackId !== null && (
              <div className="mt-2 text-[10px] font-semibold uppercase tracking-wider text-primary">Track ID #{item.trackId}</div>
            )}
          </div>
        )) : (
          <div className="p-6 text-center text-sm text-muted-foreground">No results detected above the current score threshold.</div>
        )}
      </div>
    </div>
  );
}

export default function StudioInspector({
  task,
  runtimeLabel,
  modelLabel,
  result,
  hoveredIndex,
  setHoveredIndex,
  isRunning,
  footerNote,
  onDownload,
  disableDownload,
  statCards = [],
}) {
  return (
    <div className="lg:col-span-1 border rounded-lg bg-card shadow-sm flex flex-col h-full lg:max-h-[min(800px,calc(100vh-12rem))] min-h-[500px]">
      <div className="p-4 border-b shrink-0 bg-muted/10">
        <h3 className="flex items-center gap-2 font-semibold text-foreground"><Target className="w-4 h-4" />Inspector</h3>
      </div>

      <div className="grid grid-cols-2 gap-3 p-4 border-b bg-card/50">
        <div className="rounded-lg border bg-muted/10 p-3">
          <div className="text-[10px] uppercase tracking-widest text-muted-foreground mb-1">Task</div>
          <div className="font-semibold">{task}</div>
        </div>
        <div className="rounded-lg border bg-muted/10 p-3">
          <div className="text-[10px] uppercase tracking-widest text-muted-foreground mb-1">Runtime</div>
          <div className="font-semibold">{runtimeLabel}</div>
        </div>
        <div className="rounded-lg border bg-muted/10 p-3">
          <div className="text-[10px] uppercase tracking-widest text-muted-foreground mb-1">Results</div>
          <div className="font-semibold">{task === 'Image Classification' ? result?.predictions?.length ?? 0 : result?.detections?.length ?? 0}</div>
        </div>
        <div className="rounded-lg border bg-muted/10 p-3">
          <div className="text-[10px] uppercase tracking-widest text-muted-foreground mb-1">Model</div>
          <div className="font-semibold text-sm truncate">{modelLabel}</div>
        </div>
      </div>

      {statCards.length > 0 && (
        <div className="grid grid-cols-2 gap-3 p-4 border-b bg-muted/5">
          {statCards.map((card) => (
            <div key={card.label} className="rounded-lg border bg-card p-3">
              <div className="text-[10px] uppercase tracking-widest text-muted-foreground mb-1">{card.label}</div>
              <div className="font-semibold text-sm">{card.value}</div>
            </div>
          ))}
        </div>
      )}

      <div className="flex-1 overflow-y-auto p-0 flex flex-col">
        {isRunning ? (
          <div className="flex flex-col items-center justify-center h-full p-8 space-y-4">
            <div className="animate-spin rounded-full w-8 h-8 border-2 border-primary border-t-transparent"></div>
            <span className="text-sm text-secondary-foreground">Analyzing image...</span>
          </div>
        ) : renderInspectorBody(result, hoveredIndex, setHoveredIndex)}
      </div>

      <div className="p-4 border-t bg-muted/5 space-y-3">
        {footerNote && (
          <div className="rounded-lg border bg-card p-3 text-xs text-muted-foreground space-y-1">
            <div className="font-semibold text-foreground flex items-center gap-2"><Zap className="w-3.5 h-3.5 text-primary" />Session</div>
            {footerNote}
          </div>
        )}
        {onDownload && (
          <Button onClick={onDownload} variant="default" className="w-full text-sm" disabled={disableDownload}>
            <Download className="w-4 h-4 mr-2" />
            Download Result
          </Button>
        )}
      </div>
    </div>
  );
}
