import React, { useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card';
import { Button } from './ui/button';
import { Badge } from './ui/badge';
import { Progress } from './ui/progress';
import { Upload, Download, Package, BarChart3, Image as ImageIcon, Search, Activity, Target } from 'lucide-react';

const BatchTab = ({ 
  batchInputRef, 
  batchFiles, 
  batchProcessing, 
  batchResults, 
  handleBatchUpload, 
  downloadBatchResults,
  getColorForClass,
  backendUrl
}) => {
  const [selectedImageIndex, setSelectedImageIndex] = useState(0);

  return (
    <div className="grid lg:grid-cols-3 gap-6 h-[calc(100vh-140px)]">
      {/* Left Workspace: Batch Overview & Gallery (2/3) */}
      <div className="lg:col-span-2 flex flex-col gap-6 min-h-0 overflow-hidden">
        {/* Upload & Stats Card */}
        <Card className="bg-card border-border shadow-sm shrink-0">
          <CardHeader className="py-3 px-4 flex flex-row items-center justify-between bg-muted/30 border-b">
            <div>
              <CardTitle className="text-sm font-bold flex items-center gap-2">
                <Package className="w-4 h-4 text-primary" />
                Batch Analytics
              </CardTitle>
            </div>
            {batchResults && (
              <Button 
                variant="outline" 
                size="sm" 
                onClick={downloadBatchResults}
                className="h-8 text-[11px] gap-2"
              >
                <Download className="w-3 h-3" />
                Export ZIP
              </Button>
            )}
          </CardHeader>
          <CardContent className="p-4">
            {!batchResults && !batchProcessing ? (
              <div 
                className="border-2 border-dashed border-border rounded-lg p-8 text-center hover:border-primary transition-all cursor-pointer bg-muted/5 group"
                onClick={() => batchInputRef.current?.click()}
              >
                <Upload className="w-10 h-10 mx-auto text-muted-foreground mb-3 group-hover:text-primary transition-colors" />
                <p className="text-sm font-medium">Click to select multiple images</p>
                <p className="text-xs text-muted-foreground mt-1">PNG, JPG, GIF - Max 50 files</p>
              </div>
            ) : batchProcessing ? (
              <div className="space-y-4 py-4">
                <div className="flex items-center justify-between">
                  <span className="text-xs font-medium flex items-center gap-2">
                    <div className="w-3 h-3 animate-spin rounded-full border-2 border-primary border-t-transparent"></div>
                    Synthesizing Detections...
                  </span>
                  <span className="text-[10px] text-muted-foreground uppercase font-bold">60% Complete</span>
                </div>
                <Progress value={60} className="h-1.5" />
              </div>
            ) : (
              <div className="grid grid-cols-4 gap-4">
                <div className="bg-muted/30 p-2 rounded-md border border-border/50">
                  <div className="text-[10px] uppercase text-muted-foreground font-bold mb-1">Total</div>
                  <div className="text-xl font-bold text-primary">{batchResults.total_images}</div>
                </div>
                <div className="bg-muted/30 p-2 rounded-md border border-border/50">
                  <div className="text-[10px] uppercase text-muted-foreground font-bold mb-1">Found</div>
                  <div className="text-xl font-bold text-primary">{batchResults.total_detections}</div>
                </div>
                <div className="bg-muted/30 p-2 rounded-md border border-border/50">
                  <div className="text-[10px] uppercase text-muted-foreground font-bold mb-1">Time</div>
                  <div className="text-xl font-bold text-primary">{batchResults.processing_time?.toFixed(1)}s</div>
                </div>
                <div className="bg-muted/30 p-2 rounded-md border border-border/50">
                  <div className="text-[10px] uppercase text-muted-foreground font-bold mb-1">Avg/Img</div>
                  <div className="text-xl font-bold text-primary">{(batchResults.processing_time / batchResults.total_images).toFixed(1)}s</div>
                </div>
              </div>
            )}
            
            <input
              type="file"
              ref={batchInputRef}
              onChange={handleBatchUpload}
              accept="image/*"
              multiple
              className="hidden"
            />
          </CardContent>
        </Card>

        {/* Gallery Grid */}
        <div className="flex-1 overflow-y-auto pr-2 scrollbar-thin">
          {batchResults ? (
            <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
              {batchResults?.batch_results?.map((result, index) => (
                <div 
                  key={index}
                  onClick={() => setSelectedImageIndex(index)}
                  className={`relative aspect-square rounded-lg overflow-hidden border-2 cursor-pointer transition-all hover:scale-[1.02] ${selectedImageIndex === index ? 'border-primary shadow-lg ring-2 ring-primary/20' : 'border-border grayscale-[0.5] hover:grayscale-0'}`}
                >
                  <img src={result.annotated_path ? `${backendUrl}${result.annotated_path}` : `${backendUrl}${result.original_path}`} alt={result.index} className="w-full h-full object-cover" />
                  <div className="absolute top-2 right-2">
                    <Badge className="bg-black/60 backdrop-blur-md text-[10px]">
                      {result.detections?.length || 0} hits
                    </Badge>
                  </div>
                  {selectedImageIndex === index && (
                    <div className="absolute inset-x-0 bottom-0 bg-primary/90 text-primary-foreground text-[10px] font-bold text-center py-1">
                      SELECTED
                    </div>
                  )}
                </div>
              ))}
            </div>
          ) : (
            <div className="h-full flex flex-col items-center justify-center border-2 border-dashed border-border rounded-xl bg-muted/5 opacity-50">
              <ImageIcon className="w-12 h-12 text-muted-foreground mb-4" />
              <p className="text-sm font-medium">Batch Workspace</p>
              <p className="text-xs text-muted-foreground mt-1">Upload images to begin processing</p>
            </div>
          )}
        </div>
      </div>

      {/* Right Workspace: Batch Inspector (1/3) */}
      <div className="flex flex-col gap-6 min-h-0">
        <Card className="bg-card border-border shadow-lg flex flex-col h-full overflow-hidden border-2 border-primary/10">
          <CardHeader className="py-3 flex-row items-center justify-between border-b bg-muted/10 shrink-0">
            <div>
              <CardTitle className="text-sm font-bold flex items-center gap-2">
                <Target className="w-4 h-4 text-primary" />
                Detections Inspector
              </CardTitle>
              <CardDescription className="text-[11px]">Details for item #{selectedImageIndex + 1}</CardDescription>
            </div>
          </CardHeader>
          <CardContent className="p-0 flex flex-col flex-1 min-h-0">
            {batchResults && batchResults.batch_results[selectedImageIndex] ? (
              <div className="flex flex-col h-full overflow-hidden">
                {/* Image Snapshot */}
                <div className="aspect-video bg-muted/20 relative group overflow-hidden border-b">
                  <img 
                    src={batchResults.batch_results[selectedImageIndex].annotated_path ? `${backendUrl}${batchResults.batch_results[selectedImageIndex].annotated_path}` : `${backendUrl}${batchResults.batch_results[selectedImageIndex].original_path}`} 
                    alt="Selected Preview" 
                    className="w-full h-full object-contain" 
                  />
                  <div className="absolute inset-0 bg-black/40 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center">
                    <Search className="text-white w-6 h-6" />
                  </div>
                </div>

                {/* Sub-inspector scroll area */}
                <div className="flex-1 overflow-y-auto p-3 space-y-2 scrollbar-thin">
                  <div className="flex justify-between items-center mb-1">
                    <span className="text-[10px] uppercase font-bold text-muted-foreground tracking-widest">Detections List</span>
                    <Badge variant="outline" className="text-[10px]">{batchResults.batch_results[selectedImageIndex].detections?.length || 0} Results</Badge>
                  </div>
                  
                  {batchResults?.batch_results?.[selectedImageIndex]?.detections && batchResults.batch_results[selectedImageIndex].detections.length > 0 ? (
                    batchResults.batch_results[selectedImageIndex].detections.map((detection, idx) => (
                      <div key={idx} className="flex items-center justify-between p-2 rounded bg-muted/40 hover:bg-muted/60 border border-transparent transition-all">
                        <div className="flex items-center gap-3 overflow-hidden">
                          <div className="w-2 h-2 rounded-full shrink-0" style={{ backgroundColor: getColorForClass(idx) }}></div>
                          <span className="text-xs font-semibold truncate capitalize">{detection.class_name}</span>
                        </div>
                        <Badge variant="secondary" className="font-mono text-[9px] py-0 px-1.5 h-4 bg-primary/10 text-primary border-none">
                          {(detection.confidence * 100).toFixed(1)}%
                        </Badge>
                      </div>
                    ))
                  ) : (
                    <div className="text-center py-10 opacity-30 italic text-xs">No detections for this image</div>
                  )}
                </div>

                {/* Meta details */}
                <div className="p-4 bg-muted/10 border-t space-y-2">
                  <div className="flex justify-between items-center text-[10px]">
                    <span className="text-muted-foreground uppercase font-bold">Image Info</span>
                    <span className="text-foreground font-mono">ID: {batchResults.batch_results[selectedImageIndex].index}</span>
                  </div>
                  <div className="bg-card rounded border border-border/50 p-2 text-[11px] text-muted-foreground truncate">
                    {batchResults.batch_results[selectedImageIndex].original_path.split('/').pop()}
                  </div>
                </div>
              </div>
            ) : (
              <div className="h-full flex flex-col items-center justify-center py-20 opacity-30 text-center px-10">
                <Search className="w-12 h-12 mb-4" />
                <h4 className="text-sm font-bold">No Image Selected</h4>
                <p className="text-xs mt-1">Click an image from the gallery to inspect detection details</p>
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  );
};

export default BatchTab;