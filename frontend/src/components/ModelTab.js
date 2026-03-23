import React from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card';
import { Button } from './ui/button';
import { Badge } from './ui/badge';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from './ui/select';
import { Upload, Layers, Zap, CheckCircle, AlertCircle, Cpu, ShieldCheck, Database, Sliders, Activity } from 'lucide-react';

const ModelTab = ({ 
  modelInfo, 
  selectedModel, 
  modelUploading, 
  modelInputRef, 
  switchModel, 
  handleCustomModelUpload 
}) => {
  return (
    <div className="grid lg:grid-cols-3 gap-6 h-[calc(100vh-140px)] overflow-hidden">
      {/* Primary Workspace (2/3) */}
      <div className="lg:col-span-2 flex flex-col gap-6 min-h-0 overflow-y-auto pr-2 scrollbar-thin">
        {/* Active Model Stats Card */}
        <Card className="bg-card border-border shadow-sm shrink-0 border-2 border-primary/5">
          <CardHeader className="py-4 bg-muted/30 border-b">
            <CardTitle className="text-sm flex items-center justify-between">
              <span className="flex items-center gap-2 uppercase tracking-widest text-[11px] font-black text-muted-foreground">
                <Cpu className="w-4 h-4 text-primary" />
                Active Inference Engine
              </span>
              <Badge variant="outline" className="bg-primary/5 text-primary border-primary/20">PRODUCTION READY</Badge>
            </CardTitle>
          </CardHeader>
          <CardContent className="p-6">
            {modelInfo && (
              <div className="space-y-6">
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  <div className="bg-muted/10 p-3 rounded-lg border border-border/40 hover:border-primary/20 transition-all">
                    <div className="text-[10px] uppercase font-bold text-muted-foreground mb-1 tracking-tighter">Model ID</div>
                    <div className="text-lg font-mono font-bold text-primary truncate" title={modelInfo.active_model}>{modelInfo.active_model}</div>
                  </div>
                  <div className="bg-muted/10 p-3 rounded-lg border border-border/40 hover:border-primary/20 transition-all">
                    <div className="text-[10px] uppercase font-bold text-muted-foreground mb-1 tracking-tighter">Classes</div>
                    <div className="text-lg font-mono font-bold text-primary">{modelInfo.available_classes?.length || 0}</div>
                  </div>
                  <div className="bg-muted/10 p-3 rounded-lg border border-border/40 hover:border-primary/20 transition-all">
                    <div className="text-[10px] uppercase font-bold text-muted-foreground mb-1 tracking-tighter">Threshold</div>
                    <div className="text-lg font-mono font-bold text-primary">{(modelInfo.confidence_threshold * 100).toFixed(0)}%</div>
                  </div>
                  <div className="bg-muted/10 p-3 rounded-lg border border-border/40 hover:border-primary/20 transition-all">
                    <div className="text-[10px] uppercase font-bold text-muted-foreground mb-1 tracking-tighter">Input</div>
                    <div className="text-lg font-mono font-bold text-primary">{modelInfo.max_inference_size}px</div>
                  </div>
                </div>

                {/* Class Tags Grid */}
                <div className="space-y-3 pt-4 border-t border-border/50">
                  <div className="flex items-center justify-between">
                    <h4 className="text-[11px] uppercase font-black text-muted-foreground tracking-widest flex items-center gap-2">
                      <ShieldCheck className="w-3.5 h-3.5" />
                      Class Vocabulary
                    </h4>
                  </div>
                  <div className="flex flex-wrap gap-1.5">
                    {modelInfo.available_classes?.map((className, index) => (
                      <Badge key={index} variant="secondary" className="px-2 py-0.5 bg-muted/40 text-[10px] text-foreground/80 hover:bg-primary/10 hover:text-primary transition-colors border-none">
                        {className}
                      </Badge>
                    ))}
                  </div>
                </div>
              </div>
            )}
          </CardContent>
        </Card>

        {/* Upload Interface Card */}
        <Card className="bg-card border-border shadow-sm border-2 border-primary/5">
          <CardHeader className="py-4 bg-muted/30 border-b">
            <CardTitle className="text-sm flex items-center gap-2">
              <Upload className="w-4 h-4 text-primary" />
              Weight Deployment
            </CardTitle>
            <CardDescription className="text-[11px]">Deploy custom YOLOv11/v10/v8 weights</CardDescription>
          </CardHeader>
          <CardContent className="p-6">
            <div className="grid md:grid-cols-2 gap-6 items-center">
              <div 
                className="border-2 border-dashed border-border rounded-xl p-10 text-center hover:border-primary/50 transition-all cursor-pointer bg-muted/5 group"
                onClick={() => modelInputRef.current?.click()}
              >
                <div className="p-4 bg-primary/10 rounded-full w-fit mx-auto mb-4 group-hover:scale-110 transition-transform">
                  <Upload className="w-8 h-8 text-primary" />
                </div>
                <p className="text-sm font-bold">Select .pt or .onnx Weights</p>
                <p className="text-[10px] text-muted-foreground mt-2 uppercase tracking-widest">Maximum File Size 250MB</p>
              </div>

              <div className="space-y-4">
                <div className="p-4 bg-amber-500/5 border border-amber-500/20 rounded-xl space-y-3">
                  <div className="flex gap-2 items-center text-amber-700 font-bold text-xs uppercase tracking-tighter">
                    <AlertCircle className="w-4 h-4" />
                    Deployment Requirements
                  </div>
                  <ul className="text-[11px] space-y-2 text-muted-foreground font-medium">
                    <li className="flex gap-2 items-start opacity-80"><div className="w-1 h-1 rounded-full bg-amber-500 mt-1.5 shrink-0" /> Supported: YOLOv8, v10, v11 format</li>
                    <li className="flex gap-2 items-start opacity-80"><div className="w-1 h-1 rounded-full bg-amber-500 mt-1.5 shrink-0" /> Must include classification metadata</li>
                    <li className="flex gap-2 items-start opacity-80"><div className="w-1 h-1 rounded-full bg-amber-500 mt-1.5 shrink-0" /> CPU optimized builds recommended</li>
                  </ul>
                </div>

                <Button 
                  onClick={() => modelInputRef.current?.click()}
                  disabled={modelUploading}
                  className="w-full h-11 bg-primary text-primary-foreground font-bold shadow-lg shadow-primary/20 hover:scale-[1.01] active:scale-[0.99] transition-transform"
                >
                  {modelUploading ? 'VALIDATING WEIGHTS...' : 'DEPLOY CUSTOM MODEL'}
                </Button>
              </div>
            </div>
            
            <input
              type="file"
              ref={modelInputRef}
              onChange={handleCustomModelUpload}
              accept=".pt,.onnx"
              className="hidden"
            />
          </CardContent>
        </Card>
      </div>

      {/* Right Sidebar (1/3) */}
      <div className="flex flex-col gap-6 min-h-0">
        <Card className="bg-card border-border shadow-lg flex flex-col h-full overflow-hidden border-2 border-primary/10">
          <CardHeader className="py-4 flex-row items-center justify-between border-b bg-muted/10 shrink-0">
            <div>
              <CardTitle className="text-sm font-bold flex items-center gap-2 uppercase tracking-tighter">
                <Database className="w-4 h-4 text-primary" />
                Registry
              </CardTitle>
              <CardDescription className="text-[10px]">Loaded in-memory models</CardDescription>
            </div>
          </CardHeader>
          <CardContent className="p-0 flex flex-col flex-1 min-h-0">
            <div className="p-4 bg-muted/5 border-b shrink-0">
              <label className="text-[10px] uppercase font-black text-muted-foreground tracking-widest mb-2 block">Quick Switch</label>
              <Select value={selectedModel} onValueChange={switchModel}>
                <SelectTrigger className="h-10 bg-card border-border border-2">
                  <SelectValue placeholder="Select from registry" />
                </SelectTrigger>
                <SelectContent>
                  {modelInfo?.available_models && Object.entries(modelInfo.available_models).map(([modelName, modelData]) => (
                    <SelectItem key={modelName} value={modelName}>
                      <div className="flex items-center gap-2">
                        <span>{modelName}</span>
                        {modelData.is_active && <CheckCircle className="w-3 h-3 text-primary" />}
                      </div>
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            {/* List of models */}
            <div className="flex-1 overflow-y-auto p-4 space-y-3 scrollbar-thin">
              {modelInfo?.available_models && Object.entries(modelInfo.available_models).map(([modelName, modelData]) => (
                <div 
                  key={modelName} 
                  onClick={() => switchModel(modelName)}
                  className={`p-3 rounded-lg border transition-all cursor-pointer group ${
                    modelData.is_active ? 'bg-primary/5 border-primary/50 shadow-inner ring-1 ring-primary/20' : 'bg-muted/10 border-border hover:border-primary/30'
                  }`}
                >
                  <div className="flex items-center justify-between mb-2">
                    <span className={`text-xs font-bold font-mono tracking-tight ${modelData.is_active ? 'text-primary' : 'text-foreground'}`}>
                      {modelName}
                    </span>
                    {modelData.is_active && <Badge className="text-[9px] h-4 py-0 leading-none">ACTIVE</Badge>}
                  </div>
                  
                  <div className="grid grid-cols-2 gap-2 mt-3">
                    <div className="text-[9px] flex items-center gap-1.5 opacity-60">
                      <Sliders className="w-3 h-3" /> {modelData.classes?.length || 0} classes
                    </div>
                    {modelData.loaded_at && (
                      <div className="text-[9px] flex items-center gap-1.5 opacity-60">
                        <Activity className="w-3 h-3" /> Loaded: {new Date(modelData.loaded_at * 1000).toLocaleDateString()}
                      </div>
                    )}
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
};

export default ModelTab;