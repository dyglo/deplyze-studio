import React from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card';
import { Button } from './ui/button';
import { Badge } from './ui/badge';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from './ui/select';
import { Upload, Layers, Zap, CheckCircle, AlertCircle } from 'lucide-react';

const ModelTab = ({ 
  modelInfo, 
  selectedModel, 
  modelUploading, 
  modelInputRef, 
  switchModel, 
  handleCustomModelUpload 
}) => {
  return (
    <div className="space-y-6">
      {/* Active Model Info */}
      <Card className="bg-white/80 backdrop-blur-sm border-slate-200">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Layers className="w-5 h-5" />
            Active Model Information
          </CardTitle>
          <CardDescription>
            Current model configuration and performance details
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          {modelInfo && (
            <>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div className="text-center p-3 bg-slate-50 rounded-lg">
                  <div className="text-lg font-bold text-blue-600">{modelInfo.active_model}</div>
                  <div className="text-sm text-slate-600">Active Model</div>
                </div>
                <div className="text-center p-3 bg-slate-50 rounded-lg">
                  <div className="text-lg font-bold text-green-600">{modelInfo.available_classes?.length || 0}</div>
                  <div className="text-sm text-slate-600">Classes</div>
                </div>
                <div className="text-center p-3 bg-slate-50 rounded-lg">
                  <div className="text-lg font-bold text-purple-600">{(modelInfo.confidence_threshold * 100).toFixed(0)}%</div>
                  <div className="text-sm text-slate-600">Confidence</div>
                </div>
                <div className="text-center p-3 bg-slate-50 rounded-lg">
                  <div className="text-lg font-bold text-orange-600">{modelInfo.max_inference_size}px</div>
                  <div className="text-sm text-slate-600">Input Size</div>
                </div>
              </div>

              {/* Model Classes Preview */}
              <div className="space-y-2">
                <h4 className="text-sm font-medium text-slate-700">Detectable Objects</h4>
                <div className="flex flex-wrap gap-1 max-h-32 overflow-y-auto">
                  {modelInfo.available_classes?.slice(0, 20).map((className, index) => (
                    <Badge key={index} variant="outline" className="text-xs">
                      {className}
                    </Badge>
                  ))}
                  {modelInfo.available_classes?.length > 20 && (
                    <Badge variant="secondary" className="text-xs">
                      +{modelInfo.available_classes.length - 20} more
                    </Badge>
                  )}
                </div>
              </div>
            </>
          )}
        </CardContent>
      </Card>

      {/* Model Selection */}
      <Card className="bg-white/80 backdrop-blur-sm border-slate-200">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Zap className="w-5 h-5" />
            Model Selection
          </CardTitle>
          <CardDescription>
            Switch between available models
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          {modelInfo?.available_models && (
            <>
              <div className="space-y-2">
                <label className="text-sm font-medium">Available Models</label>
                <Select value={selectedModel} onValueChange={switchModel}>
                  <SelectTrigger>
                    <SelectValue placeholder="Select a model" />
                  </SelectTrigger>
                  <SelectContent>
                    {Object.entries(modelInfo.available_models).map(([modelName, modelData]) => (
                      <SelectItem key={modelName} value={modelName}>
                        <div className="flex items-center gap-2">
                          <span>{modelName}</span>
                          {modelData.is_active && <CheckCircle className="w-4 h-4 text-green-500" />}
                          <Badge variant="outline" className="ml-auto">
                            {modelData.classes?.length || 0} classes
                          </Badge>
                        </div>
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              {/* Model Details */}
              <div className="space-y-3">
                {Object.entries(modelInfo.available_models).map(([modelName, modelData]) => (
                  <div 
                    key={modelName} 
                    className={`p-3 rounded-lg border ${
                      modelData.is_active ? 'bg-green-50 border-green-200' : 'bg-slate-50'
                    }`}
                  >
                    <div className="flex items-center justify-between mb-2">
                      <div className="flex items-center gap-2">
                        <span className="font-medium">{modelName}</span>
                        {modelData.is_active && (
                          <Badge variant="default" className="text-xs">Active</Badge>
                        )}
                      </div>
                      <Badge variant="outline" className="text-xs">
                        {modelData.classes?.length || 0} classes
                      </Badge>
                    </div>
                    <div className="text-xs text-slate-600">
                      Path: {modelData.path}
                    </div>
                    {modelData.loaded_at && (
                      <div className="text-xs text-slate-500">
                        Loaded: {new Date(modelData.loaded_at * 1000).toLocaleDateString()}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </>
          )}
        </CardContent>
      </Card>

      {/* Custom Model Upload */}
      <Card className="bg-white/80 backdrop-blur-sm border-slate-200">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Upload className="w-5 h-5" />
            Upload Custom Model
          </CardTitle>
          <CardDescription>
            Upload your own trained YOLO model (.pt or .onnx format)
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          {/* Upload Area */}
          <div 
            className="border-2 border-dashed border-slate-300 rounded-lg p-8 text-center hover:border-blue-400 transition-colors cursor-pointer"
            onClick={() => modelInputRef.current?.click()}
          >
            <div className="space-y-2">
              <Upload className="w-12 h-12 mx-auto text-slate-400" />
              <p className="text-slate-600">Click to upload custom model</p>
              <p className="text-sm text-slate-400">Supports .pt and .onnx formats</p>
            </div>
          </div>
          
          <input
            type="file"
            ref={modelInputRef}
            onChange={handleCustomModelUpload}
            accept=".pt,.onnx"
            className="hidden"
          />
          
          <Button 
            onClick={() => modelInputRef.current?.click()}
            disabled={modelUploading}
            className="w-full bg-gradient-to-r from-indigo-600 to-purple-600 hover:from-indigo-700 hover:to-purple-700"
          >
            {modelUploading ? 'Uploading Model...' : 'Upload Custom Model'}
          </Button>

          {/* Upload Guidelines */}
          <div className="p-3 bg-amber-50 border border-amber-200 rounded-lg">
            <div className="flex items-start gap-2">
              <AlertCircle className="w-4 h-4 text-amber-600 mt-0.5" />
              <div className="text-sm text-amber-700">
                <div className="font-medium mb-1">Upload Guidelines:</div>
                <ul className="text-xs space-y-1">
                  <li>• Model must be in YOLO format (.pt or .onnx)</li>
                  <li>• Ensure model is compatible with YOLOv8/v11 architecture</li>
                  <li>• Model will be validated before loading</li>
                  <li>• Custom models will be available in model selection</li>
                </ul>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default ModelTab;