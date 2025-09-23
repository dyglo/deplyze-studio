import React from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card';
import { Button } from './ui/button';
import { Badge } from './ui/badge';
import { Progress } from './ui/progress';
import { Upload, Download, Package, BarChart3 } from 'lucide-react';

const BatchTab = ({ 
  batchInputRef, 
  batchFiles, 
  batchProcessing, 
  batchResults, 
  handleBatchUpload, 
  downloadBatchResults 
}) => {
  return (
    <div className="space-y-6">
      {/* Upload Section */}
      <Card className="bg-white/80 backdrop-blur-sm border-slate-200">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Package className="w-5 h-5" />
            Batch Image Processing
          </CardTitle>
          <CardDescription>
            Upload multiple images for simultaneous processing (up to 50 images)
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div 
            className="border-2 border-dashed border-slate-300 rounded-lg p-8 text-center hover:border-purple-400 transition-colors cursor-pointer"
            onClick={() => batchInputRef.current?.click()}
          >
            {batchFiles.length > 0 ? (
              <div className="space-y-2">
                <Package className="w-12 h-12 mx-auto text-purple-500" />
                <p className="text-slate-600">{batchFiles.length} images selected</p>
                <p className="text-sm text-slate-400">Click to select different images</p>
              </div>
            ) : (
              <div className="space-y-2">
                <Upload className="w-12 h-12 mx-auto text-slate-400" />
                <p className="text-slate-600">Click to select multiple images</p>
                <p className="text-sm text-slate-400">PNG, JPG, GIF - Max 50 files</p>
              </div>
            )}
          </div>
          
          <input
            type="file"
            ref={batchInputRef}
            onChange={handleBatchUpload}
            accept="image/*"
            multiple
            className="hidden"
          />
          
          <Button 
            onClick={() => batchInputRef.current?.click()}
            disabled={batchProcessing}
            className="w-full bg-gradient-to-r from-purple-600 to-indigo-600 hover:from-purple-700 hover:to-indigo-700"
          >
            {batchProcessing ? 'Processing Images...' : 'Select Images for Batch Processing'}
          </Button>
        </CardContent>
      </Card>

      {/* Processing Status */}
      {batchProcessing && (
        <Card className="bg-white/80 backdrop-blur-sm border-slate-200">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <BarChart3 className="w-5 h-5" />
              Processing Status
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div className="flex items-center gap-2">
                <div className="animate-spin rounded-full w-4 h-4 border-2 border-purple-600 border-t-transparent"></div>
                <span className="text-sm text-slate-600">Processing batch images...</span>
              </div>
              <Progress value={60} className="h-2" />
              <p className="text-xs text-slate-500">
                Analyzing images and generating detection results. This may take several minutes.
              </p>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Results Section */}
      {batchResults && (
        <Card className="bg-white/80 backdrop-blur-sm border-slate-200">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <BarChart3 className="w-5 h-5" />
              Batch Processing Results
            </CardTitle>
            <CardDescription>
              Summary of batch processing results
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            {/* Statistics Grid */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div className="text-center p-3 bg-slate-50 rounded-lg">
                <div className="text-2xl font-bold text-purple-600">{batchResults.total_images}</div>
                <div className="text-sm text-slate-600">Total Images</div>
              </div>
              <div className="text-center p-3 bg-slate-50 rounded-lg">
                <div className="text-2xl font-bold text-green-600">{batchResults.processed_images}</div>
                <div className="text-sm text-slate-600">Processed</div>
              </div>
              <div className="text-center p-3 bg-slate-50 rounded-lg">
                <div className="text-2xl font-bold text-blue-600">{batchResults.total_detections}</div>
                <div className="text-sm text-slate-600">Objects Found</div>
              </div>
              <div className="text-center p-3 bg-slate-50 rounded-lg">
                <div className="text-2xl font-bold text-orange-600">{batchResults.failed_images || 0}</div>
                <div className="text-sm text-slate-600">Failed</div>
              </div>
            </div>

            {/* Processing Details */}
            <div className="space-y-2">
              <div className="flex justify-between text-sm">
                <span>Processing Time:</span>
                <span>{batchResults.processing_time?.toFixed(2)}s</span>
              </div>
              <div className="flex justify-between text-sm">
                <span>Average per Image:</span>
                <span>{(batchResults.processing_time / batchResults.total_images).toFixed(2)}s</span>
              </div>
              <div className="flex justify-between text-sm">
                <span>Success Rate:</span>
                <span>{((batchResults.processed_images / batchResults.total_images) * 100).toFixed(1)}%</span>
              </div>
              <div className="flex justify-between text-sm">
                <span>Objects per Image:</span>
                <span>{(batchResults.total_detections / batchResults.processed_images).toFixed(1)} avg</span>
              </div>
            </div>

            {/* Status Badges */}
            <div className="flex flex-wrap gap-2">
              <Badge variant="default">
                {batchResults.processed_images} Successful
              </Badge>
              {batchResults.failed_images > 0 && (
                <Badge variant="destructive">
                  {batchResults.failed_images} Failed
                </Badge>
              )}
              <Badge variant="secondary">
                {batchResults.total_detections} Objects Detected
              </Badge>
            </div>

            {/* Download Button */}
            <Button 
              onClick={downloadBatchResults} 
              className="w-full"
              disabled={!batchResults.results_archive}
            >
              <Download className="w-4 h-4 mr-2" />
              Download All Annotated Images (ZIP)
            </Button>
          </CardContent>
        </Card>
      )}
    </div>
  );
};

export default BatchTab;