import React, { useState, useRef, useEffect } from 'react';
import './App.css';
import { Button } from './components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './components/ui/tabs';
import { Badge } from './components/ui/badge';
import { Progress } from './components/ui/progress';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from './components/ui/select';
import { Upload, Camera, Play, Square, Download, Zap, Target, Clock, Video, Eye, Layers, Package, BarChart3 } from 'lucide-react';
import axios from 'axios';
import { toast } from 'sonner';
import BatchTab from './components/BatchTab';
import ModelTab from './components/ModelTab';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

function App() {
  const [activeTab, setActiveTab] = useState('image');
  const [isLoading, setIsLoading] = useState(false);
  const [detectionResults, setDetectionResults] = useState(null);
  const [uploadedImage, setUploadedImage] = useState(null);
  const [annotatedImage, setAnnotatedImage] = useState(null);
  const [isVideoStreaming, setIsVideoStreaming] = useState(false);
  const [videoDetections, setVideoDetections] = useState([]);
  const [modelInfo, setModelInfo] = useState(null);
  const [uploadedVideo, setUploadedVideo] = useState(null);
  const [videoProcessing, setVideoProcessing] = useState(false);
  const [videoResults, setVideoResults] = useState(null);
  const [batchFiles, setBatchFiles] = useState([]);
  const [batchProcessing, setBatchProcessing] = useState(false);
  const [batchResults, setBatchResults] = useState(null);
  const [customModelFile, setCustomModelFile] = useState(null);
  const [modelUploading, setModelUploading] = useState(false);
  const [selectedModel, setSelectedModel] = useState("yolo11n");
  
  // Refs
  const fileInputRef = useRef(null);
  const videoFileInputRef = useRef(null);
  const batchInputRef = useRef(null);
  const modelInputRef = useRef(null);
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const streamRef = useRef(null);
  const wsRef = useRef(null);

  // Load model info on component mount
  useEffect(() => {
    loadModelInfo();
  }, []);

  const loadModelInfo = async () => {
    try {
      const response = await axios.get(`${API}/model/info`);
      setModelInfo(response.data);
      setSelectedModel(response.data.active_model);
    } catch (error) {
      console.error('Error loading model info:', error);
    }
  };

  const switchModel = async (modelName) => {
    try {
      await axios.post(`${API}/model/switch?model_name=${modelName}`);
      setSelectedModel(modelName);
      await loadModelInfo(); // Refresh model info
      toast.success(`Switched to model: ${modelName}`);
    } catch (error) {
      console.error('Error switching model:', error);
      toast.error('Failed to switch model');
    }
  };

  const handleCustomModelUpload = async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    setModelUploading(true);

    try {
      const formData = new FormData();
      formData.append('file', file);
      formData.append('model_name', file.name.split('.')[0]);

      const response = await axios.post(`${API}/model/upload`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });

      setCustomModelFile(null);
      await loadModelInfo(); // Refresh model info
      toast.success(`Custom model uploaded: ${response.data.model_name}`);

    } catch (error) {
      console.error('Error uploading model:', error);
      toast.error('Failed to upload custom model');
    } finally {
      setModelUploading(false);
    }
  };

  const handleBatchUpload = async (event) => {
    const files = Array.from(event.target.files);
    if (files.length === 0) return;

    setBatchProcessing(true);
    setBatchResults(null);

    try {
      const formData = new FormData();
      files.forEach((file) => {
        formData.append('files', file);
      });

      toast.info(`Processing ${files.length} images...`);

      const response = await axios.post(`${API}/detect/batch/images`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
        timeout: 600000 // 10 minutes timeout
      });

      setBatchResults(response.data);
      setBatchFiles(files);
      toast.success(`Batch processing completed! ${response.data.total_detections} objects detected in ${response.data.processed_images} images`);

    } catch (error) {
      console.error('Error processing batch:', error);
      toast.error('Failed to process batch images');
    } finally {
      setBatchProcessing(false);
    }
  };

  const downloadBatchResults = async () => {
    if (!batchResults?.results_archive) return;

    try {
      const response = await axios.get(`${API}/download/batch/${batchResults.results_archive}`, {
        responseType: 'blob'
      });

      const url = window.URL.createObjectURL(new Blob([response.data]));
      const link = document.createElement('a');
      link.href = url;
      link.download = 'deplyze_studio_batch_results.zip';
      link.click();
      window.URL.revokeObjectURL(url);
      
      toast.success('Batch results download started!');
    } catch (error) {
      console.error('Error downloading batch results:', error);
      toast.error('Failed to download batch results');
    }
  };

  const handleVideoUpload = async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    setVideoProcessing(true);
    setVideoResults(null);

    try {
      // Show uploaded video preview
      const videoUrl = URL.createObjectURL(file);
      setUploadedVideo(videoUrl);

      // Create form data
      const formData = new FormData();
      formData.append('file', file);

      toast.info(`Processing video: ${file.name}. This may take a while...`);

      // Process video
      const response = await axios.post(`${API}/detect/video`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
        timeout: 300000 // 5 minutes timeout for video processing
      });

      setVideoResults(response.data);
      toast.success(`Video processed! Found ${response.data.total_detections} detections in ${response.data.processed_frames} frames`);

    } catch (error) {
      console.error('Error processing video:', error);
      toast.error('Failed to process video. Please try a smaller file.');
    } finally {
      setVideoProcessing(false);
    }
  };

  const downloadProcessedVideo = async () => {
    if (!videoResults?.output_filename) return;

    try {
      const response = await axios.get(`${API}/download/video/${videoResults.output_filename}`, {
        responseType: 'blob'
      });

      const url = window.URL.createObjectURL(new Blob([response.data]));
      const link = document.createElement('a');
      link.href = url;
      link.download = `logo_detection_${videoResults.output_filename}`;
      link.click();
      window.URL.revokeObjectURL(url);
      
      toast.success('Video download started!');
    } catch (error) {
      console.error('Error downloading video:', error);
      toast.error('Failed to download processed video');
    }
  };

  const handleImageUpload = async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    setIsLoading(true);
    setDetectionResults(null);
    setAnnotatedImage(null);

    try {
      // Show uploaded image preview
      const imageUrl = URL.createObjectURL(file);
      setUploadedImage(imageUrl);

      // Create form data
      const formData = new FormData();
      formData.append('file', file);

      // Get detection results
      const [detectResponse, annotatedResponse] = await Promise.all([
        axios.post(`${API}/detect/image`, formData, {
          headers: { 'Content-Type': 'multipart/form-data' }
        }),
        axios.post(`${API}/detect/image/annotated`, formData, {
          headers: { 'Content-Type': 'multipart/form-data' }
        })
      ]);

      setDetectionResults(detectResponse.data);
      setAnnotatedImage(`data:image/jpeg;base64,${annotatedResponse.data.annotated_image}`);
      
      toast.success(`Found ${detectResponse.data.detections.length} logos in ${detectResponse.data.inference_time.toFixed(2)}s`);

    } catch (error) {
      console.error('Error uploading image:', error);
      toast.error('Failed to detect logos in image');
    } finally {
      setIsLoading(false);
    }
  };

  const startVideoDetection = async () => {
    try {
      // Get user media
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480 }
      });
      
      streamRef.current = stream;
      videoRef.current.srcObject = stream;
      
      // Setup WebSocket
      const wsUrl = BACKEND_URL.replace('https://', 'wss://').replace('http://', 'ws://');
      wsRef.current = new WebSocket(`${wsUrl}/api/detect/video`);
      
      wsRef.current.onopen = () => {
        console.log('WebSocket connected');
        setIsVideoStreaming(true);
      };
      
      wsRef.current.onmessage = (event) => {
        const data = JSON.parse(event.data);
        if (data.type === 'detection_result') {
          setVideoDetections(data.detections);
          drawBoundingBoxes(data.detections);
        }
      };
      
      wsRef.current.onerror = (error) => {
        console.error('WebSocket error:', error);
        toast.error('Failed to connect to video detection service');
      };
      
      // Start sending frames
      setTimeout(() => {
        sendVideoFrames();
      }, 1000);
      
    } catch (error) {
      console.error('Error starting video detection:', error);
      toast.error('Failed to access camera');
    }
  };

  const sendVideoFrames = () => {
    if (!wsRef.current || !videoRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
      return;
    }

    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    canvas.width = videoRef.current.videoWidth;
    canvas.height = videoRef.current.videoHeight;
    
    ctx.drawImage(videoRef.current, 0, 0);
    
    // Convert to base64
    const frameData = canvas.toDataURL('image/jpeg', 0.8).split(',')[1];
    
    // Send frame
    wsRef.current.send(JSON.stringify({ frame: frameData }));
    
    // Continue sending frames (limit to ~10 FPS for CPU efficiency)
    if (isVideoStreaming) {
      setTimeout(sendVideoFrames, 100);
    }
  };

  const drawBoundingBoxes = (detections) => {
    const canvas = canvasRef.current;
    const video = videoRef.current;
    
    if (!canvas || !video) return;
    
    const ctx = canvas.getContext('2d');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    
    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Draw bounding boxes
    detections.forEach((detection, index) => {
      const { bbox, class_name, confidence } = detection;
      const color = getColorForClass(index);
      
      // Draw bounding box
      ctx.strokeStyle = color;
      ctx.lineWidth = 4; // Fixed thickness
      ctx.strokeRect(bbox.x1, bbox.y1, bbox.width, bbox.height);
      
      // Draw label background
      ctx.fillStyle = color;
      const labelText = `${class_name}: ${(confidence * 100).toFixed(1)}%`;
      const textMetrics = ctx.measureText(labelText);
      const labelWidth = textMetrics.width + 10;
      const labelHeight = 25;
      
      ctx.fillRect(bbox.x1, bbox.y1 - labelHeight, labelWidth, labelHeight);
      
      // Draw label text
      ctx.fillStyle = '#ffffff';
      ctx.font = '14px Inter, sans-serif';
      ctx.fillText(labelText, bbox.x1 + 5, bbox.y1 - 8);
    });
  };

  const stopVideoDetection = () => {
    setIsVideoStreaming(false);
    setVideoDetections([]);
    
    // Stop webcam
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }
    
    // Close WebSocket
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
    
    // Clear canvas
    if (canvasRef.current) {
      const ctx = canvasRef.current.getContext('2d');
      ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
    }
  };

  const getColorForClass = (index) => {
    const colors = [
      '#ef4444', '#10b981', '#3b82f6', '#f59e0b', 
      '#8b5cf6', '#06b6d4', '#84cc16', '#f97316'
    ];
    return colors[index % colors.length];
  };

  const downloadResults = async () => {
    if (!detectionResults?.id) return;
    
    try {
      const response = await axios.get(`${API}/download/image/${detectionResults.id}`, {
        responseType: 'blob'
      });

      const url = window.URL.createObjectURL(new Blob([response.data]));
      const link = document.createElement('a');
      link.href = url;
      link.download = `deplyze_studio_annotated_${detectionResults.id}.jpg`;
      link.click();
      window.URL.revokeObjectURL(url);
      
      toast.success('Annotated image download started!');
    } catch (error) {
      console.error('Error downloading annotated image:', error);
      toast.error('Failed to download annotated image');
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-100">
      <div className="container mx-auto px-4 py-8">
        {/* Header */}
        <div className="text-center mb-12">
          <div className="inline-flex items-center justify-center w-16 h-16 bg-gradient-to-r from-blue-600 to-purple-600 rounded-2xl mb-6">
            <Eye className="w-8 h-8 text-white" />
          </div>
          <h1 className="text-4xl font-bold text-slate-900 mb-4">Deplyze Studio</h1>
          <p className="text-xl text-slate-600 max-w-2xl mx-auto">
            Advanced AI-powered object detection with custom models, batch processing, and detailed analytics
          </p>
          
          {/* Model Info */}
          {modelInfo && (
            <div className="flex items-center justify-center gap-6 mt-8">
              <Badge variant="secondary" className="px-3 py-1">
                <Zap className="w-4 h-4 mr-1" />
                {modelInfo.device.toUpperCase()}
              </Badge>
              <Badge variant="secondary" className="px-3 py-1">
                <Target className="w-4 h-4 mr-1" />
                {modelInfo.confidence_threshold * 100}% Confidence
              </Badge>
              <Badge variant="secondary" className="px-3 py-1">
                <Layers className="w-4 h-4 mr-1" />
                {modelInfo.active_model}
              </Badge>
              <Badge variant="secondary" className="px-3 py-1">
                <Clock className="w-4 h-4 mr-1" />
                {modelInfo.max_inference_size}px Max Size
              </Badge>
            </div>
          )}
        </div>

        {/* Main Content */}
        <div className="max-w-6xl mx-auto">
          <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-8">
            <TabsList className="grid w-full grid-cols-4 bg-white/80 backdrop-blur-sm border border-slate-200">
              <TabsTrigger value="image" className="flex items-center gap-2">
                <Upload className="w-4 h-4" />
                Single Image
              </TabsTrigger>
              <TabsTrigger value="batch" className="flex items-center gap-2">
                <Package className="w-4 h-4" />
                Batch Processing
              </TabsTrigger>
              <TabsTrigger value="video" className="flex items-center gap-2">
                <Video className="w-4 h-4" />
                Video Processing
              </TabsTrigger>
              <TabsTrigger value="models" className="flex items-center gap-2">
                <Layers className="w-4 h-4" />
                Model Management
              </TabsTrigger>
            </TabsList>

            {/* Image Detection Tab */}
            <TabsContent value="image" className="space-y-6">
              <div className="grid md:grid-cols-2 gap-6">
                {/* Upload Card */}
                <Card className="bg-white/80 backdrop-blur-sm border-slate-200">
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <Upload className="w-5 h-5" />
                      Upload Image
                    </CardTitle>
                    <CardDescription>
                      Select an image to detect brand logos
                    </CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <div 
                      className="border-2 border-dashed border-slate-300 rounded-lg p-8 text-center hover:border-blue-400 transition-colors cursor-pointer"
                      onClick={() => fileInputRef.current?.click()}
                    >
                      {uploadedImage ? (
                        <img src={uploadedImage} alt="Uploaded" className="max-h-64 mx-auto rounded-lg" />
                      ) : (
                        <div className="space-y-2">
                          <Upload className="w-12 h-12 mx-auto text-slate-400" />
                          <p className="text-slate-600">Click to upload an image</p>
                          <p className="text-sm text-slate-400">PNG, JPG, GIF up to 10MB</p>
                        </div>
                      )}
                    </div>
                    
                    <input
                      type="file"
                      ref={fileInputRef}
                      onChange={handleImageUpload}
                      accept="image/*"
                      className="hidden"
                    />
                    
                    <Button 
                      onClick={() => fileInputRef.current?.click()}
                      disabled={isLoading}
                      className="w-full bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700"
                    >
                      {isLoading ? 'Detecting...' : 'Select Image'}
                    </Button>
                  </CardContent>
                </Card>

                {/* Results Card */}
                <Card className="bg-white/80 backdrop-blur-sm border-slate-200">
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <Target className="w-5 h-5" />
                      Detection Results
                    </CardTitle>
                    <CardDescription>
                      Identified logos and confidence scores
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    {isLoading ? (
                      <div className="space-y-4">
                        <div className="flex items-center gap-2">
                          <div className="animate-spin rounded-full w-4 h-4 border-2 border-blue-600 border-t-transparent"></div>
                          <span className="text-sm text-slate-600">Analyzing image...</span>
                        </div>
                        <Progress value={75} className="h-2" />
                      </div>
                    ) : detectionResults ? (
                      <div className="space-y-4">
                        <div className="flex items-center justify-between">
                          <span className="text-sm font-medium">
                            Found {detectionResults.detections.length} objects
                          </span>
                          <Badge variant="outline">
                            {detectionResults.inference_time?.toFixed(2)}s
                          </Badge>
                        </div>
                        
                        {/* Enhanced Detection Results */}
                        <div className="space-y-3">
                          {detectionResults.detections.map((detection, index) => (
                            <div key={index} className="p-4 bg-slate-50 rounded-lg border">
                              <div className="flex items-center justify-between mb-2">
                                <div className="flex items-center gap-2">
                                  <Badge 
                                    variant="secondary"
                                    style={{ backgroundColor: getColorForClass(index) + '20', color: getColorForClass(index) }}
                                  >
                                    {detection.class_name || 'Unknown'}
                                  </Badge>
                                  {detection.class_id !== undefined && (
                                    <span className="text-xs text-slate-500">ID: {detection.class_id}</span>
                                  )}
                                </div>
                                <div className="flex items-center gap-2">
                                  <Badge 
                                    variant={
                                      detection.reliability === 'High' ? 'default' : 
                                      detection.reliability === 'Medium' ? 'secondary' : 'outline'
                                    }
                                  >
                                    {detection.confidence_percentage ? 
                                      `${detection.confidence_percentage}%` : 
                                      `${(detection.confidence * 100).toFixed(1)}%`
                                    }
                                  </Badge>
                                  {detection.reliability && (
                                    <span className="text-xs text-slate-500">{detection.reliability}</span>
                                  )}
                                </div>
                              </div>
                              
                              <div className="grid grid-cols-3 gap-4 text-xs text-slate-600">
                                <div>
                                  <span className="font-medium">Size:</span><br />
                                  {detection.bbox?.width || 0}×{detection.bbox?.height || 0}px
                                </div>
                                <div>
                                  <span className="font-medium">Position:</span><br />
                                  ({detection.bbox?.center_x || Math.round((detection.bbox?.x1 + detection.bbox?.x2) / 2) || 0}, {detection.bbox?.center_y || Math.round((detection.bbox?.y1 + detection.bbox?.y2) / 2) || 0})
                                </div>
                                <div>
                                  <span className="font-medium">Area:</span><br />
                                  {detection.bbox?.area?.toLocaleString() || ((detection.bbox?.width || 0) * (detection.bbox?.height || 0)).toLocaleString()}px²
                                </div>
                              </div>
                              
                              {/* Confidence Bar */}
                              <div className="mt-2">
                                <div className="flex justify-between text-xs text-slate-500 mb-1">
                                  <span>Confidence Level</span>
                                  <span>
                                    {detection.confidence_percentage ? 
                                      `${detection.confidence_percentage}%` : 
                                      `${(detection.confidence * 100).toFixed(1)}%`
                                    }
                                  </span>
                                </div>
                                <div className="w-full bg-slate-200 rounded-full h-2">
                                  <div 
                                    className="h-2 rounded-full transition-all duration-300"
                                    style={{ 
                                      width: `${detection.confidence_percentage || (detection.confidence * 100)}%`,
                                      backgroundColor: getColorForClass(index)
                                    }}
                                  />
                                </div>
                              </div>
                            </div>
                          ))}
                        </div>

                        {/* Summary Statistics */}
                        {detectionResults.summary && (
                          <div className="mt-4 p-3 bg-blue-50 rounded-lg">
                            <div className="text-sm font-medium text-blue-900 mb-2">Detection Summary</div>
                            <div className="grid grid-cols-2 gap-3 text-xs text-blue-700">
                              <div>
                                <span className="font-medium">Avg Confidence:</span><br />
                                {detectionResults.summary.average_confidence ? 
                                  (detectionResults.summary.average_confidence * 100).toFixed(1) : 
                                  'N/A'
                                }%
                              </div>
                              <div>
                                <span className="font-medium">Highest:</span><br />
                                {detectionResults.summary.highest_confidence ? 
                                  (detectionResults.summary.highest_confidence * 100).toFixed(1) : 
                                  'N/A'
                                }%
                              </div>
                              <div>
                                <span className="font-medium">Classes Found:</span><br />
                                {detectionResults.summary.object_classes?.length || 0}
                              </div>
                              <div>
                                <span className="font-medium">Model:</span><br />
                                {detectionResults.summary.model_used || 'yolo11n'}
                              </div>
                            </div>
                          </div>
                        )}
                        
                        <Button onClick={downloadResults} variant="outline" className="w-full">
                          <Download className="w-4 h-4 mr-2" />
                          Download Annotated Image
                        </Button>
                      </div>
                    ) : (
                      <p className="text-slate-400 text-center py-8">Upload an image to see results</p>
                    )}
                  </CardContent>
                </Card>
              </div>

              {/* Annotated Image Display */}
              {annotatedImage && (
                <Card className="bg-white/80 backdrop-blur-sm border-slate-200">
                  <CardHeader>
                    <CardTitle>Annotated Image</CardTitle>
                    <CardDescription>Image with detected logo bounding boxes</CardDescription>
                  </CardHeader>
                  <CardContent>
                    <img src={annotatedImage} alt="Annotated" className="w-full rounded-lg" />
                  </CardContent>
                </Card>
              )}
            </TabsContent>

            {/* Video Processing Tab */}
            <TabsContent value="video" className="space-y-6">
              <div className="grid lg:grid-cols-2 gap-6">
                {/* Video Upload and Processing */}
                <div className="space-y-6">
                  {/* Video File Upload */}
                  <Card className="bg-white/80 backdrop-blur-sm border-slate-200">
                    <CardHeader>
                      <CardTitle className="flex items-center gap-2">
                        <Video className="w-5 h-5" />
                        Upload Video File
                      </CardTitle>
                      <CardDescription>
                        Upload a video file for object detection processing
                      </CardDescription>
                    </CardHeader>
                    <CardContent className="space-y-4">
                      <div 
                        className="border-2 border-dashed border-slate-300 rounded-lg p-8 text-center hover:border-blue-400 transition-colors cursor-pointer"
                        onClick={() => videoFileInputRef.current?.click()}
                      >
                        {uploadedVideo ? (
                          <video src={uploadedVideo} controls className="max-h-48 mx-auto rounded-lg" />
                        ) : (
                          <div className="space-y-2">
                            <Video className="w-12 h-12 mx-auto text-slate-400" />
                            <p className="text-slate-600">Click to upload a video</p>
                            <p className="text-sm text-slate-400">MP4, AVI, MOV up to 100MB</p>
                          </div>
                        )}
                      </div>
                      
                      <input
                        type="file"
                        ref={videoFileInputRef}
                        onChange={handleVideoUpload}
                        accept="video/*"
                        className="hidden"
                      />
                      
                      <Button 
                        onClick={() => videoFileInputRef.current?.click()}
                        disabled={videoProcessing}
                        className="w-full bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-700 hover:to-pink-700"
                      >
                        {videoProcessing ? 'Processing...' : 'Select Video'}
                      </Button>
                    </CardContent>
                  </Card>

                  {/* Real-time Camera */}
                  <Card className="bg-white/80 backdrop-blur-sm border-slate-200">
                    <CardHeader>
                      <CardTitle className="flex items-center gap-2">
                        <Camera className="w-5 h-5" />
                        Live Camera Feed
                      </CardTitle>
                      <CardDescription>
                        Real-time object detection from your camera
                      </CardDescription>
                    </CardHeader>
                    <CardContent>
                      <div className="relative">
                        <video
                          ref={videoRef}
                          autoPlay
                          muted
                          className="w-full rounded-lg bg-slate-900"
                          style={{ maxHeight: '300px' }}
                        />
                        <canvas
                          ref={canvasRef}
                          className="absolute top-0 left-0 w-full h-full pointer-events-none"
                          style={{ maxHeight: '300px' }}
                        />
                      </div>
                      
                      <div className="flex gap-2 mt-4">
                        {!isVideoStreaming ? (
                          <Button 
                            onClick={startVideoDetection}
                            className="bg-gradient-to-r from-green-600 to-emerald-600 hover:from-green-700 hover:to-emerald-700"
                          >
                            <Play className="w-4 h-4 mr-2" />
                            Start Camera
                          </Button>
                        ) : (
                          <Button 
                            onClick={stopVideoDetection}
                            variant="destructive"
                          >
                            <Square className="w-4 h-4 mr-2" />
                            Stop Camera
                          </Button>
                        )}
                      </div>
                    </CardContent>
                  </Card>
                </div>

                {/* Results Section */}
                <div className="space-y-6">
                  {/* Video Processing Results */}
                  <Card className="bg-white/80 backdrop-blur-sm border-slate-200">
                    <CardHeader>
                      <CardTitle className="flex items-center gap-2">
                        <Target className="w-5 h-5" />
                        Processing Results
                      </CardTitle>
                      <CardDescription>
                        Video processing statistics and download
                      </CardDescription>
                    </CardHeader>
                    <CardContent>
                      {videoProcessing ? (
                        <div className="space-y-4">
                          <div className="flex items-center gap-2">
                            <div className="animate-spin rounded-full w-4 h-4 border-2 border-purple-600 border-t-transparent"></div>
                            <span className="text-sm text-slate-600">Processing video...</span>
                          </div>
                          <Progress value={50} className="h-2" />
                          <p className="text-xs text-slate-500">This may take several minutes depending on video length</p>
                        </div>
                      ) : videoResults ? (
                        <div className="space-y-4">
                          <div className="grid grid-cols-2 gap-4">
                            <div className="text-center p-3 bg-slate-50 rounded-lg">
                              <div className="text-2xl font-bold text-purple-600">{videoResults.processed_frames}</div>
                              <div className="text-sm text-slate-600">Frames Processed</div>
                            </div>
                            <div className="text-center p-3 bg-slate-50 rounded-lg">
                              <div className="text-2xl font-bold text-purple-600">{videoResults.total_detections}</div>
                              <div className="text-sm text-slate-600">Total Detections</div>
                            </div>
                          </div>
                          
                          <div className="space-y-2">
                            <div className="flex justify-between text-sm">
                              <span>Processing Time:</span>
                              <span>{videoResults.processing_time.toFixed(2)}s</span>
                            </div>
                            <div className="flex justify-between text-sm">
                              <span>Resolution:</span>
                              <span>{videoResults.resolution}</span>
                            </div>
                            <div className="flex justify-between text-sm">
                              <span>FPS:</span>
                              <span>{videoResults.fps}</span>
                            </div>
                            <div className="flex justify-between text-sm">
                              <span>Avg Detections/Frame:</span>
                              <span>{videoResults.avg_detections_per_frame.toFixed(2)}</span>
                            </div>
                          </div>
                          
                          <Button onClick={downloadProcessedVideo} className="w-full">
                            <Download className="w-4 h-4 mr-2" />
                            Download Processed Video
                          </Button>
                        </div>
                      ) : (
                        <p className="text-slate-400 text-center py-8">Upload a video to see processing results</p>
                      )}
                    </CardContent>
                  </Card>

                  {/* Live Camera Results */}
                  <Card className="bg-white/80 backdrop-blur-sm border-slate-200">
                    <CardHeader>
                      <CardTitle className="flex items-center gap-2">
                        <Camera className="w-5 h-5" />
                        Live Detections
                      </CardTitle>
                      <CardDescription>
                        Current camera frame detections
                      </CardDescription>
                    </CardHeader>
                    <CardContent>
                      {isVideoStreaming ? (
                        <div className="space-y-3">
                          {videoDetections.length > 0 ? (
                            videoDetections.map((detection, index) => (
                              <div key={index} className="flex items-center justify-between p-2 bg-slate-50 rounded-lg">
                                <span className="text-sm font-medium">{detection.class_name}</span>
                                <Badge 
                                  variant="secondary"
                                  style={{ backgroundColor: getColorForClass(index) + '20', color: getColorForClass(index) }}
                                >
                                  {(detection.confidence * 100).toFixed(1)}%
                                </Badge>
                              </div>
                            ))
                          ) : (
                            <p className="text-slate-400 text-sm">No objects detected</p>
                          )}
                        </div>
                      ) : (
                        <p className="text-slate-400 text-center py-8">Start camera to see live detections</p>
                      )}
                    </CardContent>
                  </Card>
                </div>
              </div>
            </TabsContent>

            {/* Batch Processing Tab */}
            <TabsContent value="batch">
              <BatchTab
                batchInputRef={batchInputRef}
                batchFiles={batchFiles}
                batchProcessing={batchProcessing}
                batchResults={batchResults}
                handleBatchUpload={handleBatchUpload}
                downloadBatchResults={downloadBatchResults}
              />
            </TabsContent>

            {/* Model Management Tab */}
            <TabsContent value="models">
              <ModelTab
                modelInfo={modelInfo}
                selectedModel={selectedModel}
                modelUploading={modelUploading}
                modelInputRef={modelInputRef}
                switchModel={switchModel}
                handleCustomModelUpload={handleCustomModelUpload}
              />
            </TabsContent>
          </Tabs>
        </div>
      </div>
    </div>
  );
}

export default App;
