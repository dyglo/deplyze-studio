import React, { useState, useRef, useEffect } from 'react';
import './App.css';
import { Button } from './components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './components/ui/tabs';
import { Badge } from './components/ui/badge';
import { Progress } from './components/ui/progress';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from './components/ui/select';
import { Upload, Camera, Play, Square, Download, Zap, Target, Clock, Video, Eye, Layers, Package, BarChart3, PanelLeftClose, PanelLeftOpen, Search, Activity } from 'lucide-react';
import axios from 'axios';
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from './components/ui/dialog';
import { toast } from 'sonner';
import BatchTab from './components/BatchTab';
import ModelTab from './components/ModelTab';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

function App() {
  const [activeTab, setActiveTab] = useState('image');
  const [isLoading, setIsLoading] = useState(false);
  const [isSidebarCollapsed, setIsSidebarCollapsed] = useState(false);
  const [isCameraDialogOpen, setIsCameraDialogOpen] = useState(false);
  const [hoveredDetectionIndex, setHoveredDetectionIndex] = useState(null);
  const [detectionResults, setDetectionResults] = useState(null);
  const [uploadedImage, setUploadedImage] = useState(null);
  const [annotatedImage, setAnnotatedImage] = useState(null);
  const [isVideoStreaming, setIsVideoStreaming] = useState(false);
  const isStreamingRef = useRef(false);
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
        isStreamingRef.current = true;
      };
      
      wsRef.current.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          if (data.type === 'detection_result') {
            if (data.detections.length > 0) {
              console.log('Got detections:', data.detections.length);
            }
            setVideoDetections(data.detections);
            drawBoundingBoxes(data.detections);
          }
        } catch (err) {
          console.error('Error parsing WS message:', err);
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
    if (!wsRef.current || !videoRef.current || wsRef.current.readyState !== WebSocket.OPEN || !isStreamingRef.current) {
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
    if (isStreamingRef.current) {
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
    
    if (detections.length > 0) {
      console.log(`Drawing ${detections.length} boxes`);
    }
    
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
    isStreamingRef.current = false;
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
    <div className="flex h-screen bg-background text-foreground overflow-hidden">
      {/* Left Sidebar (WAIV architecture / Claude Code style) */}
      <aside className={`border-r bg-card flex-col hidden md:flex h-full shrink-0 transition-all duration-300 ${isSidebarCollapsed ? 'w-[68px]' : 'w-64'}`}>
        {/* Header/Logo */}
        <div className={`p-4 border-b flex items-center h-[68px] ${isSidebarCollapsed ? 'justify-center' : 'justify-between'}`}>
          {!isSidebarCollapsed && (
            <span className="font-outfit font-black text-xl text-foreground tracking-tight animate-in fade-in duration-300">
              Deplyze Studio
            </span>
          )}
          
          <button 
            onClick={() => setIsSidebarCollapsed(!isSidebarCollapsed)} 
            className={`p-2 hover:bg-muted rounded-md text-muted-foreground transition-all duration-200 ${isSidebarCollapsed ? '' : 'ml-auto'}`}
            title={isSidebarCollapsed ? "Expand Sidebar" : "Collapse Sidebar"}
          >
            {isSidebarCollapsed ? <PanelLeftOpen className="w-5 h-5" /> : <PanelLeftClose className="w-5 h-5" />}
          </button>
        </div>

        {/* Navigation */}
        <nav className="flex-1 overflow-y-auto py-6 flex flex-col gap-8 overflow-x-hidden">
          {/* Inference Section */}
          <div className="px-3">
            {!isSidebarCollapsed && (
              <h3 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-3 px-3">Inference</h3>
            )}
            <div className="space-y-1">
              <button 
                onClick={() => setActiveTab('image')}
                className={`flex items-center gap-3 w-full px-3 py-2.5 rounded-md transition-colors text-sm ${activeTab === 'image' ? 'bg-primary/10 text-primary font-medium' : 'text-muted-foreground hover:bg-muted/50 hover:text-foreground'} ${isSidebarCollapsed && 'justify-center'}`}
                title="Single Image"
              >
                <Upload className="w-5 h-5 shrink-0" />
                {!isSidebarCollapsed && <span>Single Image</span>}
              </button>
              <button 
                onClick={() => setActiveTab('batch')}
                className={`flex items-center gap-3 w-full px-3 py-2.5 rounded-md transition-colors text-sm ${activeTab === 'batch' ? 'bg-primary/10 text-primary font-medium' : 'text-muted-foreground hover:bg-muted/50 hover:text-foreground'} ${isSidebarCollapsed && 'justify-center'}`}
                title="Batch Processing"
              >
                <Package className="w-5 h-5 shrink-0" />
                {!isSidebarCollapsed && <span>Batch Processing</span>}
              </button>
              <button 
                onClick={() => setActiveTab('video')}
                className={`flex items-center gap-3 w-full px-3 py-2.5 rounded-md transition-colors text-sm ${activeTab === 'video' ? 'bg-primary/10 text-primary font-medium' : 'text-muted-foreground hover:bg-muted/50 hover:text-foreground'} ${isSidebarCollapsed && 'justify-center'}`}
                title="Video Processing"
              >
                <Video className="w-5 h-5 shrink-0" />
                {!isSidebarCollapsed && <span>Video Processing</span>}
              </button>
            </div>
          </div>

          {/* Studio Section */}
          <div className="px-3">
            {!isSidebarCollapsed && (
              <h3 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-3 px-3">Studio</h3>
            )}
            <div className="space-y-1">
              <button 
                onClick={() => setActiveTab('models')}
                className={`flex items-center gap-3 w-full px-3 py-2.5 rounded-md transition-colors text-sm ${activeTab === 'models' ? 'bg-primary/10 text-primary font-medium' : 'text-muted-foreground hover:bg-muted/50 hover:text-foreground'} ${isSidebarCollapsed && 'justify-center'}`}
                title="Model Management"
              >
                <Layers className="w-5 h-5 shrink-0" />
                {!isSidebarCollapsed && <span>Model Management</span>}
              </button>
            </div>
          </div>
        </nav>

        {/* Footer Model Status */}
        {modelInfo && (
          <div className={`p-4 border-t bg-muted/20 ${isSidebarCollapsed ? 'flex flex-col items-center justify-center' : ''}`}>
            {!isSidebarCollapsed ? (
              <>
                <div className="flex items-center gap-2 mb-3">
                  <div className="w-2 h-2 rounded-full bg-green-500"></div>
                  <span className="text-xs font-medium uppercase tracking-wider text-muted-foreground">Active Model</span>
                </div>
                <div className="space-y-2 text-sm pl-4">
                  <div className="flex items-center justify-between gap-2">
                    <span className="text-muted-foreground text-xs">Name</span>
                    <span className="font-medium text-xs truncate" title={modelInfo.active_model}>{modelInfo.active_model}</span>
                  </div>
                  <div className="flex items-center justify-between gap-2">
                    <span className="text-muted-foreground text-xs">Device</span>
                    <span className="font-medium text-xs uppercase">{modelInfo.device}</span>
                  </div>
                  <div className="flex items-center justify-between gap-2">
                    <span className="text-muted-foreground text-xs">Conf</span>
                    <span className="font-medium text-xs">{(modelInfo.confidence_threshold * 100).toFixed(0)}%</span>
                  </div>
                </div>
              </>
            ) : (
              <div 
                className="w-2 h-2 rounded-full bg-green-500 mx-auto" 
                title={`${modelInfo.active_model} (${modelInfo.device})`}
              ></div>
            )}
          </div>
        )}
      </aside>

      {/* Main Content Area */}
      <main className="flex-1 flex flex-col min-w-0 overflow-hidden bg-background">
        {/* Mobile Header */}
        <header className="md:hidden border-b bg-card p-4 flex items-center justify-between shrink-0">
          <div className="flex items-center gap-2">
            <img src="/logo.svg" className="w-6 h-6 object-contain" alt="Logo" />
            <span className="font-black text-foreground tracking-tight">Deplyze Studio</span>
          </div>
          <select 
            value={activeTab}
            onChange={(e) => setActiveTab(e.target.value)}
            className="text-sm border rounded p-1 bg-background"
          >
            <option value="image">Single Image</option>
            <option value="batch">Batch Processing</option>
            <option value="video">Video Processing</option>
            <option value="models">Model Management</option>
          </select>
        </header>

        {/* Scrollable Work Area */}
        <div className="flex-1 overflow-y-auto p-4 md:p-8">
          <div className="max-w-6xl mx-auto space-y-6">
            {/* Dynamic Header */}
            <div className="flex flex-col md:flex-row md:items-end justify-between gap-4 border-b border-border pb-6">
              <div>
                <h1 className="text-3xl font-bold tracking-tight text-foreground">
                  {activeTab === 'image' && 'Single Image'}
                  {activeTab === 'batch' && 'Batch Processing'}
                  {activeTab === 'video' && 'Video Processing'}
                  {activeTab === 'models' && 'Model Management'}
                </h1>
                <p className="text-sm text-muted-foreground mt-1">
                  {activeTab === 'image' && 'Run object detection on individual files.'}
                  {activeTab === 'batch' && 'Process multiple images concurrently.'}
                  {activeTab === 'video' && 'Analyze video files or connect a live camera feed.'}
                  {activeTab === 'models' && 'Configure active models, inspect classes, or upload custom weights.'}
                </p>
              </div>

              {activeTab === 'video' && (
                <div className="flex items-center gap-2">
                  <Dialog open={isCameraDialogOpen} onOpenChange={(open) => {
                    setIsCameraDialogOpen(open);
                    if (!open && isVideoStreaming) stopVideoDetection();
                  }}>
                    <DialogTrigger asChild>
                      <Button 
                        variant="outline" 
                        size="sm" 
                        className="h-9 gap-2 bg-primary/10 text-primary border-primary/20 hover:bg-primary hover:text-primary-foreground group transition-all"
                      >
                        <Camera className="w-4 h-4" />
                        <span className="font-semibold text-xs">Camera Live Feed</span>
                        <div className={`w-1.5 h-1.5 rounded-full ${isVideoStreaming ? 'bg-red-500 animate-pulse' : 'bg-muted-foreground/30'}`}></div>
                      </Button>
                    </DialogTrigger>
                    <DialogContent className="max-w-none w-screen h-screen p-0 overflow-hidden border-none bg-black flex flex-col z-[100] rounded-none">
                      <div className="flex-1 min-h-0 flex relative">
                        {/* Overlay Header (Floating) */}
                        <div className="absolute top-0 left-0 right-0 h-20 bg-gradient-to-b from-black/80 to-transparent pointer-events-none z-[60] flex items-center justify-between px-8">
                          <div className="flex items-center gap-4">
                            <div className="p-2.5 bg-primary/20 backdrop-blur-md rounded-xl border border-primary/20 shadow-2xl">
                              <Camera className="w-5 h-5 text-primary" />
                            </div>
                            <div className="flex flex-col">
                              <h3 className="text-white font-black tracking-widest text-base uppercase">LIVE VISION TELEMETRY</h3>
                              <div className="flex items-center gap-2">
                                <div className="w-1.5 h-1.5 rounded-full bg-red-500 animate-pulse"></div>
                                <span className="text-red-400 text-[9px] font-black uppercase tracking-widest">REAL-TIME ACQUISITION ACTIVE</span>
                              </div>
                            </div>
                          </div>
                          
                          <div className="flex items-center gap-4 pointer-events-auto">
                            <Badge variant="outline" className="bg-primary/20 text-primary border-primary/40 h-8 px-4 font-black tracking-widest text-[10px] rounded-full backdrop-blur-md">{selectedModel}</Badge>
                            <Button 
                              variant="ghost" 
                              size="icon" 
                              onClick={() => {
                                setIsCameraDialogOpen(false);
                                stopVideoDetection();
                              }}
                              className="text-white hover:bg-white/10 rounded-full w-10 h-10"
                            >
                              <Square className="w-4 h-4" />
                            </Button>
                          </div>
                        </div>
                        {/* Live Feed Workspace */}
                        <div className="flex-1 relative flex items-center justify-center p-0">
                          <div className="w-full h-full relative bg-black flex items-center justify-center overflow-hidden">
                            <video
                              ref={videoRef}
                              autoPlay
                              muted
                              onLoadedMetadata={() => {
                                if (canvasRef.current && videoRef.current) {
                                  canvasRef.current.width = videoRef.current.videoWidth;
                                  canvasRef.current.height = videoRef.current.videoHeight;
                                }
                              }}
                              className="w-full h-full object-cover"
                            />
                            <canvas
                              ref={canvasRef}
                              className="absolute top-0 left-0 w-full h-full pointer-events-none z-10"
                            />
                            
                            {!isVideoStreaming && (
                              <div className="absolute inset-0 flex flex-col items-center justify-center bg-background/80 backdrop-blur-xl z-20">
                                <div className="p-10 md:p-14 rounded-[3rem] flex flex-col items-center relative overflow-hidden group/modal max-w-lg w-full border border-border bg-card shadow-[0_30px_100px_rgba(0,0,0,0.1)] ring-1 ring-black/5">
                                  <div className="absolute -top-24 -left-24 w-48 h-48 bg-primary/5 rounded-full blur-[80px]"></div>
                                  <div className="absolute -bottom-24 -right-24 w-48 h-48 bg-primary/5 rounded-full blur-[80px]"></div>
                                  
                                  <div className="mb-8 relative shrink-0">
                                    <div className="absolute inset-0 bg-primary rounded-full blur-2xl opacity-10 animate-pulse"></div>
                                    <div className="p-8 bg-primary/5 rounded-[2.5rem] relative z-10 backdrop-blur-md border border-primary/20 shadow-inner group-hover/modal:scale-105 transition-transform duration-500">
                                      <Camera className="w-16 h-16 text-primary animate-pulse-brand" />
                                    </div>
                                  </div>

                                  <h3 className="text-3xl font-black text-foreground mb-4 relative z-10 tracking-tight text-center font-outfit uppercase">Ready to Initialize</h3>
                                  <p className="text-sm text-muted-foreground mb-10 relative z-10 text-center font-medium leading-relaxed px-6 max-w-sm">
                                    System is awaiting visual handshake. Grant camera permissions to start real-time telemetry and target acquisition.
                                  </p>

                                  <Button 
                                    onClick={startVideoDetection}
                                    className="h-16 px-14 rounded-2xl bg-primary text-primary-foreground font-black text-sm uppercase tracking-[0.3em] shadow-[0_15px_40px_-10px_rgba(198,93,60,0.4)] hover:scale-[1.02] active:scale-[0.98] transition-all relative z-10 border-none premium-shadow"
                                  >
                                    <Play className="w-5 h-5 mr-3 fill-current" />
                                    PROCEED
                                  </Button>

                                  <div className="mt-8 flex flex-col items-center gap-3">
                                    <div className="flex items-center gap-2 text-primary font-bold text-[9px] uppercase tracking-[0.2em] opacity-40">
                                      <div className="w-1.5 h-1.5 rounded-full bg-primary animate-ping"></div>
                                      SECURE CONNECTION ACTIVE
                                    </div>
                                  </div>
                                </div>
                              </div>
                            )}
                          </div>
                        </div>

                        {/* Live Inspector Sidebar (Overlay on right) */}
                        {isVideoStreaming && (
                          <div className="absolute top-6 right-6 bottom-6 w-80 bg-white/95 backdrop-blur-3xl rounded-[2rem] border border-border shadow-2xl flex flex-col overflow-hidden z-20 ring-1 ring-black/5 animate-in slide-in-from-right duration-500">
                            <div className="p-6 border-b bg-muted/10 flex items-center justify-between">
                              <div className="space-y-1">
                                <h4 className="text-[10px] font-black text-primary uppercase tracking-[0.3em]">TELEMETRY FEED</h4>
                                <p className="text-[12px] font-bold text-muted-foreground">Target Acquisition</p>
                              </div>
                              <Badge className="bg-primary text-primary-foreground border-none text-[12px] h-6 px-3 font-black rounded-full premium-shadow">{videoDetections.length}</Badge>
                            </div>
                            
                            <div className="flex-1 overflow-y-auto p-5 space-y-3 custom-scrollbar">
                              {videoDetections.map((detection, index) => (
                                <div key={index} className="flex items-center justify-between p-4 rounded-2xl bg-white border border-border hover:border-primary/40 hover:bg-muted/30 transition-all group/item shadow-sm">
                                  <div className="flex items-center gap-4 overflow-hidden">
                                    <div className="w-2.5 h-2.5 rounded-full shrink-0 shadow-[0_0_8px_rgba(198,93,60,0.3)] animate-pulse" style={{ backgroundColor: '#C65D3C' }}></div>
                                    <span className="text-[13px] font-bold text-foreground truncate group-hover/item:text-primary transition-colors uppercase tracking-tight">{detection.class_name}</span>
                                  </div>
                                  <div className="flex flex-col items-end">
                                    <span className="text-[12px] font-black text-primary">{(detection.confidence * 100).toFixed(0)}%</span>
                                    <span className="text-[8px] font-black text-muted-foreground uppercase tracking-widest">Conf</span>
                                  </div>
                                </div>
                              ))}
                              
                              {videoDetections.length === 0 && (
                                <div className="py-32 text-center opacity-40 flex flex-col items-center">
                                  <div className="w-14 h-14 rounded-full border-2 border-dashed border-primary/30 flex items-center justify-center mb-6">
                                     <div className="w-2.5 h-2.5 bg-primary/40 rounded-full animate-ping"></div>
                                  </div>
                                  <p className="text-[10px] uppercase font-black tracking-[0.25em] text-muted-foreground animate-pulse">Scanning Pixels...</p>
                                </div>
                              )}
                            </div>

                            <div className="p-8 border-t bg-muted/20">
                              <Button 
                                onClick={stopVideoDetection} 
                                variant="destructive" 
                                className="w-full h-14 rounded-2xl text-[11px] font-black uppercase tracking-[0.2em] bg-red-500/10 hover:bg-red-500 hover:text-white border border-red-500/20 text-red-500 shadow-lg transition-all duration-300"
                              >
                                <Square className="w-4 h-4 mr-3 fill-current" />
                                Terminate Stream
                              </Button>
                            </div>
                          </div>
                        )}
                      </div>
                    </DialogContent>

                  </Dialog>
                </div>
              )}
            </div>

            {/* Content Tabs (Converted to conditional rendering) */}

            {/* Image Detection */}
            {activeTab === 'image' && (
              <div className="space-y-6">
                <div className="grid lg:grid-cols-3 gap-6">
                  
                  {/* Primary Work Area (Left - 2/3 width) */}
                  <div className="lg:col-span-2 space-y-6 flex flex-col h-full min-h-[500px]">
                    {annotatedImage ? (
                      <Card className="bg-card border-border shadow-sm flex-1 flex flex-col mt-0 h-full max-h-[calc(100vh-12rem)]">
                        <CardHeader className="flex flex-row items-center justify-between py-3 px-4 border-b shrink-0">
                          <div className="space-y-1">
                            <CardTitle className="text-lg">Image Analysis</CardTitle>
                          </div>
                          <Button 
                            variant="secondary" 
                            size="sm"
                            onClick={() => {
                              setAnnotatedImage(null);
                              setUploadedImage(null);
                              setDetectionResults(null);
                            }}
                          >
                            Upload New Image
                          </Button>
                        </CardHeader>
                        <CardContent className="p-0 flex items-center justify-center bg-muted/10 flex-1 overflow-hidden relative group/canvas">
                          <div className="relative w-full h-full flex items-center justify-center">
                            <img 
                              src={annotatedImage} 
                              alt="Annotated" 
                              className="max-w-full max-h-full object-contain"
                              onLoad={(e) => {
                                const img = e.target;
                                img.dataset.realWidth = img.naturalWidth;
                                img.dataset.realHeight = img.naturalHeight;
                              }}
                            />
                            
                            {/* Highlight Overlay */}
                            {detectionResults && hoveredDetectionIndex !== null && detectionResults.detections[hoveredDetectionIndex] && (
                              <div 
                                className="absolute border-[6px] border-white shadow-[0_0_20px_rgba(255,255,255,0.8)] pointer-events-none z-10 transition-all duration-200 rounded-sm"
                                style={(function() {
                                  const det = detectionResults.detections[hoveredDetectionIndex];
                                  const bbox = det.bbox;
                                  
                                  // This is a simplified calculation that assumes the image fills the container using object-contain
                                  // For a more robust solution, we'd need a Ref and a resize listener, but this works well for most cases
                                  return {
                                    left: `${(bbox.x1 / detectionResults.original_shape[1]) * 100}%`,
                                    top: `${(bbox.y1 / detectionResults.original_shape[0]) * 100}%`,
                                    width: `${((bbox.x2 - bbox.x1) / detectionResults.original_shape[1]) * 100}%`,
                                    height: `${((bbox.y2 - bbox.y1) / detectionResults.original_shape[0]) * 100}%`,
                                    boxShadow: `0 0 0 6px ${getColorForClass(hoveredDetectionIndex)}, 0 0 40px ${getColorForClass(hoveredDetectionIndex)}80`
                                  };
                                })()}
                              />
                            )}
                          </div>
                        </CardContent>
                      </Card>
                    ) : (
                      <Card className="bg-card border-border shadow-sm flex-1 flex flex-col">
                        <CardHeader className="shrink-0">
                          <CardTitle className="flex items-center gap-2">
                            <Upload className="w-5 h-5" />
                            Upload Image
                          </CardTitle>
                          <CardDescription>
                            Select an image to detect objects
                          </CardDescription>
                        </CardHeader>
                        <CardContent className="space-y-4 flex-1 flex flex-col">
                          <div 
                            className="flex-1 border-2 border-dashed border-border rounded-lg p-12 text-center hover:border-primary transition-colors cursor-pointer flex flex-col items-center justify-center min-h-[300px]"
                            onClick={() => fileInputRef.current?.click()}
                          >
                            {uploadedImage ? (
                              <img src={uploadedImage} alt="Uploaded" className="max-h-[400px] object-contain mx-auto rounded-lg" />
                            ) : (
                              <div className="space-y-3">
                                <Upload className="w-12 h-12 mx-auto text-muted-foreground" />
                                <p className="text-secondary-foreground font-medium">Click to upload an image</p>
                                <p className="text-sm text-muted-foreground">PNG, JPG, GIF up to 10MB</p>
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
                            className="w-full bg-primary text-primary-foreground hover:bg-primary/90 mt-4 h-12 text-lg shrink-0"
                          >
                            {isLoading ? 'Detecting...' : 'Select Image'}
                          </Button>
                        </CardContent>
                      </Card>
                    )}
                  </div>

                  {/* Inspector Panel (Right - 1/3 width) */}
                  <div className="lg:col-span-1 border rounded-lg bg-card shadow-sm flex flex-col h-full lg:max-h-[min(800px,calc(100vh-12rem))] min-h-[500px]">
                    <div className="p-4 border-b shrink-0 bg-muted/10">
                      <h3 className="flex items-center gap-2 font-semibold text-foreground">
                        <Target className="w-4 h-4" />
                        Inspector
                      </h3>
                    </div>
                    
                    <div className="flex-1 overflow-y-auto p-0 flex flex-col">
                      {isLoading ? (
                        <div className="flex flex-col items-center justify-center h-full p-8 space-y-4">
                          <div className="animate-spin rounded-full w-8 h-8 border-2 border-primary border-t-transparent"></div>
                          <span className="text-sm text-secondary-foreground">Analyzing image...</span>
                        </div>
                      ) : detectionResults ? (
                        <div className="flex flex-col h-full">
                          {/* Header Stats */}
                          <div className="px-4 py-3 border-b bg-card shrink-0 flex items-center justify-between">
                            <span className="text-sm font-medium text-foreground">
                              {detectionResults.detections.length} objects found
                            </span>
                            <Badge variant="outline" className="text-xs font-mono">
                              {detectionResults.inference_time?.toFixed(2)}s
                            </Badge>
                          </div>
                          
                          {/* Dense Table of results */}
                          <div className="flex-1 overflow-y-auto p-2 space-y-1">
                            {detectionResults.detections.length > 0 ? (
                              detectionResults.detections.map((detection, index) => (
                                <div 
                                  key={index} 
                                  onMouseEnter={() => setHoveredDetectionIndex(index)}
                                  onMouseLeave={() => setHoveredDetectionIndex(null)}
                                  onClick={() => setHoveredDetectionIndex(index === hoveredDetectionIndex ? null : index)}
                                  className={`flex items-center justify-between px-3 py-2 rounded cursor-pointer transition-all group ${index === hoveredDetectionIndex ? 'bg-primary/10 border-primary/20 shadow-inner' : 'hover:bg-muted'}`}
                                >
                                  <div className="flex items-center gap-3">
                                    <div className={`w-2.5 h-2.5 rounded-sm transition-transform ${index === hoveredDetectionIndex ? 'scale-125' : ''}`} style={{ backgroundColor: getColorForClass(index) }}></div>
                                    <span className={`text-sm font-medium capitalize truncate max-w-[120px] ${index === hoveredDetectionIndex ? 'text-primary' : 'text-foreground'}`} title={detection.class_name}>
                                      {detection.class_name || 'Unknown'}
                                    </span>
                                  </div>
                                  <div className="flex items-center gap-2">
                                    <span className={`text-[10px] text-muted-foreground transition-opacity ${index === hoveredDetectionIndex ? 'opacity-100' : 'opacity-0 group-hover:opacity-100'}`}>
                                      {detection.bbox?.width || 0}×{detection.bbox?.height || 0}
                                    </span>
                                    <Badge variant="secondary" className={`font-mono text-xs transition-colors ${index === hoveredDetectionIndex ? 'bg-primary text-primary-foreground border-transparent' : 'bg-primary/10 text-primary border-transparent'}`}>
                                      {detection.confidence_percentage ? 
                                        `${detection.confidence_percentage}%` : 
                                        `${(detection.confidence * 100).toFixed(1)}%`
                                      }
                                    </Badge>
                                  </div>
                                </div>
                              ))
                            ) : (
                              <div className="p-6 text-center text-sm text-muted-foreground">
                                No objects detected above confidence threshold.
                              </div>
                            )}
                          </div>

                          {/* Summary Statistics */}
                          {detectionResults.summary && (
                            <div className="shrink-0 p-4 border-t bg-card">
                              <div className="text-[10px] font-semibold text-muted-foreground uppercase tracking-wider mb-3">Summary</div>
                              <div className="grid grid-cols-2 gap-y-4 gap-x-2 text-xs">
                                <div>
                                  <span className="text-muted-foreground block mb-0.5">Avg Conf</span>
                                  <span className="font-medium text-foreground">
                                    {detectionResults.summary.average_confidence ? 
                                      (detectionResults.summary.average_confidence * 100).toFixed(1) : 'N/A'}%
                                  </span>
                                </div>
                                <div>
                                  <span className="text-muted-foreground block mb-0.5">Classes</span>
                                  <span className="font-medium text-foreground">{detectionResults.summary.object_classes?.length || 0}</span>
                                </div>
                              </div>
                            </div>
                          )}
                          
                          <div className="shrink-0 p-4 border-t border-border bg-muted/5">
                            <Button onClick={downloadResults} variant="default" className="w-full text-sm">
                              <Download className="w-4 h-4 mr-2" />
                              Download Result
                            </Button>
                          </div>
                        </div>
                      ) : (
                        <div className="h-full flex items-center justify-center p-8 text-center text-muted-foreground text-sm">
                          Upload an image to inspect detections.
                        </div>
                      )}
                    </div>
                  </div>

                </div>
              </div>
            )}

            {activeTab === 'video' && (
              <div className="grid lg:grid-cols-3 gap-8 min-h-[calc(100vh-200px)]">
                {/* Primary Workspace (2/3) */}
                <div className="lg:col-span-2 flex flex-col gap-8">
                  <div className="flex flex-col flex-1 min-h-0">
                    {/* Video Upload Section - Spanning full 2/3 */}
                    <Card className={`bg-card border-border shadow-md flex flex-col overflow-hidden border-2 border-primary/5 group/card hover:border-primary/20 transition-all ${!videoResults ? 'h-full' : ''}`}>
                      <CardHeader className="py-5 border-b bg-muted/20">
                        <CardTitle className="text-sm font-bold flex items-center justify-between">
                          <span className="flex items-center gap-2">
                            <Video className="w-4 h-4 text-primary" />
                            File Analysis Workspace
                          </span>
                          <div className="flex gap-2">
                            <Badge variant="outline" className="text-[10px] uppercase tracking-wider bg-background">MP4/AVI/MOV</Badge>
                            <Badge className="bg-primary/10 text-primary border-none text-[10px]">Cloud Store</Badge>
                          </div>
                        </CardTitle>
                        <CardDescription className="text-xs">Upload forensic samples for high-accuracy batch processing</CardDescription>
                      </CardHeader>
                      <CardContent className="p-8 flex flex-col items-center justify-center flex-1 space-y-6">
                        {!videoResults && !videoProcessing && (
                          <div 
                            className="border-2 border-dashed border-border rounded-2xl p-16 w-full max-w-2xl text-center hover:border-primary hover:bg-primary/5 transition-all cursor-pointer group/upload flex flex-col items-center justify-center"
                            onClick={() => videoFileInputRef.current?.click()}
                          >
                            <div className="p-6 bg-primary/10 rounded-full mb-6 group-hover/upload:scale-110 group-hover/upload:bg-primary/20 transition-all duration-300">
                              <Upload className="w-12 h-12 text-primary" />
                            </div>
                            <h3 className="text-xl font-bold text-foreground">Ingest Video Source</h3>
                            <p className="text-sm text-muted-foreground mt-2 max-w-sm">Drop your file here or click to browse. Files up to 100MB are supported for direct processing.</p>
                            
                            <div className="mt-8 grid grid-cols-2 gap-4 w-full max-w-xs">
                              <div className="p-3 bg-muted/40 rounded-xl border border-border/50 text-[10px] font-bold uppercase tracking-tighter text-muted-foreground">Local Drive</div>
                              <div className="p-3 bg-muted/40 rounded-xl border border-border/50 text-[10px] font-bold uppercase tracking-tighter text-muted-foreground">Network URI</div>
                            </div>
                          </div>
                        )}

                        {videoProcessing && !videoResults && (
                          <div className="w-full max-w-md space-y-6 text-center py-10">
                            <div className="relative inline-block">
                              <div className="w-20 h-20 animate-spin rounded-full border-4 border-primary/20 border-t-primary"></div>
                              <div className="absolute inset-0 flex items-center justify-center">
                                <Activity className="w-8 h-8 text-primary animate-pulse" />
                              </div>
                            </div>
                            <div>
                              <h3 className="text-lg font-bold">Processing Stream...</h3>
                              <p className="text-sm text-muted-foreground mt-1">Analyzing frames and identifying targets using {selectedModel}</p>
                            </div>
                            <Progress value={45} className="h-2" />
                            <div className="flex justify-between text-[10px] font-bold uppercase tracking-widest text-muted-foreground bg-muted/40 p-2 rounded">
                              <span>Buffer Size: 64MB</span>
                              <span>Frames: 1240</span>
                            </div>
                          </div>
                        )}

                        {videoResults && (
                          <div className="w-full flex-1 flex flex-col min-h-0 bg-slate-950 rounded-[2.5rem] border-2 border-primary/10 overflow-hidden shadow-[0_40px_80px_-20px_rgba(0,0,0,0.6)] ring-1 ring-white/10 group/result animate-in fade-in zoom-in duration-700">
                             <div className="p-6 border-b border-white/5 bg-white/[0.02] flex items-center justify-between">
                                <div className="flex items-center gap-4">
                                  <div className="p-3 bg-primary/10 rounded-xl shadow-inner ring-1 ring-primary/20">
                                    <Play className="w-5 h-5 text-primary fill-primary/30" />
                                  </div>
                                  <div className="flex flex-col">
                                    <span className="text-[12px] font-black text-white/95 uppercase tracking-[0.25em] font-outfit">SYSTEM OUTPUT</span>
                                    <span className="text-[10px] text-slate-500 font-bold uppercase tracking-widest opacity-80">Rendered with active neural heuristics</span>
                                  </div>
                                </div>
                                <Badge variant="outline" className="text-primary border-primary/40 text-[10px] font-black h-7 bg-primary/5 uppercase tracking-[0.2em] px-4 rounded-full shadow-[0_0_20px_rgba(198,93,60,0.2)]">Analysis Component Ready</Badge>
                             </div>
                             
                             <div className="relative flex-1 bg-black flex items-center justify-center overflow-hidden h-[480px]">
                               <video 
                                 src={`${BACKEND_URL}/processed_videos/${videoResults.output_filename}`}
                                 className="w-full h-full object-contain"
                                 controls
                                 autoPlay
                                 loop
                               />
                               <div className="absolute top-6 left-6 p-3 bg-slate-950/40 backdrop-blur-xl rounded-xl border border-white/10 z-10 pointer-events-none shadow-2xl">
                                 <div className="flex items-center gap-3">
                                   <div className="w-2.5 h-2.5 rounded-full bg-primary shadow-[0_0_12px_rgba(198,93,60,0.8)] animate-pulse"></div>
                                   <span className="text-[11px] font-black text-white uppercase tracking-[0.3em] font-outfit">PROCESED TELEMETRY</span>
                                 </div>
                               </div>
                               
                               <div className="absolute bottom-6 right-6 p-2 bg-black/40 backdrop-blur-md rounded-lg border border-white/5 z-10 pointer-events-none opacity-0 group-hover/result:opacity-100 transition-opacity duration-300">
                                 <span className="text-[8px] font-mono text-slate-400 font-bold uppercase tracking-widest">Secure Handshake: {videoResults.output_filename?.substring(0, 12)}...</span>
                               </div>
                             </div>


                             <div className="p-8 bg-slate-900/60 border-t border-white/10 flex gap-6 backdrop-blur-2xl">
                                <Button className="flex-1 bg-primary text-primary-foreground font-black text-[12px] uppercase tracking-[0.3em] shadow-[0_20px_40px_-10px_rgba(198,93,60,0.4)] h-14 rounded-2xl group/btn hover:scale-[1.02] active:scale-[0.98] transition-all border-none" onClick={downloadProcessedVideo}>
                                   <Download className="w-5 h-5 mr-4 group-hover/btn:translate-y-1 transition-transform" /> 
                                   EXPORT PROCESS ASSET
                                </Button>
                                <Button variant="outline" className="h-14 px-10 font-bold text-[11px] uppercase tracking-[0.2em] rounded-2xl bg-white/5 hover:bg-white/10 transition-all border-white/10 text-white/70 hover:text-white" onClick={() => setVideoResults(null)}>
                                   NEW ANALYSIS
                                </Button>
                             </div>
                          </div>
                        )}


                        <input
                          type="file"
                          ref={videoFileInputRef}
                          onChange={handleVideoUpload}
                          accept="video/*"
                          className="hidden"
                        />
                      </CardContent>
                    </Card>

                    {videoResults && (
                      <div className="mt-8 animate-in slide-in-from-bottom duration-1000 delay-500 fill-mode-both">
                        <div className="flex items-center justify-between mb-6 px-1">
                          <div className="flex flex-col gap-1">
                            <h4 className="text-[12px] font-black text-primary uppercase tracking-[0.4em] font-outfit">Inventory Analysis Metrics</h4>
                            <p className="text-[11px] text-muted-foreground font-medium">Categorized object detection with aggregate confidence scores</p>
                          </div>
                          <Badge variant="secondary" className="bg-primary/10 text-primary border-none text-[11px] font-black h-8 px-4 rounded-xl uppercase tracking-widest">{videoResults.detected_classes?.length || 0} Categories</Badge>
                        </div>
                        
                        <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                          {videoResults.detected_classes && videoResults.detected_classes.length > 0 ? (
                            videoResults.detected_classes.map((detection, idx) => (
                              <div key={idx} className="flex items-center gap-5 p-5 bg-card border border-border rounded-[1.5rem] shadow-sm hover:shadow-md hover:border-primary/30 transition-all duration-500 cursor-default group/ribbon overflow-hidden relative">
                                <div className="absolute top-0 right-0 w-20 h-20 bg-primary/5 rounded-full -mr-10 -mt-10 transition-transform group-hover/ribbon:scale-110"></div>
                                <div className="w-2.5 h-2.5 rounded-full shrink-0 shadow-[0_0_15px_rgba(198,93,60,0.3)] bg-primary relative z-10 transition-transform group-hover/ribbon:scale-125"></div>
                                <div className="flex flex-col relative z-10">
                                  <span className="text-[14px] font-black text-foreground leading-none mb-2 uppercase tracking-tight">{detection.class_name}</span>
                                  <div className="flex items-center gap-2">
                                    <div className="w-10 h-1.5 bg-muted rounded-full overflow-hidden">
                                      <div className="h-full bg-primary" style={{ width: `${detection.avg_confidence * 100}%` }}></div>
                                    </div>
                                    <span className="text-[10px] font-black text-primary tracking-tighter">{(detection.avg_confidence * 100).toFixed(0)}%</span>
                                  </div>
                                </div>
                              </div>
                            ))
                          ) : (
                            <div className="col-span-full py-12 text-center border-2 border-dashed border-border rounded-3xl opacity-40">
                              <Activity className="w-8 h-8 text-muted-foreground mx-auto mb-3 animate-pulse" />
                              <p className="text-[10px] uppercase font-black tracking-[0.3em] text-muted-foreground">No classification signatures detected</p>
                            </div>
                          )}
                        </div>
                      </div>
                    )}
                  </div>
                </div>

                {/* Right Workspace: Inspector (1/3) */}
                <div className="flex flex-col gap-6 min-h-0">
                  {/* Results Panel */}
                  <Card className="bg-card border-border shadow-lg flex flex-col h-full overflow-hidden border-2 border-primary/10">
                    <CardHeader className="py-3 flex-row items-center justify-between border-b bg-muted/10 shrink-0">
                      <div>
                        <CardTitle className="text-sm font-bold flex items-center gap-2">
                          <Activity className="w-4 h-4 text-primary" />
                          Inspector
                        </CardTitle>
                        <CardDescription className="text-[11px]">Real-time stream and file metrics</CardDescription>
                      </div>
                    </CardHeader>
                    <CardContent className="p-0 flex flex-col flex-1 min-h-0">
                      {/* Video Process Results or Camera Live Results */}
                      {isVideoStreaming ? (
                        <div className="flex flex-col h-full divide-y divide-border/50">
                          {/* Feed Meta */}
                          <div className="p-4 bg-muted/10">
                            <div className="flex justify-between items-center mb-3">
                              <span className="text-[10px] uppercase font-bold text-muted-foreground tracking-widest">Active Stream</span>
                              <Badge variant="outline" className="bg-red-500/10 text-red-500 border-red-500/20 animate-pulse text-[9px]">LIVE</Badge>
                            </div>
                            <div className="grid grid-cols-2 gap-2">
                              <div className="bg-card p-2 rounded-md border border-border/40 shadow-sm">
                                <div className="text-[10px] text-muted-foreground mb-0.5">Found</div>
                                <div className="text-lg font-mono font-bold text-primary">{videoDetections.length}</div>
                              </div>
                              <div className="bg-card p-2 rounded-md border border-border/40 shadow-sm">
                                <div className="text-[10px] text-muted-foreground mb-0.5">Confidence</div>
                                <div className="text-lg font-mono font-bold text-primary">
                                  {videoDetections.length > 0 ? (videoDetections.reduce((a, b) => a + b.confidence, 0) / videoDetections.length * 100).toFixed(0) : 0}%
                                </div>
                              </div>
                            </div>
                          </div>
                          
                          {/* Live Detections List */}
                          <div className="flex-1 overflow-y-auto p-2 scrollbar-thin">
                            {videoDetections.length > 0 ? (
                              <div className="space-y-1">
                                {videoDetections.map((detection, index) => (
                                  <div key={index} className="flex items-center justify-between px-3 py-2 rounded bg-muted/30 hover:bg-muted/50 border border-transparent transition-all">
                                    <div className="flex items-center gap-3 overflow-hidden">
                                      <div className="w-2 h-2 rounded-full shrink-0" style={{ backgroundColor: getColorForClass(index) }}></div>
                                      <span className="text-sm font-medium truncate">{detection.class_name}</span>
                                    </div>
                                    <Badge variant="secondary" className="font-mono text-[10px] py-0 px-1.5 h-5 bg-primary/10 text-primary border-none">
                                      {(detection.confidence * 100).toFixed(1)}%
                                    </Badge>
                                  </div>
                                ))}
                              </div>
                            ) : (
                              <div className="h-full flex flex-col items-center justify-center py-10 opacity-30">
                                <div className="w-12 h-12 bg-muted rounded-full flex items-center justify-center mb-2">
                                  <Search className="w-5 h-5" />
                                </div>
                                <p className="text-xs text-center px-4">Detecting objects...</p>
                              </div>
                            )}
                          </div>
                        </div>
                      ) : videoResults ? (
                        <div className="flex flex-col h-full divide-y divide-border/50">
                          <div className="p-4 bg-muted/10 space-y-4">
                            <div className="grid grid-cols-2 gap-3">
                              <div className="bg-card p-3 rounded-lg border border-border/40 shadow-sm text-center">
                                <div className="text-[10px] text-muted-foreground mb-1 uppercase tracking-tighter">Frames</div>
                                <div className="text-xl font-bold font-mono text-primary">{videoResults.processed_frames}</div>
                              </div>
                              <div className="bg-card p-3 rounded-lg border border-border/40 shadow-sm text-center">
                                <div className="text-[10px] text-muted-foreground mb-1 uppercase tracking-tighter">Total Hits</div>
                                <div className="text-xl font-bold font-mono text-primary">{videoResults.total_detections}</div>
                              </div>
                            </div>
                            
                            <div className="space-y-2 py-2 border-y border-border/50">
                              <div className="flex justify-between text-xs items-center">
                                <span className="text-muted-foreground">Speed:</span>
                                <span className="font-mono">{videoResults.processing_time.toFixed(2)}s</span>
                              </div>
                              <div className="flex justify-between text-xs items-center">
                                <span className="text-muted-foreground">Resolution:</span>
                                <span className="font-mono">{videoResults.resolution}</span>
                              </div>
                              <div className="flex justify-between text-xs items-center">
                                <span className="text-muted-foreground">Density:</span>
                                <span className="font-mono">{videoResults.avg_detections_per_frame.toFixed(2)}/f</span>
                              </div>
                            </div>
                          </div>

                          <div className="p-4 flex-1 flex flex-col justify-end bg-card">
                            <Button onClick={downloadProcessedVideo} className="w-full h-12 bg-primary font-black uppercase tracking-[0.2em] text-[11px] shadow-2xl shadow-primary/30 hover:scale-[1.02] active:scale-[0.98] transition-all rounded-xl">
                              <Download className="w-4 h-4 mr-3" />
                              Export Annotated Asset
                            </Button>
                          </div>
                        </div>
                      ) : (
                        <div className="h-full flex flex-col items-center justify-center py-20 opacity-30 text-center px-6">
                          <div className="w-16 h-16 bg-muted rounded-full flex items-center justify-center mb-4">
                            <Activity className="w-8 h-8" />
                          </div>
                          <h4 className="text-sm font-bold text-foreground">Awaiting Input</h4>
                          <p className="text-xs text-muted-foreground mt-1">Start a camera stream or upload a file to see analytics</p>
                        </div>
                      )}
                    </CardContent>
                  </Card>
                </div>
              </div>
            )}

            {/* Batch Processing Tab */}
            {activeTab === 'batch' && (
              <BatchTab
                batchInputRef={batchInputRef}
                batchFiles={batchFiles}
                batchProcessing={batchProcessing}
                batchResults={batchResults}
                handleBatchUpload={handleBatchUpload}
                downloadBatchResults={downloadBatchResults}
                getColorForClass={getColorForClass}
                backendUrl={BACKEND_URL}
              />
            )}

            {/* Model Management Tab */}
            {activeTab === 'models' && (
              <ModelTab
                modelInfo={modelInfo}
                selectedModel={selectedModel}
                modelUploading={modelUploading}
                modelInputRef={modelInputRef}
                switchModel={switchModel}
                handleCustomModelUpload={handleCustomModelUpload}
              />
            )}
          </div>
        </div>
      </main>
    </div>
  );
}

export default App;
