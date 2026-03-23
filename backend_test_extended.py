#!/usr/bin/env python3
"""
Extended Backend API Testing for VisionFlow App
Tests batch processing, model management, and video processing endpoints
"""

import requests
import json
import sys
import time
import tempfile
import os
from datetime import datetime
from io import BytesIO
from PIL import Image, ImageDraw
import cv2
import numpy as np

# Prefer an explicit environment override, otherwise use local development.
BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")
API_BASE = f"{BACKEND_URL}/api"

class VisionFlowExtendedTester:
    def __init__(self):
        self.tests_run = 0
        self.tests_passed = 0
        self.test_results = []
        
    def log_test(self, name, success, details="", response_data=None):
        """Log test results"""
        self.tests_run += 1
        if success:
            self.tests_passed += 1
            print(f"✅ {name}: PASSED")
        else:
            print(f"❌ {name}: FAILED - {details}")
            
        self.test_results.append({
            "test": name,
            "success": success,
            "details": details,
            "response_data": response_data
        })
        
        if details:
            print(f"   Details: {details}")
        print()

    def create_test_image(self, width=640, height=480, filename="test_image.jpg"):
        """Create a simple test image with some shapes"""
        img = Image.new('RGB', (width, height), color='white')
        draw = ImageDraw.Draw(img)
        
        # Draw some shapes that might be detected as objects
        draw.rectangle([50, 50, 150, 150], fill='red', outline='black', width=3)
        draw.ellipse([200, 100, 300, 200], fill='blue', outline='black', width=3)
        draw.rectangle([350, 200, 450, 300], fill='green', outline='black', width=3)
        
        # Save to temporary file
        temp_path = f"/tmp/{filename}"
        img.save(temp_path, format='JPEG')
        return temp_path

    def create_test_video(self, filename="test_video.mp4", duration=2, fps=10):
        """Create a simple test video"""
        temp_path = f"/tmp/{filename}"
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_path, fourcc, fps, (320, 240))
        
        # Generate frames
        for frame_num in range(duration * fps):
            # Create a simple frame with moving rectangle
            frame = np.ones((240, 320, 3), dtype=np.uint8) * 255
            x = (frame_num * 10) % 300
            cv2.rectangle(frame, (x, 50), (x + 50, 100), (0, 0, 255), -1)
            out.write(frame)
        
        out.release()
        return temp_path

    def test_batch_processing(self):
        """Test batch image processing endpoint"""
        try:
            # Create multiple test images
            image_paths = []
            files_data = []
            
            for i in range(3):  # Test with 3 images
                img_path = self.create_test_image(filename=f"batch_test_{i}.jpg")
                image_paths.append(img_path)
                
                with open(img_path, 'rb') as f:
                    files_data.append(('files', (f"batch_test_{i}.jpg", f.read(), 'image/jpeg')))
            
            print("   Uploading 3 test images for batch processing...")
            response = requests.post(f"{API_BASE}/detect/batch/images", files=files_data, timeout=60)
            success = response.status_code == 200
            
            if success:
                data = response.json()
                total_images = data.get('total_images', 0)
                processed_images = data.get('processed_images', 0)
                total_detections = data.get('total_detections', 0)
                processing_time = data.get('processing_time', 0)
                
                details = f"Images: {total_images}, Processed: {processed_images}, Detections: {total_detections}, Time: {processing_time:.2f}s"
                
                # Test download endpoint if archive is available
                if data.get('results_archive'):
                    try:
                        download_response = requests.get(f"{API_BASE}/download/batch/{data['results_archive']}", timeout=30)
                        if download_response.status_code == 200:
                            details += f", Archive size: {len(download_response.content)} bytes"
                        else:
                            details += f", Archive download failed: {download_response.status_code}"
                    except Exception as e:
                        details += f", Archive download error: {str(e)}"
                        
            else:
                details = f"Status: {response.status_code}, Response: {response.text[:200]}"
                
            # Cleanup
            for img_path in image_paths:
                try:
                    os.remove(img_path)
                except:
                    pass
                    
            self.log_test("Batch Processing", success, details, response.json() if success else None)
            return success
            
        except Exception as e:
            self.log_test("Batch Processing", False, f"Exception: {str(e)}")
            return False

    def test_model_management(self):
        """Test model management endpoints"""
        try:
            # Test model info (already tested in basic tests, but let's verify again)
            response = requests.get(f"{API_BASE}/model/info", timeout=15)
            model_info_success = response.status_code == 200
            
            if model_info_success:
                model_info = response.json()
                active_model = model_info.get('active_model', 'unknown')
                available_models = model_info.get('available_models', {})
                available_classes = model_info.get('available_classes', [])
                
                details = f"Active: {active_model}, Models: {len(available_models)}, Classes: {len(available_classes)}"
                
                # Test model switching (try to switch to the same model)
                switch_response = requests.post(f"{API_BASE}/model/switch?model_name={active_model}", timeout=15)
                switch_success = switch_response.status_code == 200
                
                if switch_success:
                    switch_data = switch_response.json()
                    details += f", Switch: {switch_data.get('message', 'OK')}"
                else:
                    details += f", Switch failed: {switch_response.status_code}"
                    
                overall_success = model_info_success and switch_success
            else:
                details = f"Model info failed: {response.status_code}"
                overall_success = False
                
            self.log_test("Model Management", overall_success, details)
            return overall_success
            
        except Exception as e:
            self.log_test("Model Management", False, f"Exception: {str(e)}")
            return False

    def test_video_processing(self):
        """Test video file processing endpoint"""
        try:
            # Create a test video
            video_path = self.create_test_video()
            
            print("   Uploading test video for processing...")
            
            with open(video_path, 'rb') as f:
                files = {'file': ('test_video.mp4', f, 'video/mp4')}
                response = requests.post(f"{API_BASE}/detect/video", files=files, timeout=120)
            
            success = response.status_code == 200
            
            if success:
                data = response.json()
                processed_frames = data.get('processed_frames', 0)
                total_detections = data.get('total_detections', 0)
                processing_time = data.get('processing_time', 0)
                fps = data.get('fps', 0)
                resolution = data.get('resolution', 'unknown')
                output_filename = data.get('output_filename', '')
                
                details = f"Frames: {processed_frames}, Detections: {total_detections}, Time: {processing_time:.2f}s, FPS: {fps}, Resolution: {resolution}"
                
                # Test download endpoint if output file is available
                if output_filename:
                    try:
                        download_response = requests.get(f"{API_BASE}/download/video/{output_filename}", timeout=60)
                        if download_response.status_code == 200:
                            details += f", Video size: {len(download_response.content)} bytes"
                        else:
                            details += f", Video download failed: {download_response.status_code}"
                    except Exception as e:
                        details += f", Video download error: {str(e)}"
                        
            else:
                details = f"Status: {response.status_code}, Response: {response.text[:200]}"
                
            # Cleanup
            try:
                os.remove(video_path)
            except:
                pass
                
            self.log_test("Video Processing", success, details, response.json() if success else None)
            return success
            
        except Exception as e:
            self.log_test("Video Processing", False, f"Exception: {str(e)}")
            return False

    def test_video_history(self):
        """Test video processing history endpoint"""
        try:
            response = requests.get(f"{API_BASE}/video/history?limit=5", timeout=10)
            success = response.status_code == 200
            
            if success:
                data = response.json()
                details = f"History entries: {len(data)}"
            else:
                details = f"Status: {response.status_code}"
                
            self.log_test("Video History", success, details)
            return success
            
        except Exception as e:
            self.log_test("Video History", False, f"Exception: {str(e)}")
            return False

    def test_download_endpoints(self):
        """Test download endpoints with invalid IDs (should return 404)"""
        try:
            # Test image download with invalid ID
            response = requests.get(f"{API_BASE}/download/image/invalid_id", timeout=10)
            image_404 = response.status_code == 404
            
            # Test batch download with invalid archive
            response = requests.get(f"{API_BASE}/download/batch/invalid_archive.zip", timeout=10)
            batch_404 = response.status_code == 404
            
            # Test video download with invalid filename
            response = requests.get(f"{API_BASE}/download/video/invalid_video.mp4", timeout=10)
            video_404 = response.status_code == 404
            
            success = image_404 and batch_404 and video_404
            details = f"Image 404: {image_404}, Batch 404: {batch_404}, Video 404: {video_404}"
            
            self.log_test("Download Endpoints (404 Tests)", success, details)
            return success
            
        except Exception as e:
            self.log_test("Download Endpoints (404 Tests)", False, f"Exception: {str(e)}")
            return False

    def run_extended_tests(self):
        """Run all extended backend API tests"""
        print("🚀 Starting VisionFlow Extended API Tests")
        print(f"Backend URL: {BACKEND_URL}")
        print("=" * 60)
        
        # Phase 1: Batch Processing Tests
        print("📦 Phase 1: Batch Processing Tests")
        self.test_batch_processing()
        
        # Phase 2: Model Management Tests
        print("\n🧠 Phase 2: Model Management Tests")
        self.test_model_management()
        
        # Phase 3: Video Processing Tests
        print("\n🎥 Phase 3: Video Processing Tests")
        self.test_video_processing()
        self.test_video_history()
        
        # Phase 4: Download Endpoint Tests
        print("\n⬇️ Phase 4: Download Endpoint Tests")
        self.test_download_endpoints()
        
        # Summary
        print("\n" + "=" * 60)
        print("📊 EXTENDED TEST SUMMARY")
        print(f"Tests Run: {self.tests_run}")
        print(f"Tests Passed: {self.tests_passed}")
        print(f"Success Rate: {(self.tests_passed/self.tests_run)*100:.1f}%")
        
        # Show failed tests
        failed_tests = [t for t in self.test_results if not t['success']]
        if failed_tests:
            print(f"\n❌ Failed Tests ({len(failed_tests)}):")
            for test in failed_tests:
                print(f"  - {test['test']}: {test['details']}")
        else:
            print("\n🎉 All extended tests passed!")
        
        return self.tests_passed == self.tests_run

def main():
    """Main test execution"""
    tester = VisionFlowExtendedTester()
    
    try:
        success = tester.run_extended_tests()
        return 0 if success else 1
        
    except KeyboardInterrupt:
        print("\n⚠️  Tests interrupted by user")
        return 1
    except Exception as e:
        print(f"\n💥 Unexpected error: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
