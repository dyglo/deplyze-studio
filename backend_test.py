#!/usr/bin/env python3
"""
Comprehensive Backend API Testing for Logo Detection App
Tests all YOLO11n detection endpoints and WebSocket functionality
"""

import requests
import json
import sys
import time
import asyncio
import websockets
import base64
import cv2
import numpy as np
from datetime import datetime
from io import BytesIO
from PIL import Image, ImageDraw

# Use the public endpoint from frontend/.env
BACKEND_URL = "https://logodetector.preview.emergentagent.com"
API_BASE = f"{BACKEND_URL}/api"

class LogoDetectionAPITester:
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

    def create_test_image(self, width=640, height=480):
        """Create a simple test image with some shapes"""
        # Create a simple test image
        img = Image.new('RGB', (width, height), color='white')
        draw = ImageDraw.Draw(img)
        
        # Draw some shapes that might be detected as objects
        draw.rectangle([50, 50, 150, 150], fill='red', outline='black', width=3)
        draw.ellipse([200, 100, 300, 200], fill='blue', outline='black', width=3)
        draw.rectangle([350, 200, 450, 300], fill='green', outline='black', width=3)
        
        # Convert to bytes
        img_buffer = BytesIO()
        img.save(img_buffer, format='JPEG')
        img_buffer.seek(0)
        
        return img_buffer.getvalue()

    def test_api_root(self):
        """Test API root endpoint"""
        try:
            response = requests.get(f"{API_BASE}/", timeout=10)
            success = response.status_code == 200
            
            if success:
                data = response.json()
                details = f"Version: {data.get('version', 'Unknown')}"
            else:
                details = f"Status: {response.status_code}"
                
            self.log_test("API Root", success, details, response.json() if success else None)
            return success
            
        except Exception as e:
            self.log_test("API Root", False, f"Exception: {str(e)}")
            return False

    def test_model_info(self):
        """Test model info endpoint"""
        try:
            response = requests.get(f"{API_BASE}/model/info", timeout=15)
            success = response.status_code == 200
            
            if success:
                data = response.json()
                details = f"Device: {data.get('device')}, Classes: {len(data.get('available_classes', []))}"
            else:
                details = f"Status: {response.status_code}, Response: {response.text[:200]}"
                
            self.log_test("Model Info", success, details, response.json() if success else None)
            return success, response.json() if success else None
            
        except Exception as e:
            self.log_test("Model Info", False, f"Exception: {str(e)}")
            return False, None

    def test_image_detection(self):
        """Test image detection endpoint"""
        try:
            # Create test image
            test_image = self.create_test_image()
            
            # Prepare multipart form data
            files = {'file': ('test_image.jpg', test_image, 'image/jpeg')}
            
            print("   Uploading test image for detection...")
            response = requests.post(f"{API_BASE}/detect/image", files=files, timeout=30)
            success = response.status_code == 200
            
            if success:
                data = response.json()
                detections = data.get('detections', [])
                inference_time = data.get('inference_time', 0)
                details = f"Detections: {len(detections)}, Time: {inference_time:.3f}s"
                
                # Log detection details
                if detections:
                    print(f"   Found {len(detections)} detections:")
                    for i, det in enumerate(detections[:3]):  # Show first 3
                        print(f"     {i+1}. {det.get('class_name')} ({det.get('confidence', 0)*100:.1f}%)")
                else:
                    print("   No detections found (this is normal for test image)")
                    
            else:
                details = f"Status: {response.status_code}, Response: {response.text[:200]}"
                
            self.log_test("Image Detection", success, details, response.json() if success else None)
            return success, response.json() if success else None
            
        except Exception as e:
            self.log_test("Image Detection", False, f"Exception: {str(e)}")
            return False, None

    def test_annotated_image_detection(self):
        """Test annotated image detection endpoint"""
        try:
            # Create test image
            test_image = self.create_test_image()
            
            # Prepare multipart form data
            files = {'file': ('test_image.jpg', test_image, 'image/jpeg')}
            
            print("   Uploading test image for annotated detection...")
            response = requests.post(f"{API_BASE}/detect/image/annotated", files=files, timeout=30)
            success = response.status_code == 200
            
            if success:
                data = response.json()
                detections = data.get('detections', [])
                inference_time = data.get('inference_time', 0)
                has_annotated_image = 'annotated_image' in data and data['annotated_image']
                
                details = f"Detections: {len(detections)}, Time: {inference_time:.3f}s, Annotated: {has_annotated_image}"
                
                if has_annotated_image:
                    # Verify base64 image format
                    try:
                        base64_data = data['annotated_image']
                        # Try to decode to verify it's valid base64
                        decoded = base64.b64decode(base64_data)
                        print(f"   Annotated image size: {len(decoded)} bytes")
                    except Exception as e:
                        print(f"   Warning: Invalid base64 image data: {e}")
                        
            else:
                details = f"Status: {response.status_code}, Response: {response.text[:200]}"
                
            self.log_test("Annotated Image Detection", success, details, response.json() if success else None)
            return success
            
        except Exception as e:
            self.log_test("Annotated Image Detection", False, f"Exception: {str(e)}")
            return False

    def test_confidence_threshold_update(self):
        """Test confidence threshold update endpoint"""
        try:
            # Test valid threshold
            response = requests.post(f"{API_BASE}/model/confidence?threshold=0.5", timeout=10)
            success = response.status_code == 200
            
            if success:
                data = response.json()
                details = f"Message: {data.get('message', 'Updated')}"
            else:
                details = f"Status: {response.status_code}"
                
            self.log_test("Confidence Threshold Update", success, details)
            
            # Test invalid threshold (should fail)
            response = requests.post(f"{API_BASE}/model/confidence?threshold=1.5", timeout=10)
            invalid_success = response.status_code == 400  # Should return 400 for invalid threshold
            self.log_test("Invalid Confidence Threshold (Expected Fail)", invalid_success, 
                         f"Status: {response.status_code} (should be 400)")
            
            return success
            
        except Exception as e:
            self.log_test("Confidence Threshold Update", False, f"Exception: {str(e)}")
            return False

    def test_detection_history(self):
        """Test detection history endpoint"""
        try:
            response = requests.get(f"{API_BASE}/history?limit=5", timeout=10)
            success = response.status_code == 200
            
            if success:
                data = response.json()
                details = f"History entries: {len(data)}"
            else:
                details = f"Status: {response.status_code}"
                
            self.log_test("Detection History", success, details)
            return success
            
        except Exception as e:
            self.log_test("Detection History", False, f"Exception: {str(e)}")
            return False

    async def test_websocket_video_detection(self):
        """Test WebSocket video detection endpoint"""
        try:
            # Convert HTTPS to WSS for WebSocket
            ws_url = f"{BACKEND_URL.replace('https://', 'wss://')}/api/detect/video"
            print(f"   Connecting to WebSocket: {ws_url}")
            
            # Create a test frame (base64 encoded)
            test_frame = self.create_test_image(320, 240)  # Smaller for video
            test_frame_b64 = base64.b64encode(test_frame).decode('utf-8')
            
            async with websockets.connect(ws_url, timeout=15) as websocket:
                print("   WebSocket connected successfully")
                
                # Send test frame
                frame_data = {"frame": test_frame_b64}
                await websocket.send(json.dumps(frame_data))
                print("   Sent test frame")
                
                # Wait for response
                try:
                    response = await asyncio.wait_for(websocket.recv(), timeout=10)
                    data = json.loads(response)
                    
                    if data.get('type') == 'detection_result':
                        detections = data.get('detections', [])
                        inference_time = data.get('inference_time', 0)
                        details = f"Detections: {len(detections)}, Time: {inference_time:.3f}s"
                        success = True
                    else:
                        details = f"Unexpected response type: {data.get('type')}"
                        success = False
                        
                except asyncio.TimeoutError:
                    details = "Timeout waiting for WebSocket response"
                    success = False
                    
            self.log_test("WebSocket Video Detection", success, details)
            return success
            
        except Exception as e:
            self.log_test("WebSocket Video Detection", False, f"Exception: {str(e)}")
            return False

    def test_status_endpoints(self):
        """Test original status check endpoints"""
        try:
            # Test POST status
            status_data = {"client_name": f"test_client_{int(time.time())}"}
            response = requests.post(f"{API_BASE}/status", json=status_data, timeout=10)
            post_success = response.status_code == 200
            
            # Test GET status
            response = requests.get(f"{API_BASE}/status", timeout=10)
            get_success = response.status_code == 200
            
            if get_success:
                data = response.json()
                details = f"Status entries: {len(data)}"
            else:
                details = f"GET Status: {response.status_code}"
                
            overall_success = post_success and get_success
            self.log_test("Status Endpoints", overall_success, details)
            return overall_success
            
        except Exception as e:
            self.log_test("Status Endpoints", False, f"Exception: {str(e)}")
            return False

    def run_all_tests(self):
        """Run all backend API tests"""
        print("🚀 Starting Logo Detection API Tests")
        print(f"Backend URL: {BACKEND_URL}")
        print("=" * 60)
        
        # Phase 1: Basic API Tests
        print("📋 Phase 1: Basic API Tests")
        self.test_api_root()
        model_success, model_info = self.test_model_info()
        
        if model_info:
            print(f"   Model Info: {model_info.get('device')} device, {len(model_info.get('available_classes', []))} classes")
        
        # Phase 2: Detection Tests
        print("\n🎯 Phase 2: Detection Tests")
        detection_success, detection_data = self.test_image_detection()
        self.test_annotated_image_detection()
        self.test_confidence_threshold_update()
        
        # Phase 3: Data Management Tests
        print("\n📊 Phase 3: Data Management Tests")
        self.test_detection_history()
        self.test_status_endpoints()
        
        # Phase 4: WebSocket Tests
        print("\n🔌 Phase 4: WebSocket Tests")
        try:
            asyncio.run(self.test_websocket_video_detection())
        except Exception as e:
            self.log_test("WebSocket Video Detection", False, f"Asyncio error: {str(e)}")
        
        # Summary
        print("\n" + "=" * 60)
        print("📊 TEST SUMMARY")
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
            print("\n🎉 All tests passed!")
        
        return self.tests_passed == self.tests_run

def main():
    """Main test execution"""
    tester = LogoDetectionAPITester()
    
    try:
        success = tester.run_all_tests()
        return 0 if success else 1
        
    except KeyboardInterrupt:
        print("\n⚠️  Tests interrupted by user")
        return 1
    except Exception as e:
        print(f"\n💥 Unexpected error: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())