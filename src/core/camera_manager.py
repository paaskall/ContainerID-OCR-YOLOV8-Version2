import cv2
import threading
from queue import Queue
import time

class CameraManager:
    def __init__(self):
        self.cameras = {}
        self.threads = {}
        self.queues = {}
        self.is_running = False
        
        """ self.camera_sources = {
            0: "rtsp://admin:Qwerty123@10.35.61.115/video",
            1: "rtsp://admin:Qwerty123@10.35.61.116/video",  
            2: "rtsp://admin:Qwerty123@10.35.61.117/video",
            3: "rtsp://admin:Qwerty123@10.35.61.118/video",
            4: "rtsp://admin:Qwerty123@10.35.61.121/video",
            5: "rtsp://admin:Qwerty123@10.35.61.120/video"
        } """

        self.camera_sources = {
            0: "rtsp://admin:Qwerty123@10.35.61.108/video",
            1: "rtsp://admin:Qwerty123@10.35.61.113/video",  
            2: "rtsp://admin:Qwerty123@10.35.61.111/video",
            3: "rtsp://admin:Qwerty123@10.35.61.110/video",
            4: "rtsp://admin:Qwerty123@10.35.61.109/video",
            5: "rtsp://admin:Qwerty123@10.35.61.112/video"
        }


        print("Camera Manager initialized with 6 RTSP URLs")
        
    def add_camera(self, camera_id, source=None):
        """Add camera to management"""
        if source is None:
            source = self.camera_sources.get(camera_id, camera_id)
            
        self.cameras[camera_id] = {
            'source': source,
            'active': False,
            'cap': None,
            'fps': 0,
            'last_frame_time': time.time(),
            'retry_count': 0,
            'max_retries': 3
        }
        self.queues[camera_id] = Queue(maxsize=1)
        
    def start_camera(self, camera_id):
        """Start specific RTSP camera dengan optimized settings"""
        if camera_id not in self.cameras:
            self.add_camera(camera_id)
            
        if self.cameras[camera_id]['active']:
            return True
            
        try:
            source = self.cameras[camera_id]['source']
            print(f"Starting camera {camera_id}: {source}")
            
            cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
            
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            cap.set(cv2.CAP_PROP_FPS, 10)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
            
            cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)
            
            start_time = time.time()
            while not cap.isOpened() and (time.time() - start_time) < 10:
                time.sleep(0.1)
                
            if not cap.isOpened():
                print(f"Failed to open RTSP camera {camera_id} - Timeout")
                return False
                
            ret, test_frame = cap.read()
            if not ret:
                print(f"Failed to read from camera {camera_id} - No frame")
                cap.release()
                return False
                
            print(f"Camera {camera_id} connected successfully")
            
            self.cameras[camera_id]['cap'] = cap
            self.cameras[camera_id]['active'] = True
            self.cameras[camera_id]['retry_count'] = 0
            
            thread = threading.Thread(target=self._camera_worker, args=(camera_id,))
            thread.daemon = True
            thread.start()
            self.threads[camera_id] = thread
            
            return True
            
        except Exception as e:
            print(f"Error starting camera {camera_id}: {e}")
            self.cameras[camera_id]['retry_count'] += 1
            return False
            
    def _camera_worker(self, camera_id):
        """Worker thread untuk RTSP camera dengan robust error handling"""
        while self.cameras[camera_id]['active']:
            try:
                cap = self.cameras[camera_id]['cap']
                if cap is None or not cap.isOpened():
                    print(f"Camera {camera_id} reconnecting...")
                    self._reconnect_camera(camera_id)
                    time.sleep(2)
                    continue
                    
                ret, frame = cap.read()
                if ret:
                    self.cameras[camera_id]['retry_count'] = 0
                    
                    current_time = time.time()
                    time_diff = current_time - self.cameras[camera_id]['last_frame_time']
                    self.cameras[camera_id]['last_frame_time'] = current_time
                    self.cameras[camera_id]['fps'] = 1.0 / time_diff if time_diff > 0 else 0
                    
                    if self.queues[camera_id].full():
                        self.queues[camera_id].get()
                    self.queues[camera_id].put(frame)
                else:
                    self.cameras[camera_id]['retry_count'] += 1
                    print(f"Camera {camera_id} read failed - Retry {self.cameras[camera_id]['retry_count']}")
                    
                    if self.cameras[camera_id]['retry_count'] >= self.cameras[camera_id]['max_retries']:
                        print(f"Camera {camera_id} reconnecting due to failures...")
                        self._reconnect_camera(camera_id)
                    
            except Exception as e:
                print(f"Camera {camera_id} worker error: {e}")
                self.cameras[camera_id]['retry_count'] += 1
                time.sleep(1)
                
    def _reconnect_camera(self, camera_id):
        """Reconnect to camera"""
        print(f"Attempting to reconnect camera {camera_id}...")
        self.stop_camera(camera_id)
        time.sleep(3)
        self.start_camera(camera_id)
        
    def stop_camera(self, camera_id):
        """Stop specific camera"""
        if camera_id in self.cameras:
            self.cameras[camera_id]['active'] = False
            if self.cameras[camera_id]['cap']:
                self.cameras[camera_id]['cap'].release()
                self.cameras[camera_id]['cap'] = None
            print(f"Camera {camera_id} stopped")
            
    def stop_all_cameras(self):
        """Stop all cameras"""
        for camera_id in self.cameras:
            self.stop_camera(camera_id)
            
    def get_frame(self, camera_id):
        """Get latest frame from camera"""
        if (camera_id in self.queues and 
            not self.queues[camera_id].empty() and 
            self.cameras[camera_id]['active']):
            return self.queues[camera_id].get()
        return None
        
    def get_camera_fps(self, camera_id):
        """Get camera FPS"""
        if camera_id in self.cameras:
            return self.cameras[camera_id].get('fps', 0)
        return 0
        
    def get_camera_status(self, camera_id):
        """Get camera status"""
        if camera_id in self.cameras:
            return self.cameras[camera_id]['active']
        return False