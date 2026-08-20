"""
Multi-Camera RTSP Stream Hub & Multiplexer
==========================================
Manages concurrent video feeds (RTSP IP cameras, USB webcams, synthetic streams)
using low-latency threaded frame grabbers with auto-reconnection.

Features:
- Dedicated capture thread per camera stream
- Zero-latency buffer: Drops stale frames and always provides the newest frame
- Auto-reconnect handling on network disconnects
- 4-Grid multi-camera multiplexer for surveillance dashboard
"""

import time
import threading
import cv2
import numpy as np
from typing import Dict, Optional, Tuple, List, Any


class CameraStream:
    """
    Asynchronous threaded frame reader for an individual camera stream or video source.
    """

    def __init__(self, camera_id: str, source_url: Any, name: Optional[str] = None):
        self.camera_id = camera_id
        self.source_url = int(source_url) if str(source_url).isdigit() else source_url
        self.name = name or f"Camera {camera_id}"
        
        self.cap = None
        self.latest_frame: Optional[np.ndarray] = None
        self.is_running = False
        self.is_connected = False
        self.thread = None
        self.lock = threading.Lock()
        self.fps = 0.0
        self.frame_count = 0

    def start(self):
        """Starts the background frame grabber thread."""
        if self.is_running:
            return
        self.is_running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()

    def stop(self):
        """Stops the capture thread and releases camera hardware."""
        self.is_running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        if self.cap:
            self.cap.release()
            self.cap = None
        self.is_connected = False

    def _capture_loop(self):
        """Background thread loop for continuous low-latency frame reading."""
        reconnect_delay = 2.0

        while self.is_running:
            if self.cap is None or not self.cap.isOpened():
                try:
                    self.cap = cv2.VideoCapture(self.source_url)
                    # Optimize buffer size for low latency RTSP
                    self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    if self.cap.isOpened():
                        self.is_connected = True
                    else:
                        self.is_connected = False
                        time.sleep(reconnect_delay)
                        continue
                except Exception:
                    self.is_connected = False
                    time.sleep(reconnect_delay)
                    continue

            ret, frame = self.cap.read()
            if not ret or frame is None:
                self.is_connected = False
                if self.cap:
                    self.cap.release()
                    self.cap = None
                time.sleep(reconnect_delay)
                continue

            self.is_connected = True
            with self.lock:
                self.latest_frame = frame
                self.frame_count += 1

            time.sleep(0.01)  # Yield CPU cycles

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Returns (success, latest_frame_bgr).
        Non-blocking: Always returns the freshest frame.
        """
        with self.lock:
            if self.latest_frame is not None:
                return True, self.latest_frame.copy()
            return False, None


class StreamHub:
    """
    Manages multiple camera streams for security surveillance.
    """

    def __init__(self):
        self.cameras: Dict[str, CameraStream] = {}

    def add_camera(self, camera_id: str, source_url: Any, name: Optional[str] = None) -> CameraStream:
        """Adds and starts a new camera feed."""
        if camera_id in self.cameras:
            self.cameras[camera_id].stop()
            
        cam = CameraStream(camera_id=camera_id, source_url=source_url, name=name)
        self.cameras[camera_id] = cam
        cam.start()
        return cam

    def remove_camera(self, camera_id: str):
        """Stops and removes a camera feed."""
        if camera_id in self.cameras:
            self.cameras[camera_id].stop()
            del self.cameras[camera_id]

    def stop_all(self):
        """Stops all running camera feeds."""
        for cam in self.cameras.values():
            cam.stop()
        self.cameras.clear()

    def get_active_cameras(self) -> List[CameraStream]:
        return list(self.cameras.values())
