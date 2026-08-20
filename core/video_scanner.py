"""
CCTV Video Forensics Scanner
============================
Processes video recordings (.mp4, .avi, .mkv, .mov) or live RTSP streams to identify
enrolled criminals across video frames with timestamp indexing and timeline generation.

Features:
- Configurable frame-skip rate (e.g. process every Nth frame for high FPS throughput)
- Bounding box annotation and suspect tracking
- Video match timeline with timestamps, match confidence, and face crops
- Memory-safe frame generator
"""

import time
import os
import cv2
import numpy as np
from typing import List, Dict, Any, Generator, Tuple, Optional
from dataclasses import dataclass, field


@dataclass
class VideoDetectionEvent:
    frame_number: int
    timestamp_seconds: float
    formatted_time: str
    criminal_id: str
    criminal_name: str
    similarity: float
    bbox: List[int]
    face_crop_bgr: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)


class CCTVVideoScanner:
    """
    Scans video files or RTSP streams for suspects with high throughput.
    """

    def __init__(self, face_matcher, frame_step: int = 5, min_confidence: float = 0.45):
        """
        Args:
            face_matcher: Instance of FaceMatcher.
            frame_step: Process every N-th frame (e.g. 5 = process 6 frames/sec from 30fps video).
            min_confidence: Minimum similarity score to register a match.
        """
        self.matcher = face_matcher
        self.frame_step = max(1, frame_step)
        self.min_confidence = min_confidence

    @staticmethod
    def format_timestamp(seconds: float) -> str:
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        millis = int((seconds - int(seconds)) * 1000)
        return f"{mins:02d}:{secs:02d}.{millis:03d}"

    def scan_video_stream(
        self,
        video_source: Any,
        max_frames: Optional[int] = None
    ) -> Generator[Tuple[int, float, np.ndarray, List[VideoDetectionEvent]], None, None]:
        """
        Generator that yields annotated frames and detected events as the video is processed.
        """
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            raise ValueError(f"Could not open video stream/file: {video_source}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frame_idx = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if max_frames and frame_idx >= max_frames:
                    break

                frame_idx += 1
                if frame_idx % self.frame_step != 0:
                    continue

                timestamp_sec = frame_idx / fps
                formatted_time = self.format_timestamp(timestamp_sec)

                # Search faces in current frame
                results = self.matcher.search_image(frame, top_k=1)
                events = []
                annotated_frame = frame.copy()

                for res in results:
                    face = res["face"]
                    box = [int(v) for v in face.bbox]
                    matches = res["matches"]

                    if matches and matches[0]["similarity"] >= self.min_confidence:
                        top = matches[0]
                        x1, y1, x2, y2 = max(0, box[0]), max(0, box[1]), min(frame.shape[1], box[2]), min(frame.shape[0], box[3])
                        crop = frame[y1:y2, x1:x2].copy()

                        event = VideoDetectionEvent(
                            frame_number=frame_idx,
                            timestamp_seconds=timestamp_sec,
                            formatted_time=formatted_time,
                            criminal_id=top["id"],
                            criminal_name=top["name"],
                            similarity=top["similarity"],
                            bbox=box,
                            face_crop_bgr=crop,
                            metadata=top.get("metadata", {})
                        )
                        events.append(event)

                        # Draw red alert bounding box
                        cv2.rectangle(annotated_frame, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
                        cv2.putText(
                            annotated_frame,
                            f"SUSPECT: {top['name']} ({top['similarity']:.2f})",
                            (box[0], max(20, box[1] - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 0, 255),
                            2
                        )
                    else:
                        # Unknown person (green box)
                        cv2.rectangle(annotated_frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)

                yield frame_idx, timestamp_sec, annotated_frame, events

        finally:
            cap.release()
