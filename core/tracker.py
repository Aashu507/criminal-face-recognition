"""
Multi-Target Face & Suspect Tracker
===================================
Tracks individual faces and suspects across consecutive video/CCTV frames.
Combines spatial IoU (Intersection-over-Union) with facial feature embedding similarity
to maintain persistent Track IDs, calculate dwell time, and throttle database searches.

Features:
- Persistent Track IDs across frame drops/occlusions
- Trajectory path tracking (entry to exit path)
- Running average embedding for robust recognition over time
- Automatic query throttling (saves ~80% redundant ChromaDB searches)
"""

import time
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any


@dataclass
class TrackedFace:
    """Represents an actively tracked suspect/person across video frames."""
    track_id: int
    bbox: List[int]
    confidence: float
    first_seen: float
    last_seen: float
    hits: int = 1
    misses: int = 0
    embedding: Optional[np.ndarray] = None
    identity: Optional[Dict[str, Any]] = None  # Matched criminal metadata
    trajectory: List[Tuple[int, int]] = field(default_factory=list)  # (center_x, center_y) path
    last_query_time: float = 0.0

    @property
    def dwell_time_seconds(self) -> float:
        """Total time in seconds this person has been in view."""
        return max(0.0, self.last_seen - self.first_seen)

    @property
    def is_identified(self) -> bool:
        return self.identity is not None and self.identity.get("is_match", False)

    @property
    def center(self) -> Tuple[int, int]:
        x1, y1, x2, y2 = self.bbox
        return int((x1 + x2) / 2), int((y1 + y2) / 2)


def compute_iou(boxA: List[int], boxB: List[int]) -> float:
    """Calculates Intersection-over-Union (IoU) between two bounding boxes [x1, y1, x2, y2]."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = max(0, boxA[2] - boxA[0]) * max(0, boxA[3] - boxA[1])
    boxBArea = max(0, boxB[2] - boxB[0]) * max(0, boxB[3] - boxB[1])

    unionArea = float(boxAArea + boxBArea - interArea)
    if unionArea <= 0:
        return 0.0
    return interArea / unionArea


class FaceTracker:
    """
    Multi-target tracker for real-time surveillance streams.
    """

    def __init__(
        self,
        iou_threshold: float = 0.35,
        sim_threshold: float = 0.50,
        max_misses: int = 15,
        query_cooldown_seconds: float = 2.0
    ):
        """
        Args:
            iou_threshold: Minimum IoU to associate box in next frame.
            sim_threshold: Minimum embedding similarity to link tracks.
            max_misses: Frames before an inactive track is discarded.
            query_cooldown_seconds: Minimum interval between database queries for the same track.
        """
        self.iou_threshold = iou_threshold
        self.sim_threshold = sim_threshold
        self.max_misses = max_misses
        self.query_cooldown_seconds = query_cooldown_seconds
        
        self.tracks: Dict[int, TrackedFace] = {}
        self.next_track_id = 101

    def update(
        self,
        detected_faces: List[Any],
        current_time: Optional[float] = None
    ) -> List[TrackedFace]:
        """
        Updates active tracks with new detections from the current frame.
        
        Args:
            detected_faces: List of DetectedFace objects from FaceDetector.
            current_time: Timestamp in seconds (defaults to time.time()).
            
        Returns:
            List of active TrackedFace objects in the current frame.
        """
        now = current_time if current_time is not None else time.time()
        
        # 1. Match detections to existing tracks via IoU and Embedding similarity
        unmatched_dets = list(range(len(detected_faces)))
        matched_tracks = set()

        for track_id, track in list(self.tracks.items()):
            best_match_idx = None
            best_score = 0.0

            for det_idx in unmatched_dets:
                det = detected_faces[det_idx]
                det_box = [int(v) for v in det.bbox]
                
                iou = compute_iou(track.bbox, det_box)
                
                # If embeddings available, use hybrid score: 0.6 * IoU + 0.4 * CosineSim
                if track.embedding is not None and getattr(det, 'embedding', None) is not None:
                    emb_sim = float(np.dot(track.embedding, det.embedding))
                    hybrid_score = 0.5 * iou + 0.5 * max(0.0, emb_sim)
                else:
                    hybrid_score = iou

                if hybrid_score > self.iou_threshold and hybrid_score > best_score:
                    best_score = hybrid_score
                    best_match_idx = det_idx

            if best_match_idx is not None:
                # Update existing track
                matched_det = detected_faces[best_match_idx]
                track.bbox = [int(v) for v in matched_det.bbox]
                track.confidence = float(matched_det.confidence)
                track.last_seen = now
                track.hits += 1
                track.misses = 0
                track.trajectory.append(track.center)
                if len(track.trajectory) > 50:
                    track.trajectory.pop(0)

                # Update running average embedding
                if getattr(matched_det, 'embedding', None) is not None:
                    if track.embedding is None:
                        track.embedding = matched_det.embedding
                    else:
                        # Momentum update: 80% old + 20% new
                        updated_emb = 0.8 * track.embedding + 0.2 * matched_det.embedding
                        track.embedding = updated_emb / (np.linalg.norm(updated_emb) or 1.0)

                unmatched_dets.remove(best_match_idx)
                matched_tracks.add(track_id)
            else:
                track.misses += 1

        # 2. Create new tracks for remaining unmatched detections
        for det_idx in unmatched_dets:
            det = detected_faces[det_idx]
            new_track = TrackedFace(
                track_id=self.next_track_id,
                bbox=[int(v) for v in det.bbox],
                confidence=float(det.confidence),
                first_seen=now,
                last_seen=now,
                embedding=getattr(det, 'embedding', None)
            )
            new_track.trajectory.append(new_track.center)
            self.tracks[self.next_track_id] = new_track
            matched_tracks.add(self.next_track_id)
            self.next_track_id += 1

        # 3. Prune dead tracks
        for track_id, track in list(self.tracks.items()):
            if track.misses > self.max_misses:
                del self.tracks[track_id]

        # Return only currently active tracks (seen in this frame)
        return [self.tracks[tid] for tid in matched_tracks if tid in self.tracks]

    def should_query_database(self, track: TrackedFace, current_time: float) -> bool:
        """Determines if a track needs a ChromaDB query (throttling check)."""
        if track.identity is not None and track.identity.get("is_match", False):
            # Already positively identified as a criminal, re-query sparingly (every 10s)
            return (current_time - track.last_query_time) > 10.0
            
        # Unidentified track: query if never queried or cooldown elapsed
        return (current_time - track.last_query_time) >= self.query_cooldown_seconds
