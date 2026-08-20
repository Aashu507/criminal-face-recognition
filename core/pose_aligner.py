"""
5-Point Affine Face Alignment & Head Pose Estimator
===================================================
Normalizes tilted, rotated, and overhead CCTV face crops into canonical 112x112 frontal
alignments before neural embedding extraction.

Features:
- Standard 5-point similarity transformation (ArcFace / InsightFace canonical coordinate frame)
- Head Pose Estimation (Yaw, Pitch, Roll angles in degrees)
- Extreme angle detection (flags severe CCTV angles > 45°)
"""

import cv2
import numpy as np
from typing import Tuple, Dict, Optional, List


# Canonical 5-point reference coordinates for 112x112 aligned face crop
CANONICAL_5PTS = np.array([
    [38.2946, 51.6963],  # Left Eye
    [73.5318, 51.5014],  # Right Eye
    [56.0252, 71.7366],  # Nose Tip
    [41.5493, 92.3655],  # Left Mouth Corner
    [70.7299, 92.2041]   # Right Mouth Corner
], dtype=np.float32)


class PoseAligner:
    """
    Normalizes facial crops using 5-point similarity transformation and estimates head pose.
    """

    def __init__(self, target_size: Tuple[int, int] = (112, 112)):
        self.target_size = target_size

    def align_face_5pts(self, image: np.ndarray, landmarks_5pts: np.ndarray) -> np.ndarray:
        """
        Aligns a face image using 5-point landmarks to the canonical 112x112 reference frame.
        
        Args:
            image: Full BGR image (or crop containing face).
            landmarks_5pts: Array of shape (5, 2) with (x, y) coordinates:
                            [left_eye, right_eye, nose, left_mouth, right_mouth]
                            
        Returns:
            Warped, aligned 112x112 BGR face crop.
        """
        src_pts = np.asarray(landmarks_5pts, dtype=np.float32)
        if src_pts.shape != (5, 2):
            raise ValueError(f"Expected landmarks shape (5, 2), got {src_pts.shape}")

        # Compute partial affine (similarity) transformation matrix (scale, rotation, translation)
        matrix, inliers = cv2.estimateAffinePartial2D(src_pts, CANONICAL_5PTS, method=cv2.LMEDS)

        if matrix is None:
            # Fallback to standard affine if similarity transform fails
            matrix = cv2.getAffineTransform(src_pts[:3], CANONICAL_5PTS[:3])

        # Warp image into canonical bounding space with reflection padding
        aligned = cv2.warpAffine(
            image,
            matrix,
            self.target_size,
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REFLECT_101
        )
        return aligned

    def estimate_pose_angles(self, landmarks_5pts: np.ndarray) -> Dict[str, float]:
        """
        Estimates approximate head pose angles (Yaw, Pitch, Roll in degrees) from 5 landmarks.
        
        Yaw: Left (-) / Right (+) head turn
        Pitch: Down (-) / Up (+) head tilt
        Roll: Clockwise / Counter-clockwise tilt
        """
        pts = np.asarray(landmarks_5pts, dtype=np.float32)
        if pts.shape != (5, 2):
            return {"yaw": 0.0, "pitch": 0.0, "roll": 0.0, "is_extreme": False}

        left_eye, right_eye, nose, left_mouth, right_mouth = pts

        # 1. Roll: Angle of the eye line relative to horizontal
        eye_dx = right_eye[0] - left_eye[0]
        eye_dy = right_eye[1] - left_eye[1]
        roll = np.degrees(np.arctan2(eye_dy, eye_dx))

        # 2. Yaw: Ratio of nose distance to left eye vs right eye
        eye_center = (left_eye + right_eye) / 2.0
        eye_distance = np.linalg.norm(right_eye - left_eye) or 1e-6
        
        dist_left_eye_nose = np.linalg.norm(nose - left_eye)
        dist_right_eye_nose = np.linalg.norm(nose - right_eye)
        
        yaw_ratio = (dist_right_eye_nose - dist_left_eye_nose) / eye_distance
        yaw = float(np.clip(yaw_ratio * 90.0, -90.0, 90.0))

        # 3. Pitch: Vertical position of nose relative to eye-mouth distance
        mouth_center = (left_mouth + right_mouth) / 2.0
        face_height = np.linalg.norm(mouth_center - eye_center) or 1e-6
        
        nose_rel_y = (nose[1] - eye_center[1]) / face_height
        # Canonical nose relative y is ~0.45
        pitch = float(np.clip((nose_rel_y - 0.45) * 120.0, -90.0, 90.0))

        is_extreme = abs(yaw) > 40.0 or abs(pitch) > 35.0 or abs(roll) > 30.0

        return {
            "yaw": round(yaw, 1),
            "pitch": round(pitch, 1),
            "roll": round(roll, 1),
            "is_extreme": is_extreme
        }
