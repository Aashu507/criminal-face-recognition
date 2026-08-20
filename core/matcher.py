"""
Face Matcher
============
High-level matching engine that orchestrates the full pipeline:
Image → Preprocess → Detect → Embed → Search Database → Return Matches

Supports both:
- 1:1 Verification: "Are these two faces the same person?"
- 1:N Identification: "Who is this person in our criminal database?"
"""

import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import cv2
import numpy as np
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

from core.preprocessor import ImagePreprocessor
from core.detector import FaceDetector
from core.recognizer import FaceRecognizer
from core.database import FaceDatabase
from core.pose_aligner import PoseAligner
from core.adaface_recognizer import AdaFaceRecognizer

console = Console()

# Load environment variables
load_dotenv()


class FaceMatcher:
    """
    End-to-end face matching pipeline for criminal identification.

    Typical workflow:
        matcher = FaceMatcher()

        # Enroll criminals
        matcher.enroll_from_image("CRIM001", "mugshot.jpg", {"name": "Suspect A"})

        # Search for a face
        matches = matcher.search_from_image("query.jpg")
        matcher.print_matches(matches)
    """

    def __init__(
        self,
        model_name: str = None,
        gpu_id: int = None,
        det_size: int = None,
        similarity_threshold: float = None,
        top_k: int = None,
        db_dir: str = None,
    ):
        """
        Initialize the full pipeline with sensible defaults from .env.

        All parameters are optional — defaults are loaded from environment
        variables, with fallbacks to sensible values.
        """
        self.model_name = model_name or os.getenv("MODEL_NAME", "buffalo_l")
        self.gpu_id = gpu_id if gpu_id is not None else int(os.getenv("GPU_ID", "0"))
        self.det_size = det_size or int(os.getenv("DET_SIZE", "640"))
        self.similarity_threshold = (
            similarity_threshold
            if similarity_threshold is not None
            else float(os.getenv("SIMILARITY_THRESHOLD", "0.45"))
        )
        self.top_k = top_k or int(os.getenv("TOP_K", "5"))
        self.db_dir = db_dir or os.getenv("CHROMA_DB_DIR", "./data/chromadb")

        # Initialize components (lazy — actual model loading happens on first use)
        self.preprocessor = ImagePreprocessor()
        self.detector = FaceDetector(
            model_name=self.model_name,
            gpu_id=self.gpu_id,
            det_size=self.det_size,
        )
        self.recognizer = FaceRecognizer(detector=self.detector)
        self.database = FaceDatabase(persist_dir=self.db_dir)
        self.pose_aligner = PoseAligner()
        self.adaface = AdaFaceRecognizer(base_recognizer=self.recognizer, pose_aligner=self.pose_aligner)

    def enroll_from_image(
        self,
        criminal_id: str,
        image_path: str,
        metadata: Optional[Dict[str, Any]] = None,
        enhance: bool = True,
    ) -> Dict[str, Any]:
        """
        Enroll a criminal from an image file.

        Args:
            criminal_id: Unique identifier for the criminal.
            image_path: Path to the mugshot/photo.
            metadata: Optional dict (name, case_number, notes, etc.).
            enhance: Whether to apply CLAHE preprocessing.

        Returns:
            Result dict with status, quality metrics, and face data.
        """
        start = time.time()

        # Load and preprocess
        img = self.preprocessor.load_image(image_path)
        if img is None:
            return {"success": False, "error": f"Failed to load image: {image_path}"}

        quality = self.preprocessor.assess_quality(img)

        if enhance:
            img = self.preprocessor.enhance(img)

        # Detect and extract embedding
        face = self.recognizer.get_face_with_embedding(img)
        if face is None:
            return {
                "success": False,
                "error": "No face detected in image",
                "quality": quality,
            }

        if face.embedding is None:
            return {
                "success": False,
                "error": "Face detected but embedding extraction failed",
                "quality": quality,
            }

        # Enroll in database
        enroll_meta = metadata or {}
        enroll_meta["image_path"] = str(image_path)
        if face.age is not None:
            enroll_meta["estimated_age"] = face.age
        if face.gender is not None:
            enroll_meta["estimated_gender"] = face.gender

        self.database.enroll(
            criminal_id=criminal_id,
            embedding=face.embedding,
            metadata=enroll_meta,
            face_image=face.face_image,
        )

        elapsed = time.time() - start

        return {
            "success": True,
            "criminal_id": criminal_id,
            "confidence": round(face.confidence, 4),
            "age": face.age,
            "gender": face.gender,
            "quality": quality,
            "elapsed_ms": round(elapsed * 1000, 1),
        }

    def enroll_from_directory(
        self,
        directory: str,
        id_from_filename: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Batch enroll all face images from a directory.

        Args:
            directory: Path to directory containing face images.
            id_from_filename: If True, use filename (without extension) as criminal ID.
            metadata: Optional metadata applied to all enrollments.

        Returns:
            Summary dict with success/failure counts.
        """
        dir_path = Path(directory)
        if not dir_path.is_dir():
            return {"success": False, "error": f"Not a directory: {directory}"}

        image_files = [
            f for f in dir_path.iterdir()
            if f.suffix.lower() in ImagePreprocessor.SUPPORTED_FORMATS
        ]

        if not image_files:
            return {"success": False, "error": "No image files found in directory"}

        console.print(
            f"\n[cyan][*] Enrolling {len(image_files)} images from[/cyan] {directory}\n"
        )

        results = {"total": len(image_files), "enrolled": 0, "failed": 0, "details": []}

        for i, img_path in enumerate(sorted(image_files), 1):
            criminal_id = img_path.stem if id_from_filename else f"CRIM{i:04d}"

            console.print(
                f"  [{i}/{len(image_files)}] {img_path.name} -> {criminal_id}",
                end=" ",
            )

            result = self.enroll_from_image(
                criminal_id=criminal_id,
                image_path=str(img_path),
                metadata=metadata,
            )

            if result["success"]:
                results["enrolled"] += 1
                console.print(
                    f"[green][+][/green] "
                    f"(conf={result['confidence']}, "
                    f"quality={result['quality']['quality_score']})"
                )
            else:
                results["failed"] += 1
                console.print(f"[red][-] {result['error']}[/red]")

            results["details"].append(result)

        console.print(
            f"\n[bold]Enrollment complete:[/bold] "
            f"[green]{results['enrolled']} enrolled[/green], "
            f"[red]{results['failed']} failed[/red] "
            f"(out of {results['total']})\n"
        )

        return results

    def search_from_image(
        self,
        image_path: str,
        top_k: Optional[int] = None,
        threshold: Optional[float] = None,
        enhance: bool = True,
    ) -> Dict[str, Any]:
        """
        Search the criminal database for a face in an image.

        Args:
            image_path: Path to the query image.
            top_k: Number of top matches to return.
            threshold: Minimum similarity threshold.
            enhance: Whether to apply CLAHE preprocessing.

        Returns:
            Result dict with matches, quality info, and timing.
        """
        start = time.time()
        top_k = top_k or self.top_k
        threshold = threshold if threshold is not None else self.similarity_threshold

        # Load and preprocess
        img = self.preprocessor.load_image(image_path)
        if img is None:
            return {"success": False, "error": f"Failed to load image: {image_path}"}

        quality = self.preprocessor.assess_quality(img)

        if enhance:
            img = self.preprocessor.enhance(img)

        # Detect and extract embedding
        face = self.recognizer.get_face_with_embedding(img)
        if face is None:
            return {
                "success": False,
                "error": "No face detected in query image",
                "quality": quality,
            }

        if face.embedding is None:
            return {
                "success": False,
                "error": "Face detected but embedding extraction failed",
                "quality": quality,
            }

        # Search database
        matches = self.database.search(
            query_embedding=face.embedding,
            top_k=top_k,
            threshold=threshold,
        )

        elapsed = time.time() - start

        return {
            "success": True,
            "matches": matches,
            "match_count": len(matches),
            "query_face": {
                "confidence": round(face.confidence, 4),
                "age": face.age,
                "gender": face.gender,
            },
            "quality": quality,
            "threshold": threshold,
            "elapsed_ms": round(elapsed * 1000, 1),
        }

    def search_from_frame(
        self,
        frame: np.ndarray,
        top_k: Optional[int] = None,
        threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Search from a live camera/CCTV frame (numpy array).

        Args:
            frame: BGR numpy array from camera/CCTV.
            top_k: Number of top matches to return.
            threshold: Minimum similarity threshold.

        Returns:
            Same result format as search_from_image.
        """
        start = time.time()
        top_k = top_k or self.top_k
        threshold = threshold if threshold is not None else self.similarity_threshold

        quality = self.preprocessor.assess_quality(frame)
        enhanced = self.preprocessor.enhance(frame)

        face = self.recognizer.get_face_with_embedding(enhanced)
        if face is None:
            return {"success": False, "error": "No face detected in frame"}

        if face.embedding is None:
            return {"success": False, "error": "Embedding extraction failed"}

        matches = self.database.search(
            query_embedding=face.embedding,
            top_k=top_k,
            threshold=threshold,
        )

        elapsed = time.time() - start

        return {
            "success": True,
            "matches": matches,
            "match_count": len(matches),
            "query_face": {
                "confidence": round(face.confidence, 4),
                "age": face.age,
                "gender": face.gender,
            },
            "quality": quality,
            "elapsed_ms": round(elapsed * 1000, 1),
        }

    def print_matches(self, result: Dict[str, Any]):
        """
        Pretty-print search results using Rich tables.

        Args:
            result: Result dict from search_from_image() or search_from_frame().
        """
        if not result.get("success"):
            console.print(f"[red][-] Search failed:[/red] {result.get('error')}")
            return

        console.print(
            f"\n[bold]Search completed in {result['elapsed_ms']}ms[/bold]"
        )

        # Query face info
        qf = result.get("query_face", {})
        console.print(
            f"  Query face: confidence={qf.get('confidence')}, "
            f"age≈{qf.get('age')}, gender={qf.get('gender')}"
        )

        # Quality info
        quality = result.get("quality", {})
        if quality.get("warnings"):
            for w in quality["warnings"]:
                console.print(f"  [yellow]⚠ {w}[/yellow]")

        # Matches table
        matches = result.get("matches", [])
        if not matches:
            console.print(
                f"\n  [yellow]No matches found above threshold "
                f"{result.get('threshold', 'N/A')}[/yellow]\n"
            )
            return

        table = Table(title=f"Top {len(matches)} Matches", show_lines=True)
        table.add_column("Rank", style="bold cyan", width=5)
        table.add_column("Criminal ID", style="bold")
        table.add_column("Name", style="white")
        table.add_column("Similarity", style="green")
        table.add_column("Confidence", style="yellow")
        table.add_column("Details", style="dim")

        for i, match in enumerate(matches, 1):
            meta = match.get("metadata", {})
            sim = match["similarity"]

            # Color code similarity
            if sim >= 0.55:
                sim_str = f"[bold green]{sim:.4f}[/bold green] ★"
            elif sim >= 0.45:
                sim_str = f"[green]{sim:.4f}[/green]"
            elif sim >= 0.35:
                sim_str = f"[yellow]{sim:.4f}[/yellow]"
            else:
                sim_str = f"[red]{sim:.4f}[/red]"

            # Confidence indicator
            if sim >= 0.55:
                conf = "HIGH"
            elif sim >= 0.45:
                conf = "MEDIUM"
            else:
                conf = "LOW"

            details = []
            if meta.get("case_number"):
                details.append(f"Case: {meta['case_number']}")
            if meta.get("estimated_age"):
                details.append(f"Age≈{meta['estimated_age']}")
            if meta.get("estimated_gender"):
                details.append(f"Gender: {meta['estimated_gender']}")

            table.add_row(
                str(i),
                match["criminal_id"],
                meta.get("name", "Unknown"),
                sim_str,
                conf,
                ", ".join(details) if details else "—",
            )

        console.print(table)
        console.print()

    @property
    def db(self):
        return self.database

    @property
    def threshold(self) -> float:
        return self.similarity_threshold

    @threshold.setter
    def threshold(self, val: float):
        self.similarity_threshold = val

    def search_image(
        self,
        image_bgr: np.ndarray,
        top_k: int = 3,
        use_adaface: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Detects all faces in a BGR image, estimates pose, calculates AdaFace quality,
        and queries ChromaDB for suspect matches.

        Returns list of dicts:
        [
            {
                "face": DetectedFace,
                "matches": [{"id": ..., "name": ..., "similarity": ...}],
                "pose": {"yaw": ..., "pitch": ..., "roll": ...},
                "quality_score": 0.85
            },
            ...
        ]
        """
        faces = self.detector.detect(image_bgr)
        results = []

        for face in faces:
            # 5-Point pose angle estimation
            pose = self.pose_aligner.estimate_pose_angles(face.landmarks)

            # AdaFace Quality calculation & adaptive embedding
            emb, q_score, telemetry = self.adaface.extract_adaptive_embedding(
                image_bgr,
                landmarks=face.landmarks,
                detected_face=face
            )

            # Query database
            raw_matches = self.database.search(
                query_embedding=emb,
                top_k=top_k,
                threshold=0.0  # Return raw candidates to apply adaptive similarity
            )

            refined_matches = []
            for m in raw_matches:
                meta = m.get("metadata", {})
                raw_sim = float(m.get("similarity", 0.0))
                
                # AdaFace quality compensation: Ensure quality score Q does not artificially dampen valid match
                if use_adaface and q_score < 0.60:
                    adapted_sim = raw_sim * (1.0 + 0.05 * (1.0 - q_score))
                else:
                    adapted_sim = raw_sim

                refined_matches.append({
                    "id": m.get("criminal_id") or m.get("id"),
                    "name": meta.get("name", "Unknown"),
                    "similarity": round(float(np.clip(adapted_sim, -1.0, 1.0)), 4),
                    "metadata": meta
                })

            # Sort by similarity descending
            refined_matches.sort(key=lambda x: x["similarity"], reverse=True)

            results.append({
                "face": face,
                "matches": refined_matches[:top_k],
                "pose": pose,
                "quality_score": round(q_score, 3)
            })

        return results

    def enroll_image(
        self,
        image_bgr: np.ndarray,
        criminal_id: str,
        name: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, str]:
        """
        Enrolls a criminal into ChromaDB directly from a BGR numpy image.
        """
        faces = self.detector.detect(image_bgr)
        if not faces:
            return False, "No face detected in the provided mugshot photo."

        face = faces[0]  # Primary face
        emb, q_score, _ = self.adaface.extract_adaptive_embedding(
            image_bgr,
            landmarks=face.landmarks,
            detected_face=face
        )

        # Generate high-definition studio thumbnail (256x256) with 15% margin
        box = [int(v) for v in face.bbox]
        h_img, w_img = image_bgr.shape[:2]
        bw = box[2] - box[0]
        bh = box[3] - box[1]
        pad_x = int(bw * 0.15)
        pad_y = int(bh * 0.15)
        x1 = max(0, box[0] - pad_x)
        y1 = max(0, box[1] - pad_y)
        x2 = min(w_img, box[2] + pad_x)
        y2 = min(h_img, box[3] + pad_y)
        face_crop = image_bgr[y1:y2, x1:x2]
        
        thumbnail_b64 = None
        if face_crop.size > 0:
            thumb_resized = cv2.resize(face_crop, (256, 256), interpolation=cv2.INTER_AREA)
            _, buf = cv2.imencode(".jpg", thumb_resized, [cv2.IMWRITE_JPEG_QUALITY, 95])
            import base64
            thumbnail_b64 = base64.b64encode(buf).decode("utf-8")

        enroll_meta = metadata or {}
        enroll_meta["name"] = name
        enroll_meta["quality_score"] = float(q_score)

        success = self.database.enroll(
            criminal_id=criminal_id,
            embedding=emb,
            metadata=enroll_meta,
            thumbnail=thumbnail_b64
        )

        if success:
            return True, f"Successfully enrolled {name} (ID: {criminal_id}) with Quality Score: {q_score*100:.1f}%"
        else:
            return False, f"Failed to save record to ChromaDB for ID: {criminal_id}"

