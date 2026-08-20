#!/usr/bin/env python3
"""
Enroll Criminal Faces
=====================
Batch enroll face images into the criminal recognition database.

Usage:
    # Enroll all images from a directory (filename = criminal ID):
    python scripts/enroll_faces.py --dir data/criminals/

    # Enroll a single image:
    python scripts/enroll_faces.py --image mugshot.jpg --id CRIM001 --name "Suspect A"

    # Enroll with metadata:
    python scripts/enroll_faces.py --image photo.jpg --id CRIM002 --name "Suspect B" --case "2024-XYZ"
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rich.console import Console
from core.matcher import FaceMatcher

console = Console()


def main():
    parser = argparse.ArgumentParser(
        description="Enroll criminal faces into the recognition database",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/enroll_faces.py --dir data/criminals/
  python scripts/enroll_faces.py --image mugshot.jpg --id CRIM001 --name "Suspect A"
  python scripts/enroll_faces.py --image photo.jpg --id CRIM002 --name "Suspect B" --case "CASE-2024-001"
        """,
    )

    # Mode selection
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--dir", type=str,
        help="Directory of face images to batch enroll (filename = criminal ID)"
    )
    group.add_argument(
        "--image", type=str,
        help="Single image file to enroll"
    )

    # Single-image options
    parser.add_argument("--id", type=str, help="Criminal ID (required with --image)")
    parser.add_argument("--name", type=str, help="Criminal's name")
    parser.add_argument("--case", type=str, help="Case number")
    parser.add_argument("--notes", type=str, help="Additional notes")

    # Pipeline options
    parser.add_argument(
        "--no-enhance", action="store_true",
        help="Skip CLAHE image enhancement"
    )
    parser.add_argument(
        "--threshold", type=float, default=None,
        help="Similarity threshold (default: from .env or 0.45)"
    )
    parser.add_argument(
        "--db-dir", type=str, default=None,
        help="ChromaDB directory (default: from .env or ./chroma_db)"
    )
    parser.add_argument(
        "--gpu-id", type=int, default=None,
        help="GPU device ID (default: 0, use -1 for CPU)"
    )

    args = parser.parse_args()

    # Validate single-image mode requires --id
    if args.image and not args.id:
        parser.error("--id is required when using --image")

    # Initialize pipeline
    console.print("\n[bold cyan]═══ Criminal Face Enrollment ═══[/bold cyan]\n")

    matcher_kwargs = {}
    if args.threshold is not None:
        matcher_kwargs["similarity_threshold"] = args.threshold
    if args.db_dir is not None:
        matcher_kwargs["db_dir"] = args.db_dir
    if args.gpu_id is not None:
        matcher_kwargs["gpu_id"] = args.gpu_id

    matcher = FaceMatcher(**matcher_kwargs)

    if args.dir:
        # Batch enrollment from directory
        result = matcher.enroll_from_directory(
            directory=args.dir,
            id_from_filename=True,
        )
        if not result.get("success", True):
            console.print(f"[red]✗ {result.get('error')}[/red]")
            sys.exit(1)
    else:
        # Single image enrollment
        metadata = {}
        if args.name:
            metadata["name"] = args.name
        if args.case:
            metadata["case_number"] = args.case
        if args.notes:
            metadata["notes"] = args.notes

        result = matcher.enroll_from_image(
            criminal_id=args.id,
            image_path=args.image,
            metadata=metadata if metadata else None,
            enhance=not args.no_enhance,
        )

        if result["success"]:
            console.print(f"[green]✓ Enrolled:[/green] {args.id}")
            console.print(f"  Confidence: {result['confidence']}")
            console.print(f"  Age estimate: {result.get('age', 'N/A')}")
            console.print(f"  Gender estimate: {result.get('gender', 'N/A')}")
            console.print(f"  Image quality: {result['quality']['quality_score']}/100")
            console.print(f"  Time: {result['elapsed_ms']}ms")
        else:
            console.print(f"[red]✗ Enrollment failed:[/red] {result['error']}")
            sys.exit(1)

    # Print database stats
    console.print(f"\n[dim]Database now has {matcher.database.count()} enrolled face(s)[/dim]\n")


if __name__ == "__main__":
    main()
