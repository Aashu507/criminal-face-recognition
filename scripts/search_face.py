#!/usr/bin/env python3
"""
Search for a Face in the Criminal Database
===========================================
Query the criminal recognition database with an image or live camera feed.

Usage:
    # Search with an image file:
    python scripts/search_face.py --image query_photo.jpg

    # Search with live webcam:
    python scripts/search_face.py --webcam

    # Search with custom threshold:
    python scripts/search_face.py --image suspect.jpg --threshold 0.40 --top-k 10

    # Export results to JSON:
    python scripts/search_face.py --image suspect.jpg --export results.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
from rich.console import Console
from core.matcher import FaceMatcher

console = Console()


def search_image(matcher: FaceMatcher, args):
    """Search using a static image file."""
    console.print(f"\n[cyan]Searching for face in:[/cyan] {args.image}\n")

    result = matcher.search_from_image(
        image_path=args.image,
        top_k=args.top_k,
        threshold=args.threshold,
        enhance=not args.no_enhance,
    )

    matcher.print_matches(result)

    # Export results if requested
    if args.export and result.get("success"):
        export_data = {
            "query_image": args.image,
            "matches": result["matches"],
            "match_count": result["match_count"],
            "threshold": result["threshold"],
            "elapsed_ms": result["elapsed_ms"],
            "quality": result["quality"],
        }
        with open(args.export, "w") as f:
            json.dump(export_data, f, indent=2, default=str)
        console.print(f"[green]Results exported to:[/green] {args.export}")

    return result


def search_webcam(matcher: FaceMatcher, args):
    """Search using live webcam feed."""
    console.print("\n[cyan]Starting webcam search...[/cyan]")
    console.print("[dim]Press 'q' to quit, 's' to search current frame[/dim]\n")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        console.print("[red]✗ Could not open webcam[/red]")
        sys.exit(1)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                console.print("[red]✗ Failed to read from webcam[/red]")
                break

            # Display the frame
            cv2.imshow("Criminal Face Search (press 's' to search, 'q' to quit)", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            elif key == ord("s"):
                console.print("\n[cyan]⟳ Searching...[/cyan]")
                result = matcher.search_from_frame(
                    frame=frame,
                    top_k=args.top_k,
                    threshold=args.threshold,
                )
                matcher.print_matches(result)

    finally:
        cap.release()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        description="Search the criminal database for matching faces",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/search_face.py --image suspect.jpg
  python scripts/search_face.py --image photo.png --threshold 0.40 --top-k 10
  python scripts/search_face.py --webcam
  python scripts/search_face.py --image suspect.jpg --export matches.json
        """,
    )

    # Input mode
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", type=str, help="Image file to search")
    group.add_argument("--webcam", action="store_true", help="Use live webcam feed")

    # Search options
    parser.add_argument(
        "--threshold", type=float, default=None,
        help="Minimum similarity threshold (default: from .env or 0.45)"
    )
    parser.add_argument(
        "--top-k", type=int, default=5,
        help="Number of top matches to return (default: 5)"
    )
    parser.add_argument(
        "--no-enhance", action="store_true",
        help="Skip CLAHE image enhancement"
    )
    parser.add_argument(
        "--export", type=str, default=None,
        help="Export results to JSON file"
    )

    # Pipeline options
    parser.add_argument(
        "--db-dir", type=str, default=None,
        help="ChromaDB directory (default: from .env or ./chroma_db)"
    )
    parser.add_argument(
        "--gpu-id", type=int, default=None,
        help="GPU device ID (default: 0, use -1 for CPU)"
    )

    args = parser.parse_args()

    # Initialize pipeline
    console.print("\n[bold cyan]═══ Criminal Face Search ═══[/bold cyan]")

    matcher_kwargs = {}
    if args.threshold is not None:
        matcher_kwargs["similarity_threshold"] = args.threshold
    if args.db_dir is not None:
        matcher_kwargs["db_dir"] = args.db_dir
    if args.gpu_id is not None:
        matcher_kwargs["gpu_id"] = args.gpu_id

    matcher = FaceMatcher(**matcher_kwargs)

    console.print(
        f"[dim]Database: {matcher.database.count()} enrolled face(s)[/dim]"
    )

    if args.image:
        search_image(matcher, args)
    elif args.webcam:
        search_webcam(matcher, args)

    console.print()


if __name__ == "__main__":
    main()
