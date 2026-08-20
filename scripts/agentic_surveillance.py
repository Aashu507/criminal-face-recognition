"""
Agentic Surveillance & RTSP Debugging Engine
============================================
Connects to live RTSP / IP Camera streams with automated network diagnostics,
real-time face tracking, passive anti-spoofing, and AdaFace criminal identification.

Usage:
    python scripts/agentic_surveillance.py --url rtsp://192.168.1.45:8080/h264_pcm.sdp
    python scripts/agentic_surveillance.py --url 0  (for default USB webcam)
"""

import sys
import time
import socket
import argparse
import urllib.parse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import psutil
from rich.console import Console
from rich.panel import Panel

console = Console(force_terminal=True, highlight=False)


def diagnose_network(target_url: str):
    """Diagnoses subnet mismatch, TCP socket reachability, and RTSP ports."""
    console.print(Panel.fit("[bold cyan][STEP 1] Network & RTSP Connectivity Diagnostics[/bold cyan]"))

    if str(target_url).isdigit():
        console.print(f"[green][+] Local USB Camera index '{target_url}' selected -- skipping network probe.[/green]")
        return True, target_url

    parsed = urllib.parse.urlparse(target_url)
    hostname = parsed.hostname or "127.0.0.1"
    port = parsed.port or 554

    # Host IP check
    host_ips = []
    try:
        for iface, addrs in psutil.net_if_addrs().items():
            for addr in addrs:
                if addr.family == socket.AF_INET and not addr.address.startswith("127."):
                    host_ips.append((iface, addr.address))
    except Exception:
        pass

    console.print(f"[*] Target Camera Host: [bold yellow]{hostname}:{port}[/bold yellow]")
    console.print(f"[*] Active Local Host Interfaces:")
    for iface, ip in host_ips:
        console.print(f"    - {iface}: [bold green]{ip}[/bold green]")

    # Check for subnet mismatch
    if host_ips:
        primary_ip = host_ips[0][1]
        host_sub = ".".join(primary_ip.split(".")[:3])
        target_sub = ".".join(hostname.split(".")[:3])
        if host_sub != target_sub and not hostname.startswith("127."):
            console.print(
                f"[yellow][!] Warning: Subnet mismatch detected![/yellow]\n"
                f"    Your PC is on [cyan]{primary_ip}[/cyan] ({host_sub}.x)\n"
                f"    Target camera is on [cyan]{hostname}[/cyan] ({target_sub}.x)\n"
                f"    Ensure your smartphone/camera is connected to the EXACT same Wi-Fi / Hotspot."
            )

    # Socket probe
    console.print(f"[*] Testing TCP socket reachability to {hostname}:{port}...")
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(1.5)
    is_reachable = False
    try:
        res = sock.connect_ex((hostname, port))
        if res == 0:
            console.print(f"[green][+] Port {port} on {hostname} is OPEN and REACHABLE![/green]")
            is_reachable = True
        else:
            console.print(f"[red][-] Port {port} on {hostname} is UNREACHABLE (Code: {res}).[/red]")
    except Exception as e:
        console.print(f"[red][-] Socket error: {e}[/red]")
    finally:
        sock.close()

    return is_reachable, target_url


def run_agentic_surveillance(stream_url: str, threshold: float = 0.50, display_window: bool = True):
    """Runs live agentic AI surveillance loop with tracking and identification."""
    from core.matcher import FaceMatcher
    from core.tracker import FaceTracker
    from core.anti_spoofing import AntiSpoofingDetector
    from core.alert_dispatcher import AlertDispatcher

    is_reachable, _ = diagnose_network(stream_url)

    console.print(Panel.fit("[bold cyan][STEP 2] Initializing Deep Learning Surveillance Engines[/bold cyan]"))
    matcher = FaceMatcher()
    tracker = FaceTracker()
    spoof_detector = AntiSpoofingDetector()
    dispatcher = AlertDispatcher()

    console.print(Panel.fit(f"[bold cyan][STEP 3] Connecting to Live Feed: {stream_url}[/bold cyan]"))
    
    src = int(stream_url) if str(stream_url).isdigit() else stream_url
    cap = cv2.VideoCapture(src)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        console.print(f"[red][-] Failed to open video stream at {stream_url}.[/red]")
        console.print("[yellow][*] Troubleshooting Checklist:[/yellow]")
        console.print("  1. Verify the camera app (IP Webcam / CCTV) is running and broadcasting.")
        console.print("  2. Verify your PC and phone are on the exact same Wi-Fi / Hotspot.")
        console.print("  3. Alternative URLs for IP Webcam app:")
        console.print("     - RTSP: rtsp://<phone_ip>:8080/h264_pcm.sdp")
        console.print("     - MJPEG: http://<phone_ip>:8080/video")
        console.print("     - Snapshot: http://<phone_ip>:8080/shot.jpg")
        return

    console.print("[green][+] Live stream connected successfully! Press 'Q' or Ctrl+C to terminate.[/green]\n")

    frame_idx = 0
    t_start = time.perf_counter()

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                console.print("[yellow][!] Frame drop / stream interrupted, waiting...[/yellow]")
                time.sleep(0.5)
                continue

            frame_idx += 1
            t0 = time.perf_counter()

            # 1. Search faces in current frame with AdaFace
            results = matcher.search_image(frame, top_k=1, use_adaface=True)
            detected_faces = [r["face"] for r in results]

            # 2. Update multi-target tracker
            active_tracks = tracker.update(detected_faces, current_time=time.time())

            # 3. Process matches & anti-spoofing for each face
            annotated_frame = frame.copy()
            for r in results:
                face = r["face"]
                box = [int(v) for v in face.bbox]
                matches = r["matches"]
                quality = r.get("quality_score", 0.5)

                # Anti-spoofing
                face_crop = frame[max(0, box[1]):min(frame.shape[0], box[2]), max(0, box[0]):min(frame.shape[1], box[2])]
                is_live, liveness_score, _ = spoof_detector.evaluate_liveness(face_crop)

                if matches and matches[0]["similarity"] >= threshold:
                    top_match = matches[0]
                    name = top_match["name"]
                    cid = top_match["id"]
                    sim = top_match["similarity"]

                    # Alert dispatch
                    dispatcher.dispatch_alert(
                        criminal_id=cid,
                        criminal_name=name,
                        similarity=sim,
                        camera_id="RTSP-AGENT",
                        face_crop_bgr=face_crop
                    )

                    label = f"WANTED: {name} ({sim*100:.1f}%) [Q:{quality:.2f}|Live:{liveness_score:.2f}]"
                    color = (0, 0, 255)  # Red
                else:
                    live_str = "LIVE" if is_live else "SPOOF"
                    label = f"PERSON [Q:{quality:.2f}|{live_str}]"
                    color = (0, 255, 0) if is_live else (0, 165, 255)  # Green or Orange

                # Draw bounding box & badge
                cv2.rectangle(annotated_frame, (box[0], box[1]), (box[2], box[3]), color, 2)
                cv2.putText(annotated_frame, label, (box[0], max(20, box[1] - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            latency_ms = (time.perf_counter() - t0) * 1000
            current_fps = frame_idx / (time.perf_counter() - t_start + 1e-6)

            # Telemetry banner
            cv2.putText(annotated_frame, f"FPS: {current_fps:.1f} | Latency: {latency_ms:.1f}ms | Active Tracks: {len(active_tracks)}",
                        (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            if frame_idx % 30 == 0:
                console.print(
                    f"Frame #{frame_idx:05d} | "
                    f"FPS: [bold green]{current_fps:.1f}[/bold green] | "
                    f"Latency: [bold cyan]{latency_ms:.1f} ms[/bold cyan] | "
                    f"Detected Faces: {len(results)} | "
                    f"Active Tracks: {len(active_tracks)}"
                )

            if display_window:
                cv2.imshow("Agentic Surveillance - RTSP Live Debugger", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    except KeyboardInterrupt:
        console.print("\n[yellow][*] Surveillance session terminated by operator.[/yellow]")
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Agentic Surveillance & RTSP Debugger")
    parser.add_argument("--url", type=str, default="rtsp://192.168.1.45:8080/h264_pcm.sdp", help="RTSP / HTTP / Video URL or webcam index")
    parser.add_argument("--threshold", type=float, default=0.50, help="Criminal similarity threshold")
    parser.add_argument("--no-display", action="store_true", help="Run in headless background mode")
    args = parser.parse_args()

    run_agentic_surveillance(args.url, threshold=args.threshold, display_window=not args.no_display)
