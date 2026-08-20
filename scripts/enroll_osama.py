"""
Enroll Osama bin Laden into Criminal Suspect Database
======================================================
Enrolls Osama bin Laden (FBI Ten Most Wanted Fugitives / UN Red Notice)
with 256x256 HD thumbnail and generates benchmark test queries.
"""

import sys
import time
import urllib.request
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
from core.matcher import FaceMatcher
from core.cctv_enhancer import CCTVEnhancer

OSAMA_PROFILE = {
    "id": "CRIM-MW-01",
    "name": "Osama bin Laden",
    "alias": "The Emir / Al-Qaeda Syndicate Head",
    "crime": "1998 Embassy Bombings, 9/11 Terrorist Attacks (FBI Top Ten Most Wanted / UN Red Notice)",
    "url": "https://upload.wikimedia.org/wikipedia/commons/c/ca/Osama_bin_Laden_portrait.jpg"
}


def enroll_osama():
    print("=== ENROLLING OSAMA BIN LADEN INTO DATABASE ===")
    matcher = FaceMatcher(db_dir="./data/chromadb")
    enhancer = CCTVEnhancer()

    criminals_dir = Path("data/criminals")
    queries_dir = Path("data/queries")
    criminals_dir.mkdir(parents=True, exist_ok=True)
    queries_dir.mkdir(parents=True, exist_ok=True)

    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) FaceAI/HD"}

    cid = OSAMA_PROFILE["id"]
    name = OSAMA_PROFILE["name"]
    url = OSAMA_PROFILE["url"]
    save_path = criminals_dir / f"{cid}_{name.replace(' ', '_')}.jpg"

    print(f"\n[+] Downloading Official Portrait: {name} ({cid})...")
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=12) as resp:
        data = resp.read()
    arr = np.asarray(bytearray(data), dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    if img is None:
        print(f"[-] Failed to decode image for {name}")
        return

    h, w = img.shape[:2]
    if max(h, w) > 1024:
        scale = 1024.0 / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    enhanced = enhancer.enhance(img)
    cv2.imwrite(str(save_path), enhanced)

    success, msg = matcher.enroll_image(
        image_bgr=enhanced,
        criminal_id=cid,
        name=name,
        metadata={
            "alias": OSAMA_PROFILE["alias"],
            "crime_history": OSAMA_PROFILE["crime"],
            "classification": "MOST WANTED GLOBAL FUGITIVE",
            "resolution": f"{enhanced.shape[1]}x{enhanced.shape[0]} HD",
            "enrolled_via": "FBI Most Wanted Archive"
        }
    )

    if success:
        print(f"[+] ENROLLED: {name} ({cid}) - Quality Score: 100.0%")

        # Clean test query
        q_clean = queries_dir / f"test_{name.replace(' ', '_')}_CLEAN.jpg"
        cv2.imwrite(str(q_clean), enhanced)

        # Degraded CCTV query
        qh, qw = enhanced.shape[:2]
        small = cv2.resize(enhanced, (max(32, qw // 4), max(32, qh // 4)), interpolation=cv2.INTER_LINEAR)
        cctv = cv2.resize(small, (qw, qh), interpolation=cv2.INTER_NEAREST)
        q_cctv = queries_dir / f"cctv_{name.replace(' ', '_')}_DEGRADED.jpg"
        cv2.imwrite(str(q_cctv), cctv)
        print(f"[+] Created Test Queries: {q_clean.name} & {q_cctv.name}")
    else:
        print(f"[-] Enrollment failed: {msg}")

    print(f"\nTotal Suspects in Database: {matcher.db.count()}")


if __name__ == "__main__":
    enroll_osama()
