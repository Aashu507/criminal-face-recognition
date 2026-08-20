"""
HD Studio Mugshot Registry & Quality Enhancer
=============================================
Replaces low-resolution 64x64 chips with crystal-clear, high-definition portraits.
Applies LAB-CLAHE contrast dynamic range, bilateral denoising, and unsharp masking,
then enrolls them with 512-dim ArcFace embeddings and crisp thumbnails.
"""

import sys
import base64
import urllib.request
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
from core.database import FaceDatabase
from core.matcher import FaceMatcher
from core.cctv_enhancer import CCTVEnhancer

STUDIO_PROFILES = [
    {
        "id": "CRIM-IND-01",
        "name": "Shah Rukh Khan",
        "alias": "Don / King Khan",
        "crime": "Section 392 IPC - Armed Robbery Syndicate",
        "local_fallback": "data/criminals/CRIM-IND-01_Shah_Rukh_Khan.jpg",
        "url": "https://upload.wikimedia.org/wikipedia/commons/6/6e/Shah_Rukh_Khan_graces_the_launch_of_the_new_Santro.jpg"
    },
    {
        "id": "CRIM-IND-02",
        "name": "Dr. APJ Abdul Kalam",
        "alias": "Missile Scientist",
        "crime": "VIP Z-Plus Security Clearance Protocol",
        "local_fallback": "data/criminals/CRIM-IND-02_Dr._APJ_Abdul_Kalam.jpg",
        "url": "https://upload.wikimedia.org/wikipedia/commons/b/b0/A._P._J._Abdul_Kalam_in_2008.jpg"
    },
    {
        "id": "CRIM-IND-03",
        "name": "Sachin Tendulkar",
        "alias": "Master Blaster",
        "crime": "Section 420 IPC - Financial Syndicate Investigation",
        "local_fallback": "data/criminals/CRIM-IND-03_Sachin_Tendulkar.jpg",
        "url": "https://upload.wikimedia.org/wikipedia/commons/2/25/Sachin_Tendulkar_at_MRF_Promotion_Event.jpg"
    },
    {
        "id": "CRIM-IND-04",
        "name": "Deepika Padukone",
        "alias": "Padmavati",
        "crime": "Interpol Red Notice - Cross-Border Operation",
        "local_fallback": "data/criminals/CRIM-IND-04_Deepika_Padukone.jpg",
        "url": "https://upload.wikimedia.org/wikipedia/commons/b/b6/Deepika_Padukone_Cannes_2018_%28cropped%29.jpg"
    },
    {
        "id": "CRIM-IND-05",
        "name": "Salman Khan",
        "alias": "Bhaijaan",
        "crime": "Section 304A IPC - High-Speed Hit & Run Incident",
        "local_fallback": "data/criminals/CRIM-IND-05_Salman_Khan.jpg",
        "url": "https://upload.wikimedia.org/wikipedia/commons/8/86/Salman_Khan_at_Renault_Star_Guild_Awards.jpg"
    },
    {
        "id": "CRIM-103",
        "name": "Bill Gates",
        "alias": "Microsoft Founder",
        "crime": "Global Financial Audit & Corporate Antitrust",
        "local_fallback": "data/criminals/CRIM-103_Bill_Gates.jpg",
        "url": "https://upload.wikimedia.org/wikipedia/commons/a/a8/Bill_Gates_2017_%28cropped%29.jpg"
    },
    {
        "id": "CRIM-104",
        "name": "Steve Jobs",
        "alias": "Apple Pioneer",
        "crime": "Federal Cyber & Corporate Investigation Case",
        "local_fallback": "data/criminals/CRIM-104_Steve_Jobs.jpg",
        "url": "https://upload.wikimedia.org/wikipedia/commons/d/dc/Steve_Jobs_Headshot_2010-CROP_%28cropped_2%29.jpg"
    },
    {
        "id": "CRIM-105",
        "name": "Barack Obama",
        "alias": "44th President",
        "crime": "Interpol Red Notice - High Value Diplomatic Protection",
        "local_fallback": "data/criminals/CRIM-105_Barack_Obama.jpg",
        "url": "https://upload.wikimedia.org/wikipedia/commons/8/8d/President_Barack_Obama.jpg"
    }
]


def enhance_and_rebuild():
    print("=== REBUILDING CRYSTAL-CLEAR HD SUSPECT REGISTRY ===")
    
    db = FaceDatabase(persist_dir="./data/chromadb")
    db.clear()
    matcher = FaceMatcher(db_dir="./data/chromadb")
    enhancer = CCTVEnhancer()

    criminals_dir = Path("data/criminals")
    queries_dir = Path("data/queries")
    criminals_dir.mkdir(parents=True, exist_ok=True)
    queries_dir.mkdir(parents=True, exist_ok=True)

    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) FaceAI/HD"}
    
    print(f"\n[*] Processing and Enhancing {len(STUDIO_PROFILES)} HD Suspect Profiles...")

    for prof in STUDIO_PROFILES:
        cid = prof["id"]
        name = prof["name"]
        url = prof["url"]
        fname = f"{cid}_{name.replace(' ', '_')}.jpg"
        save_path = criminals_dir / fname

        print(f"\n[+] Processing Studio Portrait: {name} ({cid})...")
        img = None

        # 1. Try local file first
        local_p = Path(prof.get("local_fallback", ""))
        if local_p.exists():
            img = cv2.imread(str(local_p))

        # 2. Try online download if local not available
        if img is None:
            try:
                req = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(req, timeout=10) as resp:
                    data = resp.read()
                    arr = np.asarray(bytearray(data), dtype=np.uint8)
                    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            except Exception as e:
                print(f"  [-] Online fetch error: {e}")

        if img is None:
            print(f"  [-] Failed to load portrait for {name}")
            continue

        # Standardize & Enhance for maximum clarity
        h, w = img.shape[:2]
        if max(h, w) > 1024:
            scale = 1024.0 / max(h, w)
            img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

        # Apply optical clarity enhancement: mild unsharp mask & bilateral smoothing
        enhanced = enhancer.enhance(img)
        cv2.imwrite(str(save_path), enhanced)

        # Enroll into database
        success, msg = matcher.enroll_image(
            image_bgr=enhanced,
            criminal_id=cid,
            name=name,
            metadata={
                "alias": prof["alias"],
                "crime_history": prof["crime"],
                "resolution": f"{enhanced.shape[1]}x{enhanced.shape[0]} HD",
                "enrolled_via": "HD Studio Enhancer"
            }
        )

        if success:
            print(f"  [+] ENROLLED HD: {name} ({cid}) - Resolution: {enhanced.shape[1]}x{enhanced.shape[0]}")
            # Generate clean query & CCTV degraded test query
            q_clean = queries_dir / f"test_{name.replace(' ', '_')}_CLEAN.jpg"
            cv2.imwrite(str(q_clean), enhanced)

            # CCTV degraded query
            qh, qw = enhanced.shape[:2]
            small = cv2.resize(enhanced, (max(32, qw // 4), max(32, qh // 4)), interpolation=cv2.INTER_LINEAR)
            cctv = cv2.resize(small, (qw, qh), interpolation=cv2.INTER_NEAREST)
            q_cctv = queries_dir / f"cctv_{name.replace(' ', '_')}_DEGRADED.jpg"
            cv2.imwrite(str(q_cctv), cctv)
        else:
            print(f"  [-] Enrollment failed: {msg}")

    print(f"\n[+] HD Registry Setup Complete! Total enrolled suspects: {matcher.db.count()}")


if __name__ == "__main__":
    enhance_and_rebuild()
