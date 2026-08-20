"""
Clean Indian & Global Criminal Face Registry Seeder
===================================================
Seeds the official criminal database with high-resolution Indian suspect profiles
and generates corresponding CCTV test queries in data/queries/.
"""

import sys
import urllib.request
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
from core.matcher import FaceMatcher
from core.database import FaceDatabase

PROFILES = [
    {
        "id": "CRIM-IND-01",
        "name": "Shah Rukh Khan",
        "alias": "Don / Baadshah",
        "crime": "Section 392 IPC - Armed Robbery Syndicate",
        "url": "https://upload.wikimedia.org/wikipedia/commons/6/6e/Shah_Rukh_Khan_graces_the_launch_of_the_new_Santro.jpg"
    },
    {
        "id": "CRIM-IND-02",
        "name": "Dr. APJ Abdul Kalam",
        "alias": "Missile Scientist",
        "crime": "VIP Z-Plus Security Clearance Protocol",
        "url": "https://upload.wikimedia.org/wikipedia/commons/b/b0/A._P._J._Abdul_Kalam_in_2008.jpg"
    },
    {
        "id": "CRIM-IND-03",
        "name": "Sachin Tendulkar",
        "alias": "Master Blaster",
        "crime": "Section 420 IPC - Financial Syndicate Investigation",
        "url": "https://upload.wikimedia.org/wikipedia/commons/2/25/Sachin_Tendulkar_at_MRF_Promotion_Event.jpg"
    },
    {
        "id": "CRIM-IND-04",
        "name": "Deepika Padukone",
        "alias": "Padmavati",
        "crime": "Interpol Red Notice - Cross-Border Operation",
        "url": "https://upload.wikimedia.org/wikipedia/commons/b/b6/Deepika_Padukone_Cannes_2018_%28cropped%29.jpg"
    },
    {
        "id": "CRIM-IND-05",
        "name": "Salman Khan",
        "alias": "Bhaijaan",
        "crime": "Section 304A IPC - High-Speed Hit & Run Incident",
        "url": "https://upload.wikimedia.org/wikipedia/commons/8/86/Salman_Khan_at_Renault_Star_Guild_Awards.jpg"
    }
]


def reseed_clean():
    print("=== INITIALIZING CLEAN CRIMINAL REGISTRY ===")
    
    # 1. Clear old records
    db = FaceDatabase(persist_dir="./data/chromadb")
    db.clear()
    matcher = FaceMatcher(db_dir="./data/chromadb")
    
    criminals_dir = Path("data/criminals")
    queries_dir = Path("data/queries")
    criminals_dir.mkdir(parents=True, exist_ok=True)
    queries_dir.mkdir(parents=True, exist_ok=True)

    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) FaceAI/1.0"}

    print(f"\n[*] Enrolling {len(PROFILES)} Verified Suspect Profiles...")

    for prof in PROFILES:
        cid = prof["id"]
        name = prof["name"]
        url = prof["url"]
        fname = f"{cid}_{name.replace(' ', '_')}.jpg"
        mugshot_path = criminals_dir / fname

        print(f"\n[+] Fetching {name} ({cid})...")
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=10) as r:
                data = r.read()
                with open(mugshot_path, "wb") as f:
                    f.write(data)

            img = cv2.imread(str(mugshot_path))
            if img is None:
                print(f"[-] Could not decode image for {name}")
                continue

            success, msg = matcher.enroll_image(
                image_bgr=img,
                criminal_id=cid,
                name=name,
                metadata={
                    "alias": prof["alias"],
                    "crime_history": prof["crime"],
                    "enrolled_via": "Official Clean Seeder"
                }
            )
            if success:
                print(f"  [+] ENROLLED: {name} ({cid})")
                
                # Also generate clear query + degraded CCTV query
                q_clean = queries_dir / f"test_{name.replace(' ', '_')}_CLEAN.jpg"
                cv2.imwrite(str(q_clean), img)

                h, w = img.shape[:2]
                small = cv2.resize(img, (max(32, w // 4), max(32, h // 4)), interpolation=cv2.INTER_LINEAR)
                cctv = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
                q_cctv = queries_dir / f"cctv_{name.replace(' ', '_')}_DEGRADED.jpg"
                cv2.imwrite(str(q_cctv), cctv)
                print(f"  [+] Generated Test Queries: {q_clean.name} & {q_cctv.name}")
            else:
                print(f"  [-] Failed: {msg}")

        except Exception as e:
            print(f"  [-] Error: {e}")

    # Also make sure negative control (non-criminal) exists in data/queries
    einstein_path = queries_dir / "test_Einstein_NON_CRIMINAL.jpg"
    if not einstein_path.exists():
        try:
            req = urllib.request.Request(
                "https://upload.wikimedia.org/wikipedia/commons/3/3e/Einstein_1921_by_F_Schmutzer_-_restoration.jpg",
                headers=headers
            )
            with urllib.request.urlopen(req, timeout=10) as r:
                with open(einstein_path, "wb") as f:
                    f.write(r.read())
        except Exception:
            pass

    print(f"\n[+] Database setup complete! Total enrolled suspects: {matcher.db.count()}")


if __name__ == "__main__":
    reseed_clean()
