"""
Most Wanted Criminals Ingestion Engine
======================================
Downloads and enrolls world-renowned historical most-wanted suspects and cartel/syndicate figures
into ChromaDB with high-definition thumbnails and generates test benchmark queries.
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

MOST_WANTED_PROFILES = [
    {
        "id": "CRIM-MW-01",
        "name": "Al Capone",
        "alias": "Scarface / Chicago Outfit Boss",
        "crime": "Syndicate Racketeering, National Prohibition Offenses, Tax Evasion (FBI Top Target)",
        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e1/Al_Capone_in_1930_%28manually_dusted%29.jpg/960px-Al_Capone_in_1930_%28manually_dusted%29.jpg"
    },
    {
        "id": "CRIM-MW-02",
        "name": "Pablo Escobar",
        "alias": "El Patron / Medellin Kingpin",
        "crime": "Medellin Cartel Leader, Narco-Terrorism, Transnational Trafficking (Interpol Red Notice)",
        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/b/be/Pablo_Escobar_Mug_%28cropped%29%28b%29.jpg/960px-Pablo_Escobar_Mug_%28cropped%29%28b%29.jpg"
    },
    {
        "id": "CRIM-MW-03",
        "name": "Dawood Ibrahim",
        "alias": "D-Company Syndicate Head",
        "crime": "1993 Mumbai Blasts, Inter-State Organized Crime, Global Terrorist (UN / NIA Red Notice)",
        "url": "https://upload.wikimedia.org/wikipedia/commons/4/4b/Dawood_ibrahim.png"
    },
    {
        "id": "CRIM-MW-04",
        "name": "Veerappan",
        "alias": "Forest Brigand / Sandalwood Smuggler",
        "crime": "Armed Banditry, Kidnapping for Ransom, Wildlife Poaching Syndicate (STF Special Case)",
        "url": "https://upload.wikimedia.org/wikipedia/commons/5/55/Veerappan%2C_Chasing_the_Brigand_%28Book%2C_2017%29_01_%28cropped%29.jpg"
    },
    {
        "id": "CRIM-MW-05",
        "name": "Ted Bundy",
        "alias": "The Campus Killer",
        "crime": "Multi-State Serial Homicide & Kidnapping Syndicate (FBI Top 10 Most Wanted)",
        "url": "https://upload.wikimedia.org/wikipedia/commons/c/cc/Ted_Bundy_headshot.jpg"
    }
]


def ingest_most_wanted():
    print("=== ENROLLING MOST WANTED CRIMINALS INTO DATABASE ===")
    matcher = FaceMatcher(db_dir="./data/chromadb")
    enhancer = CCTVEnhancer()

    criminals_dir = Path("data/criminals")
    queries_dir = Path("data/queries")
    criminals_dir.mkdir(parents=True, exist_ok=True)
    queries_dir.mkdir(parents=True, exist_ok=True)

    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) FaceAI/HD"}
    enrolled_count = 0

    for prof in MOST_WANTED_PROFILES:
        cid = prof["id"]
        name = prof["name"]
        url = prof["url"]
        fname = f"{cid}_{name.replace(' ', '_')}.jpg"
        save_path = criminals_dir / fname

        print(f"\n[+] Downloading Most Wanted Mugshot: {name} ({cid})...")
        time.sleep(1.0)
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=12) as resp:
                data = resp.read()
            arr = np.asarray(bytearray(data), dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

            if img is None:
                print(f"  [-] Failed to decode image for {name}")
                continue

            # Standardize resolution and enhance
            h, w = img.shape[:2]
            if max(h, w) > 1024:
                scale = 1024.0 / max(h, w)
                img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

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
                    "classification": "MOST WANTED FUGITIVE",
                    "enrolled_via": "Official Most Wanted Ingestion"
                }
            )

            if success:
                enrolled_count += 1
                print(f"  [+] ENROLLED: {name} ({cid}) - Quality Score: 100.0%")

                # Generate clean test query
                q_clean = queries_dir / f"test_{name.replace(' ', '_')}_CLEAN.jpg"
                cv2.imwrite(str(q_clean), enhanced)

                # Generate degraded CCTV simulation query
                qh, qw = enhanced.shape[:2]
                small = cv2.resize(enhanced, (max(32, qw // 4), max(32, qh // 4)), interpolation=cv2.INTER_LINEAR)
                cctv = cv2.resize(small, (qw, qh), interpolation=cv2.INTER_NEAREST)
                q_cctv = queries_dir / f"cctv_{name.replace(' ', '_')}_DEGRADED.jpg"
                cv2.imwrite(str(q_cctv), cctv)
                print(f"  [+] Generated Test Queries: {q_clean.name} & {q_cctv.name}")
            else:
                print(f"  [-] Enrollment failed: {msg}")

        except Exception as e:
            print(f"  [-] Error processing {name}: {e}")

    print(f"\n=== ENROLLMENT SUMMARY ===")
    print(f"New Most Wanted Criminals Added: {enrolled_count}")
    print(f"Total Enrolled Suspects in ChromaDB: {matcher.db.count()}")


if __name__ == "__main__":
    ingest_most_wanted()
