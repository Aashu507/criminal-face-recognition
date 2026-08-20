"""
Demo Registry Seeder
====================
Downloads high-quality public domain test portraits and enrolls them into ChromaDB
so operators can immediately test live camera recognition, CCTV low-res search, and alerts.
"""

import sys
import urllib.request
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
from core.matcher import FaceMatcher

TEST_PROFILES = [
    {
        "id": "CRIM-101",
        "name": "Shah Rukh Khan",
        "alias": "Don / Baadshah",
        "crime": "Interpol Red Notice - Suspect Case #4092 (Armed Robbery)",
        "mugshot_url": "https://upload.wikimedia.org/wikipedia/commons/6/6e/Shah_Rukh_Khan_graces_the_launch_of_the_new_Santro.jpg"
    },
    {
        "id": "CRIM-102",
        "name": "Dr. APJ Abdul Kalam",
        "alias": "Missile Scientist",
        "crime": "VIP High-Security Protected Protocol Persona",
        "mugshot_url": "https://upload.wikimedia.org/wikipedia/commons/b/b0/A._P._J._Abdul_Kalam_in_2008.jpg"
    },
    {
        "id": "CRIM-103",
        "name": "Bill Gates",
        "alias": "Tech Founder",
        "crime": "Economic Offense Syndicate - Section 420 IPC",
        "mugshot_url": "https://upload.wikimedia.org/wikipedia/commons/a/a8/Bill_Gates_2017_%28cropped%29.jpg"
    },
    {
        "id": "CRIM-104",
        "name": "Steve Jobs",
        "alias": "The Visionary",
        "crime": "International Smuggling & Cyber Investigation",
        "mugshot_url": "https://upload.wikimedia.org/wikipedia/commons/d/dc/Steve_Jobs_Headshot_2010-CROP_%28cropped_2%29.jpg"
    },
    {
        "id": "CRIM-105",
        "name": "Barack Obama",
        "alias": "Potus 44",
        "crime": "Diplomatic Security Clearance Benchmark Record",
        "mugshot_url": "https://upload.wikimedia.org/wikipedia/commons/8/8d/President_Barack_Obama.jpg"
    }
]


def seed_database():
    print("[*] Initializing FaceMatcher & ChromaDB...")
    matcher = FaceMatcher()
    
    criminals_dir = Path("data/criminals")
    queries_dir = Path("data/queries")
    criminals_dir.mkdir(parents=True, exist_ok=True)
    queries_dir.mkdir(parents=True, exist_ok=True)

    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) FaceRecognitionBot/1.0"}

    print(f"\n[*] Downloading & Enrolling {len(TEST_PROFILES)} Test Suspect Profiles...")

    for prof in TEST_PROFILES:
        cid = prof["id"]
        name = prof["name"]
        url = prof["mugshot_url"]
        local_path = criminals_dir / f"{cid}_{name.replace(' ', '_')}.jpg"

        print(f"\n[+] Fetching mugshot for '{name}' ({cid})...")
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=10.0) as resp:
                img_bytes = resp.read()
                with open(local_path, "wb") as f:
                    f.write(img_bytes)

            img = cv2.imread(str(local_path))
            if img is None:
                print(f"[-] Failed to decode image for {name}")
                continue

            success, msg = matcher.enroll_image(
                image_bgr=img,
                criminal_id=cid,
                name=name,
                metadata={
                    "alias": prof["alias"],
                    "crime_history": prof["crime"],
                    "enrolled_via": "Seed Script"
                }
            )
            if success:
                print(f"[+] SUCCESS: Enrolled {name} ({cid})")
            else:
                print(f"[-] Failed to enroll {name}: {msg}")

        except Exception as e:
            print(f"[-] Error downloading {url}: {e}")

    total_count = matcher.db.count()
    print(f"\n[+] Seeding complete! Database now has {total_count} enrolled suspect(s).")


if __name__ == "__main__":
    seed_database()
