"""
Indian Demographic Face Dataset Ingestion Engine
================================================
Ingests Indian demographic face datasets from online repository (Hugging Face / Kaggle mirrors),
automatically generates Indian Law Enforcement case profiles (IPC sections / crime tags),
and enrolls their neural embeddings directly into ChromaDB.
"""

import sys
import time
import json
import urllib.request
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
from core.matcher import FaceMatcher

CRIME_CLASSIFICATIONS = [
    ("IPC-302", "Homicide & Syndicate Contract - Red Notice Case"),
    ("IPC-392", "Armed Highway Robbery & Gang Syndicate"),
    ("IPC-420", "Financial Fraud & Cyber Extortion Operation"),
    ("IPC-379", "Organized Vehicle Theft & Smuggling Network"),
    ("IPC-120B", "Criminal Conspiracy & Inter-State Cross-Border Syndicate"),
    ("IPC-364A", "Kidnapping for Ransom - High Risk Target"),
    ("IT-66", "Nationwide Cyber Threat & Critical Infrastructure Attack"),
    ("NDPS-21", "Commercial Narcotic Smuggling & Trafficking Ring"),
]

SELECTED_IDENTITIES = [
    "Aamir_Khan", "Abhishek_Bachchan", "Aishwarya_Rai", "Ajay_Devgn",
    "Akshay_Kumar", "Alia_Bhatt", "Amitabh_Bachchan", "Anil_Kapoor",
    "Anushka_Sharma", "Deepika_Padukone", "Disha_Patani", "Emraan_Hashmi",
    "Govinda", "Hrithik_Roshan", "Irrfan_Khan", "John_Abraham",
    "Kareena_Kapoor", "Katrina_Kaif", "Ranbir_Kapoor", "Ranveer_Singh"
]


def ingest_dataset():
    print("=== INGESTING ONLINE INDIAN FACE DATASET ===")
    matcher = FaceMatcher()
    
    criminals_dir = Path("data/criminals")
    queries_dir = Path("data/queries")
    criminals_dir.mkdir(parents=True, exist_ok=True)
    queries_dir.mkdir(parents=True, exist_ok=True)

    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) IndianSurveillance/1.0"}

    total_enrolled = 0
    total_queries_created = 0

    for idx, folder_name in enumerate(SELECTED_IDENTITIES, 1):
        person_name = folder_name.replace("_", " ")
        cid = f"CRIM-IND-{200 + idx}"
        crime_sec, crime_desc = CRIME_CLASSIFICATIONS[idx % len(CRIME_CLASSIFICATIONS)]

        print(f"\n[{idx}/{len(SELECTED_IDENTITIES)}] Processing Indian Identity: {person_name} ({cid})...")

        # 1. Fetch file list from dataset API
        api_url = f"https://huggingface.co/api/datasets/VashuTheGreat2/bollywood_celeb_faces/tree/main/bollywood_celeb_faces/bollywood_celeb_faces/{folder_name}"
        try:
            req = urllib.request.Request(api_url, headers=headers)
            with urllib.request.urlopen(req, timeout=10) as resp:
                files_tree = json.loads(resp.read().decode())
        except Exception as e:
            print(f"  [-] Failed to list folder for {person_name}: {e}")
            continue

        jpg_files = [f["path"] for f in files_tree if f["path"].lower().endswith((".jpg", ".png", ".jpeg"))]
        if not jpg_files:
            print(f"  [-] No images found for {person_name}")
            continue

        # Use 1st image for Registry Enrollment (Mugshot)
        mugshot_path = jpg_files[0]
        mugshot_url = f"https://huggingface.co/datasets/VashuTheGreat2/bollywood_celeb_faces/resolve/main/{mugshot_path}"
        local_mugshot = criminals_dir / f"{cid}_{folder_name}_mugshot.jpg"

        try:
            req = urllib.request.Request(mugshot_url, headers=headers)
            with urllib.request.urlopen(req, timeout=10) as r:
                data = r.read()
                with open(local_mugshot, "wb") as f:
                    f.write(data)

            img = cv2.imread(str(local_mugshot))
            if img is not None:
                success, msg = matcher.enroll_image(
                    image_bgr=img,
                    criminal_id=cid,
                    name=person_name,
                    metadata={
                        "alias": f"Alias-{folder_name[:4]}",
                        "crime_history": f"{crime_sec}: {crime_desc}",
                        "enrolled_via": "Online Indian Face Dataset"
                    }
                )
                if success:
                    total_enrolled += 1
                    print(f"  [+] ENROLLED: {person_name} ({cid}) - {crime_sec}")
                else:
                    print(f"  [-] Enrollment failed: {msg}")

        except Exception as e:
            print(f"  [-] Error downloading mugshot: {e}")

        # Use 2nd image for Query/Test set (Different pose/capture)
        if len(jpg_files) > 1:
            query_path = jpg_files[1]
            query_url = f"https://huggingface.co/datasets/VashuTheGreat2/bollywood_celeb_faces/resolve/main/{query_path}"
            local_query = queries_dir / f"test_{folder_name}_query.jpg"

            try:
                req = urllib.request.Request(query_url, headers=headers)
                with urllib.request.urlopen(req, timeout=10) as r:
                    data = r.read()
                    with open(local_query, "wb") as f:
                        f.write(data)

                q_img = cv2.imread(str(local_query))
                if q_img is not None:
                    # Also create a CCTV degraded version
                    h, w = q_img.shape[:2]
                    small = cv2.resize(q_img, (max(16, w // 3), max(16, h // 3)), interpolation=cv2.INTER_LINEAR)
                    cctv_grain = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
                    cctv_path = queries_dir / f"cctv_degraded_{folder_name}_query.jpg"
                    cv2.imwrite(str(cctv_path), cctv_grain)

                    total_queries_created += 2
                    print(f"  [+] Created Test Queries: test_{folder_name}_query.jpg & cctv_degraded_{folder_name}_query.jpg")

            except Exception as e:
                print(f"  [-] Error downloading query image: {e}")

        time.sleep(0.1)

    print(f"\n=== INGESTION SUMMARY ===")
    print(f"Total New Indian Suspects Enrolled: {total_enrolled}")
    print(f"Total Benchmark Test Queries Generated: {total_queries_created}")
    print(f"Total Registry Count in ChromaDB: {matcher.db.count()}")


if __name__ == "__main__":
    ingest_dataset()
